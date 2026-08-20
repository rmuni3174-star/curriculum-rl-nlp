# curriculum-rl-nlp
Dynamic Curriculum Generation via Reinforcement Learning for Data-Efficient NLP
# ================= DAY 1 (GPT-2 VERSION): Setup + Baseline Training =================

from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT_DIR = '/content/drive/MyDrive/curriculum_rl_project_gpt2'
os.makedirs(PROJECT_DIR, exist_ok=True)
print(f"Project folder ready at: {PROJECT_DIR}")

!pip install -q -U datasets huggingface_hub fsspec transformers accelerate scikit-learn

import torch
import numpy as np
import random
import json
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= Load and subsample SST-2 =================
print("Loading SST-2...")
dataset = load_dataset("stanfordnlp/sst2")

full_train = dataset["train"].shuffle(seed=SEED)
val_data = dataset["validation"]

POOL_SIZE = 8000
train_pool = full_train.select(range(POOL_SIZE))

print(f"Training pool size: {len(train_pool)}")
print(f"Validation size: {len(val_data)}")

# ================= Tokenizer (GPT-2 specific setup) =================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# GPT-2 has no pad token by default — this is required or training will crash
tokenizer.pad_token = tokenizer.eos_token

text_col = "sentence" if "sentence" in train_pool.column_names else "text"

def tokenize_fn(batch):
    return tokenizer(batch[text_col], padding="max_length", truncation=True, max_length=64)

train_pool_tok = train_pool.map(tokenize_fn, batched=True)
val_data_tok = val_data.map(tokenize_fn, batched=True)

train_pool_tok = train_pool_tok.rename_column("label", "labels")
val_data_tok = val_data_tok.rename_column("label", "labels")

keep_cols = ["input_ids", "attention_mask", "labels"]
train_pool_tok = train_pool_tok.remove_columns(
    [c for c in train_pool_tok.column_names if c not in keep_cols]
)
val_data_tok = val_data_tok.remove_columns(
    [c for c in val_data_tok.column_names if c not in keep_cols]
)

train_pool_tok.set_format(type="torch", columns=keep_cols)
val_data_tok.set_format(type="torch", columns=keep_cols)

# ================= Metric function =================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# ================= Helper: train a GPT-2 student on a given subset =================
def train_student(train_subset, budget_name, epochs=3):
    print(f"\n{'='*50}")
    print(f"Training baseline on budget: {budget_name} ({len(train_subset)} examples)")
    print(f"{'='*50}")

    model = GPT2ForSequenceClassification.from_pretrained(
        "gpt2", num_labels=2
    ).to(device)
    # GPT-2 classification head also needs the pad token id set
    model.config.pad_token_id = tokenizer.pad_token_id

    args = TrainingArguments(
        output_dir=f"/content/results_{budget_name}",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,   # smaller batch than DistilBERT — GPT-2 uses more memory
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_subset,
        eval_dataset=val_data_tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    print(f"Final validation accuracy ({budget_name}): {eval_result['eval_accuracy']:.4f}")
    return eval_result["eval_accuracy"], model

# ================= Run baselines at multiple data budgets =================
BUDGETS = [1000, 2000, 4000, 8000]
baseline_results = {}

for budget in BUDGETS:
    subset = train_pool_tok.select(range(budget))
    acc, _ = train_student(subset, f"random_{budget}")
    baseline_results[budget] = acc
    torch.cuda.empty_cache()

print("\n\n================= DAY 1 BASELINE SUMMARY (GPT-2) =================")
for budget, acc in baseline_results.items():
    print(f"Random-order training | Budget: {budget:5d} examples | Val Accuracy: {acc:.4f}")

with open(f"{PROJECT_DIR}/day1_baseline_results.json", "w") as f:
    json.dump(baseline_results, f, indent=2)

train_pool_tok.save_to_disk(f"{PROJECT_DIR}/train_pool_tok")
val_data_tok.save_to_disk(f"{PROJECT_DIR}/val_data_tok")
print(f"\nSaved everything to: {PROJECT_DIR}")
DAY 2 Training
# ================= DAY 2 (IMPROVED v2): RL Curriculum Agent =================

from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT_DIR = '/content/drive/MyDrive/curriculum_rl_project_gpt2'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import copy

from datasets import load_from_disk
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print("Loading Day 1 outputs from Google Drive...")
train_pool_tok = load_from_disk(f"{PROJECT_DIR}/train_pool_tok")
val_data_tok = load_from_disk(f"{PROJECT_DIR}/val_data_tok")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

keep_cols = ["input_ids", "attention_mask", "labels"]
train_pool_tok.set_format(type="torch", columns=keep_cols)
val_data_tok.set_format(type="torch", columns=keep_cols)

with open(f"{PROJECT_DIR}/day1_baseline_results.json") as f:
    baseline_results = json.load(f)
print("Loaded Day 1 baseline results:", baseline_results)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ================= STEP 1: Warm-start base model =================
print("\n" + "="*60)
print("STEP 1: Training warm-start base model")
print("="*60)

warmstart_model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2).to(device)
warmstart_model.config.pad_token_id = tokenizer.pad_token_id

warmstart_subset = train_pool_tok.select(range(1000))

warmstart_args = TrainingArguments(
    output_dir="/content/warmstart_results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_strategy="no",
    eval_strategy="no",
    logging_steps=50,
    report_to="none",
    seed=SEED,
)

warmstart_trainer = Trainer(model=warmstart_model, args=warmstart_args, train_dataset=warmstart_subset)
warmstart_trainer.train()

warmstart_state_dict = copy.deepcopy(warmstart_model.state_dict())
print("Warm-start model ready.")

# ================= STEP 2: Difficulty Scoring =================
print("\n" + "="*60)
print("STEP 2: Scoring example difficulty")
print("="*60)

warmstart_model.eval()
all_losses = []
batch_size = 16
loss_fn = nn.CrossEntropyLoss(reduction="none")

with torch.no_grad():
    for i in range(0, len(train_pool_tok), batch_size):
        batch = train_pool_tok[i:i+batch_size]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = warmstart_model(input_ids=input_ids, attention_mask=attention_mask)
        losses = loss_fn(outputs.logits, labels)
        all_losses.extend(losses.cpu().numpy().tolist())

all_losses = np.array(all_losses)
print(f"Difficulty score range: min={all_losses.min():.3f}, max={all_losses.max():.3f}, mean={all_losses.mean():.3f}")

NUM_TIERS = 4
tier_edges = np.percentile(all_losses, [25, 50, 75])
tier_labels = np.digitize(all_losses, tier_edges)
tier_indices = {t: np.where(tier_labels == t)[0].tolist() for t in range(NUM_TIERS)}
for t in range(NUM_TIERS):
    name = 'easy' if t==0 else 'hard' if t==NUM_TIERS-1 else 'medium'
    print(f"Tier {t} ({name}): {len(tier_indices[t])} examples")

del warmstart_model
torch.cuda.empty_cache()

# ================= STEP 3: REINFORCE RL Curriculum Agent (v2 — improved) =================
print("\n" + "="*60)
print("STEP 3: Training RL curriculum agent (v2)")
print("="*60)

class CurriculumPolicy(nn.Module):
    def __init__(self, state_dim=4, num_tiers=NUM_TIERS):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 32), nn.ReLU(), nn.Linear(32, num_tiers))

    def forward(self, state):
        return F.softmax(self.net(state), dim=-1)

def get_state(step, max_steps, current_val_acc, prev_val_acc):
    return torch.tensor([
        step / max_steps, current_val_acc, current_val_acc - prev_val_acc, 1.0
    ], dtype=torch.float32).to(device)

def evaluate_student(model, eval_dataset, batch_size=16):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(eval_dataset), batch_size):
            batch = eval_dataset[i:i+batch_size]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct / total

def run_rl_curriculum_episode(data_budget, num_rl_steps=40, batch_per_step=None):
    if batch_per_step is None:
        batch_per_step = max(1, data_budget // num_rl_steps)

    student = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2).to(device)
    student.load_state_dict(warmstart_state_dict)
    student.config.pad_token_id = tokenizer.pad_token_id
    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-5)

    log_probs, rewards = [], []
    prev_val_acc = evaluate_student(student, val_data_tok)
    examples_used = 0
    current_val_acc = prev_val_acc

    for step in range(num_rl_steps):
        state = get_state(step, num_rl_steps, prev_val_acc, prev_val_acc)
        tier_probs = policy(state)
        dist = torch.distributions.Categorical(tier_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        chosen_tier = action.item()
        available = tier_indices[chosen_tier]
        if len(available) == 0:
            chosen_tier = 0
            available = tier_indices[0]

        sample_idx = random.sample(available, min(batch_per_step, len(available)))
        batch = train_pool_tok[sample_idx]

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        student.train()
        outputs = student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        examples_used += len(sample_idx)

        if step % 4 == 0 or step == num_rl_steps - 1:
            current_val_acc = evaluate_student(student, val_data_tok)

        # IMPROVED REWARD: accuracy delta + small bonus for absolute accuracy level
        delta_reward = current_val_acc - prev_val_acc
        level_bonus = current_val_acc * 0.1
        reward = delta_reward + level_bonus
        prev_val_acc = current_val_acc

        log_probs.append(log_prob)
        rewards.append(reward)

    final_acc = evaluate_student(student, val_data_tok)
    return student, log_probs, rewards, final_acc, examples_used

# ================= STEP 4: Train the RL agent — more episodes, faster learning rate =================
policy = CurriculumPolicy().to(device)
policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)

NUM_EPISODES = 35
TARGET_BUDGET = 2000

print(f"Running {NUM_EPISODES} RL episodes at data budget = {TARGET_BUDGET}...")
episode_final_accs = []
best_acc_so_far = 0.0
best_policy_state = None

for ep in range(NUM_EPISODES):
    student, log_probs, rewards, final_acc, used = run_rl_curriculum_episode(TARGET_BUDGET, num_rl_steps=40)

    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.95 * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    if returns.std() > 1e-6:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    policy_loss = torch.stack([-lp * G for lp, G in zip(log_probs, returns)]).sum()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    episode_final_accs.append(final_acc)

    if final_acc > best_acc_so_far:
        best_acc_so_far = final_acc
        best_policy_state = copy.deepcopy(policy.state_dict())

    print(f"Episode {ep+1}/{NUM_EPISODES} | Examples used: {used} | Final val accuracy: {final_acc:.4f} | Best so far: {best_acc_so_far:.4f}")

    del student
    torch.cuda.empty_cache()

print("\n" + "="*60)
print("DAY 2 RL CURRICULUM RESULTS (v2 - IMPROVED)")
print("="*60)
print(f"RL curriculum accuracy at budget {TARGET_BUDGET}: mean={np.mean(episode_final_accs):.4f}, best={np.max(episode_final_accs):.4f}")
print(f"Last 10 episodes average (agent after learning): {np.mean(episode_final_accs[-10:]):.4f}")

baseline_at_budget = baseline_results.get(str(TARGET_BUDGET), baseline_results.get(TARGET_BUDGET))
if baseline_at_budget:
    print(f"Random-order baseline at same budget: {baseline_at_budget:.4f}")
    print(f"Improvement (best episode): {np.max(episode_final_accs) - baseline_at_budget:+.4f}")
    print(f"Improvement (last-10 avg): {np.mean(episode_final_accs[-10:]) - baseline_at_budget:+.4f}")

day2_results = {
    "episode_final_accs": episode_final_accs,
    "target_budget": TARGET_BUDGET,
    "baseline_at_budget": baseline_at_budget,
    "best_accuracy": float(best_acc_so_far),
}
with open(f"{PROJECT_DIR}/day2_rl_results.json", "w") as f:
    json.dump(day2_results, f, indent=2)

# Save the BEST policy found, not just the last one
if best_policy_state is not None:
    torch.save(best_policy_state, f"{PROJECT_DIR}/curriculum_policy_best.pt")
torch.save(policy.state_dict(), f"{PROJECT_DIR}/curriculum_policy_final.pt")
print(f"\nSaved Day 2 results to: {PROJECT_DIR}")
Day 3 task:
# ================= DAY 3: Final Comparison + Graph =================

from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT_DIR = '/content/drive/MyDrive/curriculum_rl_project_gpt2'

import json
import numpy as np
import matplotlib.pyplot as plt

# ================= Load Day 1 and Day 2 results =================
with open(f"{PROJECT_DIR}/day1_baseline_results.json") as f:
    baseline_results = json.load(f)

with open(f"{PROJECT_DIR}/day2_rl_results.json") as f:
    rl_results = json.load(f)

print("Day 1 baseline results:", baseline_results)
print("Day 2 RL results loaded:", {k: v for k, v in rl_results.items() if k != "episode_final_accs"})

episode_accs = rl_results["episode_final_accs"]
target_budget = rl_results["target_budget"]
baseline_at_budget = rl_results["baseline_at_budget"]

# ================= COMPARISON TABLE =================
print("\n" + "="*70)
print("FINAL COMPARISON TABLE")
print("="*70)
print(f"{'Method':<35} {'Data Budget':<15} {'Accuracy':<10}")
print("-"*70)
for budget, acc in baseline_results.items():
    print(f"{'Random-order baseline':<35} {budget:<15} {acc:.4f}")
print(f"{'RL curriculum (mean, 35 ep)':<35} {target_budget:<15} {np.mean(episode_accs):.4f}")
print(f"{'RL curriculum (best episode)':<35} {target_budget:<15} {np.max(episode_accs):.4f}")
print(f"{'RL curriculum (last 10 avg)':<35} {target_budget:<15} {np.mean(episode_accs[-10:]):.4f}")
print("-"*70)

# ================= GRAPH 1: RL learning curve over episodes =================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
episodes_x = list(range(1, len(episode_accs) + 1))
ax1.plot(episodes_x, episode_accs, marker='o', markersize=4, color='#2563eb', label='RL Curriculum (per episode)')
ax1.axhline(y=baseline_at_budget, color='#dc2626', linestyle='--', linewidth=2, label=f'Random Baseline ({baseline_at_budget:.4f})')

# Rolling average to show learning trend
window = 5
if len(episode_accs) >= window:
    rolling_avg = np.convolve(episode_accs, np.ones(window)/window, mode='valid')
    ax1.plot(range(window, len(episode_accs) + 1), rolling_avg, color='#16a34a', linewidth=2, label=f'RL {window}-episode Rolling Avg')

ax1.set_xlabel('RL Episode')
ax1.set_ylabel('Validation Accuracy')
ax1.set_title(f'RL Curriculum Learning Progress (Budget = {target_budget} examples)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ================= GRAPH 2: Accuracy vs Data Budget (main result) =================
ax2 = axes[1]
budgets_sorted = sorted([int(b) for b in baseline_results.keys()])
baseline_accs_sorted = [baseline_results[str(b)] for b in budgets_sorted]

ax2.plot(budgets_sorted, baseline_accs_sorted, marker='s', markersize=8, color='#dc2626',
         linewidth=2, label='Random-order (baseline)')

# Plot RL point(s) at its tested budget
ax2.scatter([target_budget], [np.mean(episode_accs)], color='#2563eb', s=120, zorder=5,
            marker='D', label='RL Curriculum (mean)')
ax2.scatter([target_budget], [np.max(episode_accs)], color='#16a34a', s=120, zorder=5,
            marker='^', label='RL Curriculum (best)')

ax2.set_xlabel('Number of Labeled Training Examples')
ax2.set_ylabel('Validation Accuracy')
ax2.set_title('Data Efficiency: Accuracy vs. Training Data Used')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PROJECT_DIR}/day3_comparison_plot.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\nGraph saved to: {PROJECT_DIR}/day3_comparison_plot.png")

# ================= Save final combined summary =================
day3_summary = {
    "baseline_results": baseline_results,
    "rl_target_budget": target_budget,
    "rl_mean_accuracy": float(np.mean(episode_accs)),
    "rl_best_accuracy": float(np.max(episode_accs)),
    "rl_last10_avg_accuracy": float(np.mean(episode_accs[-10:])),
    "baseline_at_rl_budget": baseline_at_budget,
    "improvement_best": float(np.max(episode_accs) - baseline_at_budget),
    "improvement_mean": float(np.mean(episode_accs) - baseline_at_budget),
}

with open(f"{PROJECT_DIR}/day3_final_summary.json", "w") as f:
    json.dump(day3_summary, f, indent=2)

print(f"\nFinal summary saved to: {PROJECT_DIR}/day3_final_summary.json")
print("\n" + "="*70)
print("DAY 3 COMPLETE — Project evaluation finished.")
print("="*70)
final result:
# ================= LIVE DEMO: Apne khud ke sentences test karein =================

import torch

# ---- Check karein ke trained model session mein maujood hai ya nahi ----
try:
    warmstart_state_dict
    model_available = True
    print("Session mein trained model mil gaya. Demo shuru kar rahe hain.")
except NameError:
    model_available = False
    print("Session mein model nahi mila — Drive se best policy ke saath ek naya student train kar rahe hain...")

from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

if model_available:
    demo_model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2).to(device)
    demo_model.load_state_dict(warmstart_state_dict)
    demo_model.config.pad_token_id = tokenizer.pad_token_id
else:
    # Agar session band ho chuka hai, Drive se ek quick model train kar lein demo ke liye
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_DIR = '/content/drive/MyDrive/curriculum_rl_project_gpt2'

    from datasets import load_from_disk
    from transformers import Trainer, TrainingArguments

    train_pool_tok = load_from_disk(f"{PROJECT_DIR}/train_pool_tok")
    keep_cols = ["input_ids", "attention_mask", "labels"]
    train_pool_tok.set_format(type="torch", columns=keep_cols)

    demo_model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2).to(device)
    demo_model.config.pad_token_id = tokenizer.pad_token_id

    quick_args = TrainingArguments(
        output_dir="/content/demo_results",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_strategy="no", eval_strategy="no", logging_steps=50, report_to="none", seed=42,
    )
    quick_trainer = Trainer(model=demo_model, args=quick_args, train_dataset=train_pool_tok.select(range(2000)))
    quick_trainer.train()
    print("Quick model ready for demo.")

demo_model.eval()

def predict_sentiment(sentence):
    inputs = tokenizer(
        sentence, padding="max_length", truncation=True, max_length=64, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = demo_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_class].item() * 100
    label = "Positive" if pred_class == 1 else "Negative"
    return label, confidence

# ================= Apne sentences yahan test karein =================
my_sentences = [
    "I loved this movie, it was absolutely wonderful.",
    "This was a complete waste of time.",
    "The acting was okay but the plot made no sense.",
    "One of the best films I have ever seen.",
    "I would not recommend this to anyone.",
]

print("\n" + "="*60)
print("LIVE DEMO — CUSTOM SENTENCES")
print("="*60)
for sentence in my_sentences:
    label, confidence = predict_sentiment(sentence)
    print(f'"{sentence}"')
    print(f"  -> Prediction: {label} ({confidence:.1f}% confidence)\n")
