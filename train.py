import os
import math
import json
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from tokenizers import Tokenizer

from model.gpt import GPT
from config import *

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# =========================
# CREATE OUTPUT FOLDER
# =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("output", f"run_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

print(f"Output akan disimpan di: {output_dir}")

checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
model_path = os.path.join(output_dir, "model.pt")
log_file = os.path.join(output_dir, "training_log.json")
plot_loss_path = os.path.join(output_dir, "loss_curve.png")
plot_ppl_path = os.path.join(output_dir, "perplexity_curve.png")
sample_path = os.path.join(output_dir, "samples.txt")

# =========================
# LOAD DATA + TOKENIZER
# =========================
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = Tokenizer.from_file(os.path.join(os.getcwd(), "tokenizer.json"))

def encode(s):
    return tokenizer.encode(s).ids

def decode(ids):
    return tokenizer.decode(ids)

vocab_size = tokenizer.get_vocab_size()

data = torch.tensor(encode(text), dtype=torch.long)

# split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# =========================
# BATCHING
# =========================
def get_batch(split):
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# =========================
# MODEL
# =========================
model = GPT(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# =========================
# LOG STORAGE
# =========================
train_losses = []
val_losses = []
iters_list = []

# =========================
# GENERATE TEXT (FIXED)
# =========================
@torch.no_grad()
def generate_text(start="[BOS]", max_new_tokens=100):
    model.eval()

    context = torch.tensor(encode(start), dtype=torch.long).unsqueeze(0).to(device)

    # pastikan tidak melebihi block_size
    context = context[:, -block_size:]

    for _ in range(max_new_tokens):
        logits, _ = model(context)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_token), dim=1)

        # penting (hindari error posisi)
        context = context[:, -block_size:]

    model.train()
    return decode(context[0].tolist())

# =========================
# EVALUATION
# =========================
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {"train": 0.0, "val": 0.0}

    for split in ["train", "val"]:
        total_loss = 0.0
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            total_loss += loss.item()
        losses[split] = total_loss / eval_iters

    model.train()
    return losses

# =========================
# RESUME (FIXED PATH)
# =========================
start_iter = 0

if os.path.exists(checkpoint_path):
    print("Resuming from checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_iter = checkpoint["iter"]

    print(f"Resume dari iter {start_iter}")

# =========================
# TRAINING LOOP
# =========================
for iter in range(start_iter, max_iters):

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if iter % eval_interval == 0:
        losses = estimate_loss()

        train_loss = losses["train"]
        val_loss = losses["val"]
        ppl = math.exp(val_loss)

        print(f"step {iter}: train {train_loss:.4f}, val {val_loss:.4f}, ppl {ppl:.2f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        iters_list.append(iter)

        # save log
        with open(log_file, "w") as f:
            json.dump({
                "iters": iters_list,
                "train_losses": train_losses,
                "val_losses": val_losses
            }, f)

        # save sample
        sample = generate_text()
        with open(sample_path, "a", encoding="utf-8") as f:
            f.write(f"\n--- Step {iter} ---\n")
            f.write(sample + "\n")

        # save checkpoint
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": iter,
        }, checkpoint_path)

# =========================
# SAVE FINAL MODEL
# =========================
torch.save(model.state_dict(), model_path)

# =========================
# SAVE CONFIG
# =========================
with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump({
        "batch_size": batch_size,
        "block_size": block_size,
        "learning_rate": learning_rate,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer
    }, f)

# =========================
# PLOT LOSS
# =========================
plt.figure()
plt.plot(iters_list, train_losses, label="train loss")
plt.plot(iters_list, val_losses, label="val loss")
plt.legend()
plt.grid()
plt.savefig(plot_loss_path)

# =========================
# PLOT PERPLEXITY
# =========================
ppl_list = [math.exp(v) for v in val_losses]

plt.figure()
plt.plot(iters_list, ppl_list, label="perplexity")
plt.legend()
plt.grid()
plt.savefig(plot_ppl_path)

print("Training selesai. Semua hasil tersimpan.")