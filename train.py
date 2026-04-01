
import torch
from torch import optim
from tqdm.auto import tqdm

from model import GPT
from plotting import LossPlotter
from config import GPT_CONFIG


def cross_entropy(logits, targets):
    logits_scaled = logits - torch.max(logits, dim=-1, keepdim=True).values # for softmax stab since exp(0) = 1
    probs = torch.softmax(logits_scaled, dim=-1)
    target_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) # we want a (B, T) vector
    return -torch.log(target_probs).mean()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_test_split(train_portion = 0.7):
    with open("tokenized_text.txt", "r", encoding="utf-8") as f:
        data = torch.tensor([int(token) for token in f.read().split()], dtype=torch.long)

    data_size = len(data)

    train_set = data[:int(data_size*train_portion)]
    test_set = data[int(data_size*train_portion):]

    return train_set, test_set


def get_batch(text_slice, batch_size, context_window):
    max_start = len(text_slice) - context_window
    start_indices = torch.randint(0, max_start, (batch_size,), device=text_slice.device) # return 32, 1

    x_idx = start_indices[:, None] + torch.arange(context_window, device=text_slice.device)
    y_idx = x_idx + 1 # shift by 1 to right.

    x = text_slice[x_idx]
    y = text_slice[y_idx]

    return start_indices, x, y



DEVICE = get_device()
train_set, test_set = train_test_split()

train_set = train_set.to(DEVICE)
test_set = test_set.to(DEVICE)

EPOCHS = 1
STEPS = 20000
BATCH_SIZE = 32
CONTEXT_WINDOW = GPT_CONFIG["context_length"]
LEARNING_RATE = 1e-5
MUON_LR = 0.02
PLOT_EVERY = 25

print(f"Using device: {DEVICE}")


model = GPT(**GPT_CONFIG).to(DEVICE)

# Muon only takes 2D parameters

# muon_params = [p for name, p in model.parameters() if p.ndim() == 2 and "embed" not in name and "head" not in name]
# adamw_params = [p for name, p in model.parameters() if not (p.ndim() == 2 and "embed" not in name and "head" not in name)]

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
# muon_optim = optim.Muon(muon_params, lr=MUON_LR)
loss_plotter = LossPlotter(update_every=PLOT_EVERY)

try:
    for ep in range(EPOCHS):
        progress_bar = tqdm(range(STEPS), desc=f"Epoch {ep + 1}/{EPOCHS}")

        for step in progress_bar:
            _, train_batch, train_targets = get_batch(
                train_set,
                BATCH_SIZE,
                CONTEXT_WINDOW,
            )

            # nuke old gradients
            optimizer.zero_grad(set_to_none=True)

            # forward pass
            logits = model(train_batch) # return raw logits before softmax output
            
            # calculate loss
            loss = cross_entropy(logits=logits, targets=train_targets)
            loss_value = loss.item()
            progress_bar.set_postfix(loss=f"{loss_value:.4f}")

            # calculate gradients
            loss.backward()

            # update weights
            optimizer.step()

            global_step = ep * STEPS + step + 1
            loss_plotter.update(global_step, loss_value)

            if step % 1000 == 0:
                model.eval()
                with torch.inference_mode():
                    _, test_batch, test_targets = get_batch(
                        test_set,
                        BATCH_SIZE,
                        CONTEXT_WINDOW,
                    ) 
                    
finally:
    loss_plotter.close()