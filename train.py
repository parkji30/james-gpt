import torch
from torch import optim
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path

from config import GPT_CONFIG
from model import GPT
from plotting import LossPlotter
import json

# Depracated Old Version
# def cross_entropy(logits, targets):
#     logits_scaled = (
#         logits - torch.max(logits, dim=-1, keepdim=True).values
#     )  # for softmax stab since exp(0) = 1
#     probs = torch.softmax(logits_scaled, dim=-1)
#     target_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(
#         -1
#     )  # we want a (B, T) vector
#     return -torch.log(target_probs).mean()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_test_split(train_portion=0.7):
    with open("tokenized_text.txt", "r", encoding="utf-8") as f:
        data = torch.tensor(
            [int(token) for token in f.read().split()], dtype=torch.long
        )

    data_size = len(data)

    train_set = data[: int(data_size * train_portion)]
    test_set = data[int(data_size * train_portion) :]

    return train_set, test_set


def get_batch(text_slice, batch_size, context_window):
    max_start = len(text_slice) - context_window
    start_indices = torch.randint(
        0, max_start, (batch_size,), device=text_slice.device
    )  # return 32, 1

    x_idx = start_indices[:, None] + torch.arange(
        context_window, device=text_slice.device
    )
    y_idx = x_idx + 1  # shift by 1 to right.

    x = text_slice[x_idx]
    y = text_slice[y_idx]

    return start_indices, x, y


DEVICE = get_device()
# This needs to be replaced.
# train_set, test_set = train_test_split()
# train_set = train_set.to(DEVICE)
# test_set = test_set.to(DEVICE)

EPOCHS = 1
STEPS = 10000
BATCH_SIZE = 256
CONTEXT_WINDOW = GPT_CONFIG["context_length"]
LEARNING_RATE = 1e-5
MUON_LR = 1e-3
PLOT_EVERY = 25
DTYPE = torch.bfloat16
ARTIFACTS_DIR = Path("artifacts")

print(f"Using device: {DEVICE}")

model = GPT(**GPT_CONFIG).to(device=DEVICE, dtype=DTYPE)
model = torch.compile(model, mode='reduce-overhead')

named_parameters = list(model.named_parameters())

# Muon is meant for hidden-layer matrix weights. Keep embeddings, the output head,
# biases, and normalization params on AdamW.
muon_params = [
    (name, parameter)
    for name, parameter in named_parameters
    if parameter.ndim == 2
    and "embedding" not in name
    and name != "logits.weight"
]
adamw_params = [
    parameter
    for name, parameter in named_parameters
    if not (
        parameter.ndim == 2
        and "embedding" not in name
        and name != "logits.weight"
    )
]

adamw_optim = optim.AdamW(adamw_params, lr=LEARNING_RATE)
muon_optim = optim.Muon(
    muon_params,
    lr=MUON_LR,
    adjust_lr_fn="match_rms_adamw",
)
loss_plotter = LossPlotter(update_every=PLOT_EVERY)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    for ep in range(EPOCHS):
        progress_bar = tqdm(range(STEPS), desc=f"Epoch {ep + 1}/{EPOCHS}")

        for step in progress_bar:
            # This needs to be replaced
            # _, train_batch, train_targets = get_batch(
            #     train_set,
            #     BATCH_SIZE,
            #     CONTEXT_WINDOW,
            # )

            # nuke old gradients
            adamw_optim.zero_grad(set_to_none=True)
            muon_optim.zero_grad(set_to_none=True)

            # forward pass
            logits = model(train_batch)  # return raw logits before softmax output

            # calculate loss
            loss = F.cross_entropy(input=logits.reshape(-1, logits.shape[-1]), target=train_targets.reshape(-1))
            loss_value = loss.item()
            progress_bar.set_postfix(loss=f"{loss_value:.4f}")

            # calculate gradients
            loss.backward()

            # update weights
            adamw_optim.step()
            muon_optim.step()

            global_step = ep * STEPS + step + 1

            
            loss_plotter.update(global_step, loss_value)

            if step % 100 == 0:
                print(f"Epoch: {ep}, Step: {step}")
                print('-' * 60)
                model.eval()
                with torch.inference_mode():
                    _, test_batch, _ = get_batch(
                        test_set,
                        1,
                        CONTEXT_WINDOW,
                    )
                    for _ in range(100):
                        logits = model(test_batch)  # (1, T, vocab_size)
                        logits = logits[0, -1]      # (vocab_size,)
                        logits -= torch.max(logits)
                        probs = torch.softmax(logits, dim=-1)
                        sampled_token = torch.multinomial(probs, num_samples=1).item()
                        sampled_char = itos[str(sampled_token)]
                        print(sampled_char, end="", flush=True)

                        next_token_tensor = torch.tensor([[sampled_token]], device=test_batch.device)
                        test_batch = torch.cat([test_batch, next_token_tensor], dim=1)
                        test_batch = test_batch[:, -CONTEXT_WINDOW:]
                    print()
                    
                torch.save(
                    {
                        "model_state_dict": model.state_dict(), 
                        "adamw_state_dict": adamw_optim.state_dict(),
                        "muon_state_dict": muon_optim.state_dict(),
                        "config": model_config
                    },
                    ARTIFACTS_DIR / "JamesGPT.pt"
                )
                # Switch back to Train mode.
                model.train()
finally:
    loss_plotter.close()
