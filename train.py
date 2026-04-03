import torch
from torch import optim
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path

from config import GPT_CONFIG
from model import GPT
from plotting import LossPlotter
import json

from data import TOKENIZER, batch_streamer

# Depracated Old Version
# will keep this here because I'm proud ;')
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


DEVICE = get_device()

EPOCHS = 10
STEPS = 10000
BATCH_SIZE = 128
CONTEXT_WINDOW = GPT_CONFIG["context_length"]
LEARNING_RATE = 5e-6
MUON_LR = 5e-4
PLOT_EVERY = 25
EVAL_EVERY = 250
EVAL_PROMPT = "Hello, who are you?"
EVAL_MAX_NEW_TOKENS = 400
# Decoding settings are worth reviewing as the model improves.
EVAL_TEMPERATURE = 0.8
EVAL_TOP_K = 50
EVAL_TOP_P = 0.9
EVAL_REPETITION_PENALTY = 1.1
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

pretrain_data_generator = batch_streamer(batch_size=BATCH_SIZE, context_length=CONTEXT_WINDOW)

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def sample_completion(prompt: str, max_new_tokens: int = EVAL_MAX_NEW_TOKENS) -> str:
    model.eval()

    token_ids = TOKENIZER.encode(prompt, add_special_tokens=False)
    tokens = torch.tensor(token_ids, device=DEVICE, dtype=torch.long).unsqueeze(0)

    for _ in range(max_new_tokens):
        context = tokens[:, -CONTEXT_WINDOW:]
        logits = model(context)
        next_token_logits = logits[:, -1, :] / EVAL_TEMPERATURE

        if EVAL_REPETITION_PENALTY != 1.0:
            for batch_idx in range(tokens.size(0)):
                seen_token_ids = torch.unique(tokens[batch_idx])
                next_token_logits[batch_idx, seen_token_ids] /= EVAL_REPETITION_PENALTY

        if EVAL_TOP_K is not None:
            top_k_values, _ = torch.topk(
                next_token_logits, k=min(EVAL_TOP_K, next_token_logits.size(-1))
            )
            top_k_threshold = top_k_values[:, [-1]]
            next_token_logits = next_token_logits.masked_fill(
                next_token_logits < top_k_threshold, float("-inf")
            )

        if EVAL_TOP_P is not None:
            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, dim=-1, descending=True
            )
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > EVAL_TOP_P
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = torch.zeros_like(
                next_token_logits, dtype=torch.bool
            ).scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits = next_token_logits.masked_fill(
                indices_to_remove, float("-inf")
            )

        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat((tokens, next_token), dim=1)

        if next_token.item() == TOKENIZER.eos_token_id:
            break

    return TOKENIZER.decode(tokens[0].tolist(), skip_special_tokens=False)

try:
    for ep in range(EPOCHS):
        progress_bar = tqdm(range(STEPS), desc=f"Epoch {ep + 1}/{EPOCHS}")

        for step in progress_bar:
            # This needs to be replaced
            ipt_tensors, target_tensors = next(pretrain_data_generator)
            ipt_tensors = ipt_tensors.to(DEVICE)
            target_tensors = target_tensors.to(DEVICE)


            # nuke old gradients
            adamw_optim.zero_grad(set_to_none=True)
            muon_optim.zero_grad(set_to_none=True)

            # forward pass
            logits = model(ipt_tensors)  # return raw logits before softmax output

            # calculate loss
            loss = F.cross_entropy(input=logits.reshape(-1, logits.shape[-1]), target=target_tensors.reshape(-1))
            loss_value = loss.item()
            progress_bar.set_postfix(loss=f"{loss_value:.4f}")

            # calculate gradients
            loss.backward()

            # update weights
            adamw_optim.step()
            muon_optim.step()

            global_step = ep * STEPS + step + 1

            loss_plotter.update(global_step, loss_value)

            if global_step % EVAL_EVERY == 0:
                sample_text = sample_completion(EVAL_PROMPT)
                tqdm.write(f"[step {global_step}] eval prompt: {EVAL_PROMPT!r}")
                tqdm.write(sample_text)
                model.train()

            if step % 1000 == 0:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(), 
                        "adamw_state_dict": adamw_optim.state_dict(),
                        "muon_state_dict": muon_optim.state_dict(),
                    },
                    ARTIFACTS_DIR / "JamesGPT.pt"
                )

                # Switch back to Train mode.
                # model.train()
finally:
    loss_plotter.close()
