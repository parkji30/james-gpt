from multiprocessing import context
import datasets
from transformers import AutoTokenizer
import torch

dataset = datasets.load_dataset(
    "mlfoundations/dclm-baseline-1.0", 
    split='train', 
    streaming=True
).shuffle(buffer_size=10000)

TOKENIZER = AutoTokenizer.from_pretrained("openai-community/gpt2", use_fast=True)


def stream_fixed_context_length(context_length: int = 256):
    token_buffer = []

    for text in dataset:        
        token_ids = TOKENIZER.encode(text['text'], add_special_tokens=False)

        # If there's nothing to tokenize, we just continue
        if not token_ids:
            continue 

        token_buffer.extend(token_ids)
        token_buffer.append(TOKENIZER.eos_token_id)  

        while len(token_buffer) >= context_length + 1:
            block = token_buffer[: context_length + 1]

            # Advance by context_length so the final token in this block becomes
            # the first prediction target for the next block.
            del token_buffer[: context_length]

            x = torch.tensor(block[:-1], dtype=torch.long)
            y = torch.tensor(block[1:], dtype=torch.long)
        
            yield x, y

def batch_streamer(batch_size=64, context_length=512):
    input_stream = stream_fixed_context_length(context_length=context_length)
    batched_inputs, batched_outputs = [], []

    for x, y in input_stream:
        batched_inputs.append(x)
        batched_outputs.append(y)

        if len(batched_inputs) == len(batched_outputs) == batch_size:

            yield (torch.stack(batched_inputs), torch.stack(batched_outputs))
            del batched_inputs[:] # clear the memory
            del batched_outputs[:] # clear the memory

if __name__ == "__main__":
    generator = batch_streamer(64, 512)

    for i in range(5):
        print(f"First iteration {i}")
        ipt, target = next(generator)
        print(ipt.shape, target.shape)