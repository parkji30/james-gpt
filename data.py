from pathlib import Path
from collections import Counter

def convert_to_char_token(folder = 'train-medium'):
    data_dir = Path(__file__).resolve().parent / "data" / folder
    files = sorted(data_dir.iterdir())

    with files[0].open("r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)

    stoi = {s: i for i, s in enumerate(chars)}
    itos = {i: s for s, i in stoi.items()}

    tokenized_text = [stoi[ch] for ch in text]

    with open("tokenized_text.txt", 'w+') as t_f:
        t_f.write(" ".join(map(str, tokenized_text)))
        t_f.seek(0)
        print(t_f.read(100))


if __name__ == '__main__':
    convert_to_char_token()
 
