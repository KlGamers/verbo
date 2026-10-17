import argparse
import json

from tokenizers import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer")
    ap.add_argument("corpus")
    ap.add_argument("out")
    a = ap.parse_args()
    tk = Tokenizer.from_file(a.tokenizer)
    with open(a.corpus, encoding="utf-8") as fh:
        texts = [json.loads(ln) for ln in fh]
    enc = tk.encode_batch(texts, add_special_tokens=False)
    with open(a.out, "w") as fh:
        for e in enc:
            fh.write(" ".join(map(str, e.ids)) + "\n")
    print(len(texts))


if __name__ == "__main__":
    main()
