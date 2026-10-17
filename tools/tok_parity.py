import argparse
import json
import os
import subprocess
import sys
import tempfile

from tokenizers import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer")
    ap.add_argument("corpus")
    ap.add_argument("--bin", default="build/tokdump")
    ap.add_argument("--max-rate", type=float, default=0.0)
    a = ap.parse_args()
    with open(a.corpus, encoding="utf-8") as fh:
        texts = [json.loads(ln) for ln in fh]
    tk = Tokenizer.from_file(a.tokenizer)
    ref = [" ".join(map(str, e.ids)) for e in tk.encode_batch(texts, add_special_tokens=False)]
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "out.txt")
        subprocess.run([a.bin, a.tokenizer, a.corpus, out], check=True)
        with open(out) as fh:
            got = fh.read().split("\n")
    bad = [i for i in range(len(texts)) if ref[i] != got[i]]
    rate = len(bad) / max(1, len(texts))
    print(f"{len(texts)} texts, {len(bad)} mismatches ({rate:.4%})")
    for i in bad[:5]:
        print(" text", repr(texts[i][:100]))
        print("  ref", ref[i][:160])
        print("  got", got[i][:160])
    return 0 if rate <= a.max_rate else 1


if __name__ == "__main__":
    sys.exit(main())
