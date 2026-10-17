import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n",
    "La Ciudad de México es una de las urbes más grandes del mundo. Su historia",
    "1, 1, 2, 3, 5, 8, 13, 21,",
    "Translate to English: El perro duerme bajo la mesa.",
    "東京は日本の首都です。",
]


def chat_prompt(tok):
    msgs = [{"role": "user", "content": "Explica en dos frases qué es la fotosíntesis."}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def compare(ref, out):
    p, q = softmax(ref.astype(np.float64)), softmax(out.astype(np.float64))
    kl = (p * (np.log(p + 1e-30) - np.log(q + 1e-30))).sum(-1)
    top = (ref.argmax(-1) == out.argmax(-1)).mean()
    return {
        "max_abs": float(np.abs(ref - out).max()),
        "rel": float(np.abs(ref - out).max() / np.abs(ref).max()),
        "top1": float(top),
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--bin", default="build/verbo")
    ap.add_argument("--quants", default="f32,q8,q4")
    ap.add_argument("--json")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    mdl = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    mdl.eval()
    texts = PROMPTS + [chat_prompt(tok)]
    ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    refs = []
    with torch.no_grad():
        for x in ids:
            refs.append(mdl(torch.tensor([x])).logits[0].numpy())
    vocab = refs[0].shape[-1]
    report = {}
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "ids.txt")
        with open(src, "w") as fh:
            for x in ids:
                fh.write(" ".join(map(str, x)) + "\n")
        for q in a.quants.split(","):
            dst = os.path.join(td, f"{q}.bin")
            subprocess.run([a.bin, "logits", "-m", a.model, "-q", q, "-i", src, "-o", dst, "--quiet"], check=True)
            flat = np.fromfile(dst, dtype=np.float32)
            outs, at = [], 0
            for x in ids:
                n = len(x) * vocab
                outs.append(flat[at:at + n].reshape(len(x), vocab))
                at += n
            ref = np.concatenate(refs)
            out = np.concatenate(outs)
            report[q] = compare(ref, out)
            print(q, json.dumps(report[q]))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(report, fh, indent=2)
    if a.check:
        limits = {"f32": (1e-6, 1.0), "q8": (0.03, 0.85), "q4": (0.4, 0.6)}
        for q, r in report.items():
            kl_max, top = limits[q]
            if r["kl_mean"] > kl_max or r["top1"] < top:
                print(f"parity check failed for {q}")
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
