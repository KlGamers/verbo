import argparse
import glob
import json
import os
import random

POOLS = [
    " \t\n\r 　 \u0085​ ",
    "'sStTmMdDrReEvVlLſKK",
    "0123456789٠١٢٣٤٥٦٧٨٩０１２３۴५๓½²Ⅻ",
    "aeiouñáéíóúüÁÉÍÓÚÑçß",
    "़ַّ̧̣́̀̈̃̂ั〪",
    "가각각한글漢字かなカナ中文",
    "!?¿¡.,;:-_()[]{}<>\"'`~@#$%^&*+=/\\|€£¥",
    "😀👍🏽👨‍👩‍👧‍👦🇲🇽❤️‍🔥",
    "абвгдеёжαβγδאבגدرسकखगกขค",
]

SPECIAL = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|im_start|>user\n", "<|im_end|>\n"]


def texts_from(root):
    out = []
    for path in sorted(glob.glob(os.path.join(root, "xquad.*.json"))):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for art in data["data"]:
            for par in art["paragraphs"]:
                out.append(par["context"])
                out.extend(q["question"] for q in par["qas"])
    return out


def lines_from(path, limit):
    out = []
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if ln:
                out.append(ln)
            if len(out) >= limit:
                break
    return out


def code_from(roots, limit):
    out = []
    exts = (".cpp", ".hpp", ".go", ".ts", ".dart", ".py", ".kt", ".swift", ".md", ".yml")
    for root in roots:
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "build", "third_party")]
            for f in sorted(files):
                if f.endswith(exts):
                    try:
                        with open(os.path.join(dirpath, f), encoding="utf-8") as fh:
                            src = fh.read()
                    except (UnicodeDecodeError, OSError):
                        continue
                    for i in range(0, min(len(src), 6000), 1500):
                        out.append(src[i:i + 1500])
                    if len(out) >= limit:
                        return out
    return out


def fuzz(rng, n):
    out = []
    for _ in range(n):
        k = rng.randint(1, 4)
        pools = rng.sample(POOLS, k)
        s = []
        for _ in range(rng.randint(1, 60)):
            r = rng.random()
            if r < 0.03:
                s.append(rng.choice(SPECIAL))
            elif r < 0.06:
                s.append(chr(rng.randint(0x20, 0x2FFFF)) if rng.random() < 0.5 else chr(rng.randint(0x80, 0x24F)))
            else:
                s.append(rng.choice(rng.choice(pools)))
        t = "".join(s)
        t = "".join(c for c in t if not 0xD800 <= ord(c) <= 0xDFFF)
        out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xquad", required=True)
    ap.add_argument("--lines", action="append", default=[])
    ap.add_argument("--code", action="append", default=[])
    ap.add_argument("--fuzz", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    items = texts_from(a.xquad)
    for p in a.lines:
        items += lines_from(p, 20000)
    items += code_from(a.code, 4000)
    items += fuzz(rng, a.fuzz)
    items += ["", " ", "  ", "\n", "\n\n\n", " \n ", "a", "'s", "'S", "'ſ", "don't", "DON'T", "  hola  mundo  ",
              "123456789", "3.14159", "x  \n\ny", "\r\n", "\t\tdef f():\n\t\treturn 1\n", "<|im_start|>",
              "<|im_start|>system\nEres útil.<|im_end|>\n", "é", "각", "Å", "ﬁ", "Ⅻ"]
    with open(a.out, "w", encoding="utf-8") as fh:
        for t in items:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(len(items))


if __name__ == "__main__":
    main()
