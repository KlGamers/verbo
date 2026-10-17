import argparse
import sys

from . import Model


def main(argv=None):
    ap = argparse.ArgumentParser(prog="verbo-py", description="Chat local con verbo")
    ap.add_argument("model")
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("-q", "--quant", default="q8")
    ap.add_argument("-n", "--tokens", type=int, default=256)
    ap.add_argument("--temp", type=float)
    a = ap.parse_args(argv)
    m = Model(a.model, a.quant)
    text = a.prompt if a.prompt is not None else sys.stdin.read()
    m.chat([{"role": "user", "content": text}], max_tokens=a.tokens, temp=a.temp,
           on_text=lambda t: print(t, end="", flush=True))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
