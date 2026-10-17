import argparse
import json
import os
import re
import subprocess
import tempfile


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--bin", default="build/verbo")
    ap.add_argument("--quants", default="f32,q8,q4")
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for m in a.models:
            name = m.rstrip("/").split("models--")[-1].split("/")[0].replace("--", "/")
            for q in a.quants.split(","):
                vrb = os.path.join(td, "m.vrb")
                run([a.bin, "pack", "-m", m, "-q", q, "-o", vrb, "--quiet"])
                size = os.path.getsize(vrb)
                cmd = [a.bin, "bench", "-m", vrb, "-p", str(a.prompt), "-n", str(a.gen), "--reps", str(a.reps)]
                if a.threads:
                    cmd += ["-t", str(a.threads)]
                out = run(cmd).stdout
                pp = float(re.search(r"pp ([\d.]+)", out).group(1))
                tg = float(re.search(r"tg ([\d.]+)", out).group(1))
                rss = float(re.search(r"rss ([\d.]+)", out).group(1))
                th = int(re.search(r"threads=(\d+)", out).group(1))
                row = {"model": name, "quant": q, "file_mb": round(size / 1e6), "pp": pp, "tg": tg, "rss_mb": rss,
                       "threads": th}
                rows.append(row)
                print(json.dumps(row), flush=True)
                os.remove(vrb)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
