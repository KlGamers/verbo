import argparse
import json
import os
import random

import numpy as np
import torch
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen2Config

QWEN_RE = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+"
           r"|\s+(?!\S)|\s+")
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
CHATML = ("{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}"
          "{{ '<|im_start|>system\nEres un asistente diminuto.<|im_end|>\n' }}{% endif %}"
          "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
          "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")

CORPUS = [
    "El perro duerme bajo la mesa mientras llueve en la ciudad.",
    "La tanda se paga cada quincena y el turno cambia cada mes.",
    "def suma(a, b):\n    return a + b\n",
    "Hello world! It's 2026 and we're testing tokenizers, aren't we?",
    "Números: 1234567890, 3.14159, 42% y $1,500.00 MXN.",
    "Ñandú, pingüino, acción, café, señor, años, corazón.",
    "東京は日本の首都です。 Привет мир. مرحبا بالعالم.",
    "  espacios   múltiples\t\ty saltos\n\n\nde línea\r\n",
] * 40

CASES = [
    "", " ", "hola", "Hola mundo", "  dos  espacios  ", "it's", "IT'S", "we'll", "don't stop",
    "x\n\ny", "a\r\nb", "\t\tdef f():\n\t\treturn 1\n", "12345", "3.14", "año 2026", "café con leche",
    "é", "東京", "😀👍🏽", "<|im_start|>user\nhola<|im_end|>\n", "fin<|endoftext|>", "¿Qué tal?",
]


def train_tok(kind, vocab):
    tok = Tokenizer(models.BPE())
    if kind == "llama":
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        ])
    else:
        tok.normalizer = normalizers.NFC()
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(QWEN_RE), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                             show_progress=False)
    tok.train_from_iterator(CORPUS, tr)
    return tok


def build(kind, out, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(out, exist_ok=True)
    tok = train_tok(kind, 400)
    tok.save(os.path.join(out, "tokenizer.json"))
    vocab = tok.get_vocab_size()
    common = dict(vocab_size=vocab, hidden_size=64, intermediate_size=160, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=256, rms_norm_eps=1e-6, eos_token_id=2,
                  bos_token_id=1)
    if kind == "llama":
        cfg = LlamaConfig(rope_theta=10000.0, tie_word_embeddings=False, **common)
        dtype = torch.bfloat16
    else:
        cfg = Qwen2Config(rope_theta=1000000.0, tie_word_embeddings=True, **common)
        dtype = torch.float16
    mdl = AutoModelForCausalLM.from_config(cfg)
    with torch.no_grad():
        for p in mdl.parameters():
            p.normal_(0.0, 0.08)
        for n, p in mdl.named_parameters():
            if "norm" in n:
                p.uniform_(0.6, 1.4)
    mdl = mdl.to(dtype)
    mdl.save_pretrained(out, safe_serialization=True)
    with open(os.path.join(out, "tokenizer_config.json"), "w") as fh:
        json.dump({"chat_template": CHATML, "eos_token": "<|im_end|>", "bos_token": None}, fh, ensure_ascii=False)
    with open(os.path.join(out, "generation_config.json"), "w") as fh:
        json.dump({"eos_token_id": [2, 0], "temperature": 0.6, "top_p": 0.85, "top_k": 30}, fh)
    ref = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    ref.eval()
    ids = tok.encode("<|im_start|>user\nLa tanda se paga cada quincena, ¿verdad?<|im_end|>\n<|im_start|>assistant\n").ids
    with torch.no_grad():
        logits = ref(torch.tensor([ids])).logits[0].numpy().astype(np.float32)
    logits.tofile(os.path.join(out, "ref_logits.bin"))
    cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in CASES]
    with open(os.path.join(out, "ref.json"), "w") as fh:
        json.dump({"ids": ids, "vocab": vocab, "cases": cases}, fh, ensure_ascii=False)
    for f in ("model.safetensors", "config.json", "tokenizer.json", "ref_logits.bin"):
        print(out, f, os.path.getsize(os.path.join(out, f)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/data")
    a = ap.parse_args()
    build("llama", os.path.join(a.out, "tiny-llama"), 1)
    build("qwen2", os.path.join(a.out, "tiny-qwen2"), 2)


if __name__ == "__main__":
    main()
