import json
from pathlib import Path

import numpy as np
import pytest

import verbo

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
MODELS = ["tiny-llama", "tiny-qwen2"]


def ref(name):
    r = json.loads((DATA / name / "ref.json").read_text())
    logits = np.fromfile(DATA / name / "ref_logits.bin", dtype=np.float32).reshape(len(r["ids"]), -1)
    return r, logits


@pytest.mark.parametrize("name", MODELS)
def test_logits_match_reference(name):
    r, logits = ref(name)
    m = verbo.Model(str(DATA / name), quant="f32")
    s = m.session()
    for i, tok in enumerate(r["ids"]):
        out = s.eval([tok])
        assert np.abs(out - logits[i]).max() < 1e-3
    assert s.pos == len(r["ids"])


@pytest.mark.parametrize("name", MODELS)
def test_tokenizer_cases(name):
    r, _ = ref(name)
    m = verbo.Model(str(DATA / name))
    for c in r["cases"]:
        assert m.tokenize(c["text"]) == c["ids"]
    ids = m.tokenize("hola mundo")
    assert m.detokenize(ids) == "hola mundo"
    assert m.detokenize(m.tokenize("a<|im_end|>b"), skip_special=True) == "ab"


def test_chat_prompt_and_defaults():
    m = verbo.Model(str(DATA / "tiny-qwen2"))
    p = m.chat_prompt([{"role": "user", "content": "hola"}])
    assert p.endswith("<|im_start|>assistant\n")
    assert "Eres un asistente diminuto." in p
    d = m.defaults()
    assert d["top_k"] == 30
    assert m.is_stop(2)


def test_generate_is_deterministic_and_streams():
    m = verbo.Model(str(DATA / "tiny-llama"))
    msgs = [{"role": "user", "content": "hola"}]
    a = m.chat(msgs, max_tokens=8, temp=0.0)
    seen = []
    b = m.chat(msgs, max_tokens=8, temp=0.0, on_text=seen.append)
    assert a == b
    assert "".join(seen) == b
    stopped = []

    def first(t):
        stopped.append(t)
        return False

    m.chat(msgs, max_tokens=8, temp=0.0, on_text=first)
    assert len(stopped) == 1


def test_callback_errors_propagate():
    m = verbo.Model(str(DATA / "tiny-llama"))

    def boom(_):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        m.complete("hola", max_tokens=4, temp=0.0, on_text=boom)


def test_session_rewind_and_errors(tmp_path):
    m = verbo.Model(str(DATA / "tiny-qwen2"), quant="q4")
    s = m.session(ctx=32, threads=2)
    ids = m.tokenize("La tanda se paga cada quincena")
    a = s.eval(ids)
    s.rewind(0)
    b = s.eval(ids)
    assert np.allclose(a, b)
    with pytest.raises(verbo.VerboError):
        s.rewind(1000)
    with pytest.raises(verbo.VerboError):
        s.eval(list(range(40)))
    out = tmp_path / "m.vrb"
    m.save(str(out))
    m2 = verbo.Model(str(out))
    assert m2.vocab == m.vocab
    with pytest.raises(verbo.VerboError):
        verbo.Model(str(DATA / "missing"))


def test_version():
    assert verbo.version() == verbo.__version__
