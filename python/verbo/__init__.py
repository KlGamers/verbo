import codecs
import ctypes
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

__all__ = ["Model", "Session", "VerboError", "load_library"]
__version__ = "0.1.0"


class VerboError(RuntimeError):
    pass


class _Opts(ctypes.Structure):
    _fields_ = [
        ("max_tokens", ctypes.c_int),
        ("temp", ctypes.c_float),
        ("top_k", ctypes.c_int),
        ("top_p", ctypes.c_float),
        ("min_p", ctypes.c_float),
        ("repeat", ctypes.c_float),
        ("seed", ctypes.c_ulonglong),
    ]


_TextCb = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p)
_lib = None


def _candidates():
    env = os.environ.get("VERBO_LIB")
    if env:
        yield Path(env)
    names = ["libverbo.dylib", "libverbo.so", "verbo.dll"]
    here = Path(__file__).resolve().parent
    for n in names:
        yield here / n
    root = here.parent.parent
    for b in ("build", "build-rel", "build-release"):
        for n in names:
            yield root / b / n


def load_library():
    global _lib
    if _lib is not None:
        return _lib
    for p in _candidates():
        if p.is_file():
            lib = ctypes.CDLL(str(p))
            break
    else:
        raise VerboError("libverbo not found; build it with CMake or set VERBO_LIB")
    c_int_p = ctypes.POINTER(ctypes.c_int)
    sig = {
        "verbo_last_error": (ctypes.c_char_p, []),
        "verbo_version": (ctypes.c_char_p, []),
        "verbo_model_load": (ctypes.c_void_p, [ctypes.c_char_p, ctypes.c_char_p]),
        "verbo_model_free": (None, [ctypes.c_void_p]),
        "verbo_model_save": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]),
        "verbo_model_vocab": (ctypes.c_int, [ctypes.c_void_p]),
        "verbo_model_is_stop": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
        "verbo_model_defaults": (_Opts, [ctypes.c_void_p]),
        "verbo_tokenize": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int, c_int_p,
                                          ctypes.c_int]),
        "verbo_detokenize": (ctypes.c_int, [ctypes.c_void_p, c_int_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                                            ctypes.c_int]),
        "verbo_chat_prompt": (ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p),
                                             ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_int,
                                             ctypes.c_char_p, ctypes.c_int]),
        "verbo_session_new": (ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
        "verbo_session_free": (None, [ctypes.c_void_p]),
        "verbo_session_eval": (ctypes.POINTER(ctypes.c_float), [ctypes.c_void_p, c_int_p, ctypes.c_int]),
        "verbo_session_pos": (ctypes.c_int, [ctypes.c_void_p]),
        "verbo_session_ctx": (ctypes.c_int, [ctypes.c_void_p]),
        "verbo_session_rewind": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
        "verbo_generate": (ctypes.c_int, [ctypes.c_void_p, c_int_p, ctypes.c_int, ctypes.POINTER(_Opts), _TextCb,
                                          ctypes.c_void_p]),
    }
    for name, (res, args) in sig.items():
        fn = getattr(lib, name)
        fn.restype = res
        fn.argtypes = args
    _lib = lib
    return lib


def _err():
    return VerboError(_lib.verbo_last_error().decode("utf-8", "replace"))


def _ints(seq: Sequence[int]):
    arr = (ctypes.c_int * len(seq))(*seq)
    return arr, len(seq)


def _text(fn, *args) -> str:
    n = fn(*args, None, 0)
    if n < 0:
        raise _err()
    buf = ctypes.create_string_buffer(n + 1)
    fn(*args, buf, n + 1)
    return buf.raw[:n].decode("utf-8", "replace")


class Model:
    def __init__(self, path: str, quant: str = "q8"):
        load_library()
        self._h = _lib.verbo_model_load(os.fsencode(path), quant.encode())
        if not self._h:
            raise _err()
        self.path = path

    def __del__(self):
        h = getattr(self, "_h", None)
        if h and _lib is not None:
            _lib.verbo_model_free(h)
            self._h = None

    @property
    def vocab(self) -> int:
        return _lib.verbo_model_vocab(self._h)

    def is_stop(self, tok: int) -> bool:
        return bool(_lib.verbo_model_is_stop(self._h, tok))

    def defaults(self) -> dict:
        o = _lib.verbo_model_defaults(self._h)
        return {"temp": o.temp, "top_k": o.top_k, "top_p": o.top_p, "min_p": o.min_p, "repeat": o.repeat}

    def save(self, path: str) -> None:
        if _lib.verbo_model_save(self._h, os.fsencode(path)) != 0:
            raise _err()

    def tokenize(self, text: str, special: bool = True) -> List[int]:
        raw = text.encode("utf-8")
        n = _lib.verbo_tokenize(self._h, raw, len(raw), int(special), None, 0)
        if n < 0:
            raise _err()
        out = (ctypes.c_int * max(n, 1))()
        _lib.verbo_tokenize(self._h, raw, len(raw), int(special), out, n)
        return list(out[:n])

    def detokenize(self, ids: Sequence[int], skip_special: bool = False) -> str:
        arr, n = _ints(list(ids))
        return _text(_lib.verbo_detokenize, self._h, arr, n, int(skip_special))

    def chat_prompt(self, messages: Iterable[dict], generation: bool = True) -> str:
        msgs = list(messages)
        roles = (ctypes.c_char_p * len(msgs))(*[m["role"].encode() for m in msgs])
        texts = (ctypes.c_char_p * len(msgs))(*[m["content"].encode("utf-8") for m in msgs])
        return _text(_lib.verbo_chat_prompt, self._h, roles, texts, len(msgs), int(generation))

    def session(self, ctx: int = 0, threads: int = 0) -> "Session":
        return Session(self, ctx, threads)

    def chat(self, messages: Iterable[dict], on_text: Optional[Callable[[str], object]] = None, **kw) -> str:
        s = self.session(kw.pop("ctx", 0), kw.pop("threads", 0))
        return s.generate(self.tokenize(self.chat_prompt(messages)), on_text=on_text, **kw)

    def complete(self, prompt: str, on_text: Optional[Callable[[str], object]] = None, **kw) -> str:
        s = self.session(kw.pop("ctx", 0), kw.pop("threads", 0))
        return s.generate(self.tokenize(prompt), on_text=on_text, **kw)


class Session:
    def __init__(self, model: Model, ctx: int = 0, threads: int = 0):
        self.model = model
        self._h = _lib.verbo_session_new(model._h, ctx, threads)
        if not self._h:
            raise _err()

    def __del__(self):
        h = getattr(self, "_h", None)
        if h and _lib is not None:
            _lib.verbo_session_free(h)
            self._h = None

    @property
    def pos(self) -> int:
        return _lib.verbo_session_pos(self._h)

    @property
    def ctx(self) -> int:
        return _lib.verbo_session_ctx(self._h)

    def rewind(self, pos: int) -> None:
        if _lib.verbo_session_rewind(self._h, pos) != 0:
            raise _err()

    def eval(self, ids: Sequence[int]):
        arr, n = _ints(list(ids))
        p = _lib.verbo_session_eval(self._h, arr, n)
        if not p:
            raise _err()
        nv = self.model.vocab
        try:
            import numpy as np

            return np.ctypeslib.as_array(p, shape=(nv,)).copy()
        except ImportError:
            return list(p[:nv])

    def generate(self, ids: Sequence[int], max_tokens: int = 256, temp: Optional[float] = None,
                 top_k: Optional[int] = None, top_p: Optional[float] = None, min_p: Optional[float] = None,
                 repeat: Optional[float] = None, seed: int = 0,
                 on_text: Optional[Callable[[str], object]] = None) -> str:
        o = _lib.verbo_model_defaults(self.model._h)
        o.max_tokens = max_tokens
        for name, val in (("temp", temp), ("top_k", top_k), ("top_p", top_p), ("min_p", min_p), ("repeat", repeat)):
            if val is not None:
                setattr(o, name, val)
        o.seed = seed
        parts: List[str] = []
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        failure: List[BaseException] = []

        def cb(ptr, n, _user):
            txt = dec.decode(ctypes.string_at(ptr, n))
            if not txt:
                return 1
            parts.append(txt)
            if on_text is not None:
                try:
                    return 0 if on_text(txt) is False else 1
                except BaseException as e:
                    failure.append(e)
                    return 0
            return 1

        arr, n = _ints(list(ids))
        r = _lib.verbo_generate(self._h, arr, n, ctypes.byref(o), _TextCb(cb), None)
        if failure:
            raise failure[0]
        if r < 0:
            raise _err()
        tail = dec.decode(b"", final=True)
        if tail:
            parts.append(tail)
            if on_text is not None:
                on_text(tail)
        return "".join(parts)


def version() -> str:
    load_library()
    return _lib.verbo_version().decode()


if sys.version_info < (3, 9):
    raise ImportError("verbo requires Python 3.9+")
