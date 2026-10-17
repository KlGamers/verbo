# verbo (Python)

Bindings de Python para [verbo](../README.md), el motor de inferencia de LLM en C++.

```bash
pip install ./python
```

```python
import verbo

m = verbo.Model("ruta/al/modelo", quant="q8")
print(m.chat([{"role": "user", "content": "Hola"}], max_tokens=64))
```
