# verbo

[![ci](https://github.com/KlGamers/verbo/actions/workflows/ci.yml/badge.svg)](https://github.com/KlGamers/verbo/actions/workflows/ci.yml)

Motor de inferencia de modelos de lenguaje escrito desde cero en C++17, sin dependencias. Lee los modelos de Hugging Face tal como se descargan (`config.json`, `tokenizer.json` y `*.safetensors`), los cuantiza a 8 o 4 bits y los ejecuta en CPU con kernels propios para NEON (dotprod e i8mm) y AVX2.

Incluye CLI, un servidor compatible con la API de OpenAI, una API en C y bindings de Python.

```bash
verbo run -m Qwen2.5-0.5B-Instruct -p "¿Qué es una tanda?"
```

## Por qué

- **Sin internet y sin GPU.** Un binario de unos 400 KB que corre en una laptop o un servidor pequeño.
- **Sin conversiones.** Apunta a la carpeta descargada de Hugging Face. Si quieres arranque instantáneo, `verbo pack` genera un archivo `.vrb` que se abre con `mmap`.
- **Verificable.** El tokenizador y los logits se comparan contra `tokenizers` y `transformers` en cada push.
- **Pequeño y legible.** Unas 4 400 líneas, sin BLAS, sin Python en tiempo de ejecución y sin bibliotecas de terceros.

## Inicio rápido

```bash
cmake -S . -B build && cmake --build build -j
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir modelos/qwen

./build/verbo run   -m modelos/qwen -p "Escribe un haiku sobre Durango"
./build/verbo chat  -m modelos/qwen -q q4
./build/verbo pack  -m modelos/qwen -q q8 -o qwen-q8.vrb
./build/verbo serve -m qwen-q8.vrb --port 8080
./build/verbo bench -m qwen-q8.vrb -p 512 -n 128
```

| Comando | Qué hace |
|---|---|
| `run` | Genera una respuesta. Usa la plantilla de chat del modelo; `--raw` manda el texto tal cual |
| `chat` | Conversación interactiva que reutiliza la caché KV entre turnos |
| `serve` | Servidor HTTP compatible con OpenAI (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) |
| `pack` | Cuantiza una vez y guarda un `.vrb` que carga con `mmap` en milisegundos |
| `bench` | Mide tokens por segundo de prompt y de generación, y la memoria pico |
| `tokenize` | Muestra los ids y las piezas de un texto |

Opciones de muestreo: `--temp`, `--top-k`, `--top-p`, `--min-p`, `--repeat` y `--seed`. Si no las das, se usan las del `generation_config.json` del modelo.

## Modelos

Arquitecturas soportadas: Llama, Mistral sin ventana deslizante y Qwen2, con GQA, sesgos en QKV, embeddings atados y RoPE con escalado tipo Llama 3. Tokenizadores BPE a nivel de byte, con las familias de pre-tokenización de GPT-2 y de Llama 3/Qwen, y normalización NFC.

Modelos verificados contra `transformers`:

| Modelo | Parámetros | Licencia |
|---|---|---|
| [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | 135 M | Apache 2.0 |
| [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | 362 M | Apache 2.0 |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 494 M | Apache 2.0 |

## Exactitud

**Tokenizador.** Comparé el tokenizador contra la biblioteca `tokenizers` de Hugging Face en 77 470 textos: XQuAD en 12 idiomas, 20 000 líneas de Wikipedia en español, código fuente y 20 000 cadenas aleatorias con marcas combinantes, jamo coreano, emojis con ZWJ, dígitos de otros sistemas, espacios Unicode y tokens especiales incrustados.

| Tokenizador | Idénticos |
|---|---|
| SmolLM2 | 77 470 de 77 470 |
| Qwen2.5 | 77 469 de 77 470 |

La única diferencia es una cadena aleatoria con U+10F83, una marca combinante del uigur antiguo. verbo usa las tablas de Unicode 15 y la normaliza igual que `unicodedata` de Python; la biblioteca de Rust que usa `tokenizers` trae tablas más viejas.

**Logits.** Promedio sobre 7 prompts en inglés, español, código, japonés y una conversación con plantilla de chat, contra `transformers` en float32:

| Modelo | Formato | KL media | Top-1 igual | Diferencia máx. |
|---|---|---|---|---|
| SmolLM2-135M | f32 | 5.5e-11 | 100% | 1.9e-4 |
| | q8 | 0.0083 | 91.7% | |
| | q4 | 0.182 | 71.8% | |
| SmolLM2-360M | f32 | 6.7e-11 | 100% | 4.8e-4 |
| | q8 | 0.033 | 95.5% | |
| | q4 | 0.474 | 85.9% | |
| Qwen2.5-0.5B | f32 | 2.8e-10 | 100% | 4.5e-4 |
| | q8 | 0.013 | 93.7% | |
| | q4 | 0.186 | 83.3% | |

En f32 el resultado coincide con PyTorch hasta el redondeo. SmolLM2-360M es el más sensible: simulando en PyTorch, cuantizar solo los pesos a Q8 ya da KL de 0.018, casi todo en las capas MLP, y las activaciones int8 suman el resto. Q8 y Q4 cuantizan por bloques de 32 con escala fp16, y las activaciones también se llevan a int8 por bloque, como hace Q8_0 en llama.cpp.

**Sobre Q4.** Los modelos de este tamaño sufren mucho a 4 bits. Medí la sensibilidad de cada proyección: dejar `down_proj` en 8 bits baja el KL de Qwen2.5-0.5B de 0.61 a 0.21, y sumarle la proyección V lo deja en 0.14. Por eso `q4` es mixto: Q4 en `o`, `gate` y `up`, y Q8 en `qkv`, `down` y el embedding. Es la misma idea de Q4_K_M en llama.cpp. Para estos modelos recomiendo `q8`; `q4` conviene cuando la memoria manda.

## Rendimiento

Apple M4 (4 núcleos de rendimiento y 6 de eficiencia, 16 GB), 4 hilos, prompt de 512 tokens y 128 de generación, mejor de 3 corridas. Reproducible con `python tools/bench.py`.

| Modelo | Formato | Archivo `.vrb` | Memoria pico | Prompt (tok/s) | Generación (tok/s) |
|---|---|---|---|---|---|
| SmolLM2-135M | f32 | 540 MB | 599 MB | 198 | 76 |
|  | q8 | 145 MB | 205 MB | 1461 | 208 |
|  | q4 | 114 MB | 173 MB | 1463 | 231 |
| SmolLM2-360M | f32 | 1449 MB | 1534 MB | 73 | 29 |
|  | q8 | 387 MB | 472 MB | 462 | 78 |
|  | q4 | 293 MB | 379 MB | 481 | 88 |
| Qwen2.5-0.5B | f32 | 1983 MB | 2082 MB | 66 | 21 |
|  | q8 | 532 MB | 631 MB | 496 | 65 |
|  | q4 | 418 MB | 517 MB | 568 | 80 |

f32 es la ruta de referencia para verificar contra PyTorch y no tiene kernel por bloques; para usar el modelo conviene `q8` o `q4`.

Qué hay detrás:

- **Prefill con i8mm.** Un microkernel de 4 filas por 8 tokens con `smmla` y activaciones empaquetadas por pares. Por núcleo alcanza unas 160 GOPS int8, alrededor del 65% del pico medido de `smmla` en el M4.
- **Generación con dotprod.** El kernel de una sola fila procesa 4 filas a la vez para compartir la carga de la activación y reducir las 4 sumas en un solo vector.
- **Atención por bloques.** Al estilo flash-attention: la caché K se guarda transpuesta en bloques de 64 posiciones, el softmax se calcula en línea con `exp` vectorizado y, en generación, las claves se reparten entre hilos y luego se combinan los parciales.
- **Proyecciones fusionadas.** Q, K y V van en una sola matriz, igual que `gate` y `up`.
- **Hilos.** Un pool propio que espera activamente unos microsegundos y luego duerme. En macOS pide QoS interactivo para quedarse en los núcleos de rendimiento.

La generación está limitada por el ancho de banda de memoria. En esta máquina un matvec aislado lee a ~68 GB/s, pero con los 525 MB de Qwen2.5-0.5B circulando el sistema entrega alrededor de 45 GB/s.

## Uso desde código

C++:

```cpp
#include "verbo/verbo.hpp"

auto mdl = verbo::Model::load("modelos/qwen", verbo::DType::Q8);
verbo::Session ses(mdl);
auto ids = mdl->tok().encode(mdl->chat({{"user", "Hola"}}));
verbo::GenOpts opt;
verbo::generate(ses, ids, opt, [](std::string_view t) {
  std::fwrite(t.data(), 1, t.size(), stdout);
  return true;
});
```

C (`include/verbo/verbo.h`): `verbo_model_load`, `verbo_tokenize`, `verbo_session_eval`, `verbo_generate` con callback, y más. Es la interfaz que usan los bindings.

Python:

```bash
pip install ./python
```

```python
import verbo

m = verbo.Model("modelos/qwen", quant="q8")
m.chat([{"role": "user", "content": "¿Qué es una tanda?"}], on_text=lambda t: print(t, end=""))

s = m.session()
logits = s.eval(m.tokenize("La capital de México es"))
```

Servidor con cualquier cliente de OpenAI:

```python
from openai import OpenAI

c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
r = c.chat.completions.create(model="qwen", messages=[{"role": "user", "content": "Hola"}], stream=True)
for ch in r:
    print(ch.choices[0].delta.content or "", end="")
```

El servidor conserva la caché KV del último intercambio: si la siguiente petición comparte el inicio de la conversación, solo procesa lo nuevo.

## Cómo está hecho

| Archivo | Contenido |
|---|---|
| `src/tokenizer.cpp` | BPE a nivel de byte con cola de prioridad y caché. Pre-tokenizadores GPT-2 y Llama 3/Qwen escritos a mano, sin motor de regex |
| `src/unicode.cpp` | Categorías Unicode y NFC completa con Hangul; tablas generadas por `tools/gen_unicode.py` |
| `src/json.cpp` | Parser JSON para `config.json`, `tokenizer.json` y las cabeceras de safetensors |
| `src/model.cpp` | Carga safetensors (BF16, F16 y F32) por `mmap`, fusiona y cuantiza en paralelo; lee y escribe `.vrb` |
| `src/kernels.cpp` | Cuantización Q8/Q4, productos punto y matmul para NEON, i8mm, AVX2 y escalar |
| `src/attn.cpp` | Atención por bloques con softmax en línea, `exp` y SiLU vectorizados |
| `src/session.cpp` | Forward del transformer, caché KV y RoPE |
| `src/sampler.cpp` | Temperatura, top-k, top-p, min-p y penalización de repetición; generación con texto de parada |
| `cli/serve.cpp` | Servidor HTTP/1.1 con SSE |

**Formato `.vrb`.** Cabecera de 16 bytes (`VRBO`, versión y largo), un JSON con la configuración, el tokenizador original, la plantilla de chat, los tokens de parada y la tabla de tensores, y después los bloques alineados a 64 bytes. Cada tensor cuantizado guarda los enteros y las escalas fp16 por separado, así el kernel lee directo del archivo mapeado.

## Pruebas

```bash
cmake -S . -B build -DVERBO_SANITIZE=ON && cmake --build build -j && ctest --test-dir build
```

- **Suite en C++.** JSON, NFC, tokenizador, fp16, cuantización, matmul contra una referencia escalar en todas las combinaciones de formato, filas, tokens e hilos, atención contra una implementación ingenua en float64, pool de hilos, modelos diminutos contra logits de `transformers`, ida y vuelta por `.vrb`, rebobinado de la caché, muestreo y API en C.
- **Modelos diminutos.** `tests/data` tiene dos modelos aleatorios (Llama en BF16 y Qwen2 en F16), con tokenizadores entrenados para cada familia de pre-tokenización. Los genera `tools/tiny_model.py`.
- **CI.** Corre en Linux x86 (AVX2) y macOS ARM con ASan y UBSan, repite todo con los kernels escalares, prueba el paquete de Python y compara el tokenizador y los logits de SmolLM2 contra Hugging Face.

## Limitaciones

- Solo CPU. No hay Metal, CUDA ni Vulkan.
- No soporta ventana deslizante, mezcla de expertos ni arquitecturas fuera de Llama, Mistral y Qwen2.
- La plantilla de chat se reconoce para ChatML; otros modelos funcionan con `--raw`.
- La caché KV está en float32.

## Ecosistema

verbo es parte de un conjunto de herramientas para software local-first:

| Repo | Papel |
|---|---|
| [hallar](https://github.com/KlGamers/hallar) | Búsqueda semántica sin internet; con verbo forma un RAG completamente local |
| [rumbo](https://github.com/KlGamers/rumbo) | Router de intenciones: decide cuándo vale la pena llamar al modelo |
| [vecdex](https://github.com/KlGamers/vecdex) | Índice vectorial HNSW con SIMD |
| [trama](https://github.com/KlGamers/trama) | Sincronización local-first con CRDTs para web y Flutter |
| [guardia](https://github.com/KlGamers/guardia) | Revisión antes de cada commit |
| [pulso](https://github.com/KlGamers/pulso) | Monitoreo de servicios autohospedados |

## Licencia

MIT. Los modelos tienen su propia licencia; los tres verificados son Apache 2.0.
