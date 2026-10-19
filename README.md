# Despliegue Local: Ollama y Gemma 4

**Práctica 2 — chat dual (local vs remoto) con el mismo modelo en ambos lados.**

Chat por consola contra dos backends que sirven **el mismo modelo** (`gemma4:e4b`):

- **local** — Ollama en este portátil (CPU), `http://127.0.0.1:11434`.
- **remoto** — Ollama de una VM compartida de Azure, alcanzado por **túnel SSH** (`http://127.0.0.1:21434`).

## Contenido

| Fichero | Qué es |
|---|---|
| `chat_dual.py` | Cliente dual (solo `requests`): modos `check`, `chat` y `batch`; métricas por respuesta; flag `--no-think` |
| `prompts_5.json` | Los 5 prompts comparados (idénticos para ambos backends) |
| `informe_despliegue_local.md` | Informe completo: despliegue, comparación (métricas + rúbrica de calidad), reflexión desarrollo/producción, limitaciones y fuentes |
| `requirements.txt` | Dependencia Python (`requests`) |
| `results/` | Evidencias crudas (ver abajo) |

### Evidencias (`results/`)

| Fichero | Contenido |
|---|---|
| `p2_local_deploy_20261019.json` | Verificación del despliegue local (CLI + API REST) |
| `vm_ollama_tags_20261019.json` | Inventario de los Ollama de la VM (uso solo-consulta) |
| `p2_thinking_exploratorio_20261019.log` | Tanda exploratoria con razonamiento activo (respuestas truncadas: motivo del `--no-think`) |
| `chat_dual_20261019.json` + `_tabla.md` | Tanda comparativa final: 5 prompts × 2 backends (respuestas + métricas) |
| `validacion_calidad_20261019.txt` | Validación objetiva de calidad (ejecución del código, parseo del JSON, nº de frases, idioma) |
| `demo_chat_20261019.txt` | Transcripción de la demo del chat interactivo (selección de backend al iniciar) |

## Requisitos y puesta en marcha

```bash
pip install -r requirements.txt
ollama pull gemma4:e4b        # descargar el modelo, si no está ya
```

Para el backend **remoto**, abrir el túnel SSH (los Ollama de la VM escuchan solo en su loopback):

```bash
ssh -N -L 21434:127.0.0.1:11434 -L 21435:127.0.0.1:11435 vm-b_ruben_munoz_blanco
```

## Uso

```bash
python chat_dual.py check             # ¿responden los dos backends? (versión + modelos)
python chat_dual.py chat              # chat interactivo: pregunta el backend al iniciar (local / remoto / ambos)
python chat_dual.py batch             # los 5 prompts de prompts_5.json contra ambos backends
python chat_dual.py batch --no-think  # recomendado con gemma4:e4b (ver nota)
```

`batch` guarda `results/chat_dual_<marca>.json` (respuestas + métricas) y `results/chat_dual_<marca>_tabla.md`.

> **Nota `--no-think`:** `gemma4:e4b` razona por defecto y, sin desactivarlo, un presupuesto corto se consume dentro del bloque `thinking` sin llegar a la respuesta final (`finish=length`). `--no-think` envía `reasoning_effort=none` (API compatible OpenAI de Ollama). Ver §3.2 y §7 del informe.

## Resultado en una línea

Con el mismo modelo en ambos lados, la **calidad fue equivalente** (4/5 prompts correctos en cada uno; el único fallo —el significado de la "E" de `Gemma 4 E4B`— idéntico en los dos). La diferencia está en el **hardware/entorno**: el portátil generó ~**1,7× más rápido** de media (**5,2 tok/s** frente a **3,1 tok/s**; latencias medias **43,4 s** vs **75,1 s**). Detalle en `informe_despliegue_local.md` (§5–§6).
