# Práctica 2 — Despliegue local de LLMs: chat local vs remoto

> **Fecha:** 19-oct-2026
> **Entorno local:** portátil HP ProBook 440 G8 — Intel i5-1135G7 (4c/8t), 16 GB RAM, **sin GPU dedicada** (Iris Xe), 366 GB libres.
> **Entorno remoto:** VM de Azure de Inetum, acceso por **túnel SSH** (las APIs de la VM no están publicadas a internet).
> **Código:** `chat_dual.py` · **Evidencia:** `results/chat_dual_*.json` + `results/chat_dual_*_tabla.md` · **Prompts:** `prompts_5.json`

---

## 1. Ollama: arquitectura y conceptos

*Fuente: documentación oficial (`docs.ollama.com`: Windows, Quickstart, API, CLI).*

### 1.1 Componentes

Ollama es un **runtime local de LLMs**: un servidor que ejecuta modelos en tu máquina y los expone por CLI y API REST.

| Componente | Qué es |
|---|---|
| **Servidor (daemon)** | Escucha en `http://localhost:11434` (solo loopback). Gestiona la carga/descarga de modelos en memoria, el planificador de peticiones y el motor de inferencia (**llama.cpp**, con backends CUDA / ROCm / Vulkan y *fallback* a CPU). En Windows se instala como aplicación nativa que corre en segundo plano (icono de bandeja). |
| **CLI (`ollama`)** | Cliente del servidor y gestor de modelos: `run`, `pull`, `ls`, `ps`, `show`, `stop`, `rm`, `create`… |
| **API REST** | Dos superficies: **API nativa** `/api/*` y **compatibilidad OpenAI** `/v1/*` (además de compatibilidad Anthropic `/v1/messages`). |
| **Almacén de modelos** | Directorio del usuario (`%HOMEPATH%\.ollama` en Windows), con manifiestos + pesos en formato **GGUF** y sus **cuantizaciones** (`Q4_K_M`, `Q8_0`…), que se eligen por *tag* (`gemma4:e4b`, `gemma4:e2b`…). |

### 1.2 Cómo viaja una petición

```
script Python  ──HTTP POST──▶  localhost:11434/v1/chat/completions
                                    │
                                    ▼
                        servidor Ollama: resuelve el modelo
                        (lo carga en RAM si no lo está) → tokeniza
                                    │
                                    ▼
                        llama.cpp: inferencia (CPU o GPU)
                                    │
                                    ▼
                        JSON de respuesta: choices[0].message.content,
                        usage (tokens), finish_reason
```

- Los modelos permanecen cargados en memoria un tiempo tras la última petición (**keep-alive**, ~5 min por defecto; configurable con `OLLAMA_KEEP_ALIVE`).
- La **primera** petición a un modelo incluye su carga en RAM → latencia alta; las siguientes son más rápidas.
- Configuración por variables de entorno: `OLLAMA_HOST` (interfaz/puerto), `OLLAMA_MODELS` (ruta de modelos), `OLLAMA_KEEP_ALIVE`, etc. (`ollama serve --help` las lista).

### 1.3 API REST en detalle

| Superficie | URL local | Endpoints principales |
|---|---|---|
| **Nativa Ollama** | `http://localhost:11434/api` | `chat`, `generate`, `tags`, `show`, `ps`, `pull`, `delete`… |
| **OpenAI-compatible** | `http://localhost:11434/v1` | `chat/completions`, `responses` → usado por `chat_dual.py` |
| **Anthropic-compatible** | `http://localhost:11434/v1/messages` | alternativa para clientes Anthropic |

Notas: las peticiones **locales no requieren API key**; crear o borrar modelos **solo** es posible contra un servidor local (no contra la nube). La API es estable y retrocompatible, sin versionado estricto.

### 1.4 Gestión de modelos

| Comando | Para qué |
|---|---|
| `ollama pull gemma4:e4b` | Descargar un modelo (por *tag*) |
| `ollama run gemma4:e4b` | Descargar (si falta) y abrir chat interactivo |
| `ollama ls` / `ollama ps` | Modelos descargados / cargados en memoria |
| `ollama show gemma4:e4b` | Ficha técnica: parámetros, cuantización, contexto, capacidades |
| `ollama stop` / `ollama rm` | Descargar de RAM / borrar del disco |
| `ollama create -f Modelfile` | Crear variantes propias (p. ej. con `SYSTEM` fijo) partiendo de un modelo base |

### 1.5 Instalación en Windows (sin administrador)

- El instalador oficial (`OllamaSetup.exe`) **no requiere administrador**: instala en `%LOCALAPPDATA%\Programs\Ollama` y añade el binario al **PATH del usuario**.
- Rutas: modelos en `%HOMEPATH%\.ollama`, logs en `%LOCALAPPDATA%\Ollama` (`server.log`, `app.log`).
- Alternativas oficiales: script `irm https://ollama.com/install.ps1 | iex` y zip *standalone* `ollama-windows-amd64.zip` (CLI + librerías GPU de NVIDIA; útil para servicios).
- **Seguridad de red:** por defecto el servidor escucha **solo en loopback** (127.0.0.1); no es accesible desde la red. En esta práctica el Ollama del portátil se queda así, y para el de la VM se usa **túnel SSH** en vez de exponer puertos.

---

## 2. Local vs nube: marco de comparación

| Dimensión | **Local** (portátil con Ollama) | **Nube / servidor remoto** (VM de Azure) |
|---|---|---|
| **Modelo máximo** | Limitado por RAM/VRAM (~16 GB aquí → hasta ~8B cuantizado) | Modelos grandes (30B+ F16) en GPUs especializadas |
| **Latencia** | Sin red; primer token depende de carga del modelo + velocidad de CPU/GPU propia | Incluye red; cola compartida si el servidor es multiusuario |
| **Privacidad** | Los datos **no salen** del equipo | Los datos viajan a (y se procesan en) otro sistema |
| **Dependencia de red** | Ninguna (funciona offline) | Total: sin red, no hay servicio |
| **Coste** | Fijo (hardware ya comprado; electricidad) | Por uso / instancia / tiempo de GPU |
| **Escalado** | 1 usuario (unos pocos procesos en paralelo según RAM) | Muchos usuarios concurrentes |
| **Mantenimiento** | Manual (actualizar Ollama, gestionar modelos) | Del proveedor/administrador |
| **Reproducibilidad** | Exacta (misma máquina, mismo modelo) | Depende del despliegue y de la carga ajena |

Esta práctica mide el contraste con un caso real: **el mismo modelo (`gemma4:e4b`) en los dos entornos**, para que la diferencia observada sea de *hardware/entorno*, no de modelo.

---

## 3. Despliegue realizado (medido)

### 3.1 Ollama en el portátil (local)

| Paso | Resultado |
|---|---|
| Instalación | `winget` **no utilizable** en este equipo (`0x8a15000f`; su arreglo requiere administrador) → se usó el script **oficial** `install.ps1` de ollama.com |
| Seguridad | El script verificó la **firma Authenticode** del instalador antes de ejecutarlo: *válida — "Ollama Inc." (Toronto, CA)* |
| Modo | Silencioso (`/VERYSILENT /NORESTART /SUPPRESSMSGBOXES`), **sin administrador**, en `%LOCALAPPDATA%\Programs\Ollama` |
| Versión instalada | **0.40.6** |
| Servidor | Escucha en `127.0.0.1:11434` (solo loopback; no expuesto a la red) |
| Primer arranque | **~5 min**: descubrimiento de backends GPU secuencial (Vulkan ≈ 78 s + CUDA ≈ 90 s hasta *timeout* — este equipo no tiene GPU NVIDIA). Después queda operativo con normalidad |
| iGPU | La Iris Xe aparece como **1 dispositivo Vulkan** (`OLLAMA_VULKAN=true` por defecto), pero Ollama acaba usando **CPU** (ver §3.2) |

### 3.2 Verificación local (CLI + API REST)

| Verificación | Resultado |
|---|---|
| Descarga del modelo | `gemma4:e4b` — **6,6 GB** en **250 s** (~28 MB/s); **Q4_K_M**, **7,5B** parámetros, contexto **131 072**, capacidades `completion`/`vision`/`audio`/`tools`/**`thinking`** (activo por defecto), proyector `clip` de 478M |
| CLI | Prueba *one-shot* por stdin (`ollama run gemma4:e4b`) → respuesta correcta en **81,9 s** (incluye la carga del modelo en RAM) |
| REST `POST /api/generate` | HTTP 200 — **160 tokens en 21,4 s ≈ 8,2 tok/s** |
| REST `POST /v1/chat/completions` | HTTP 200 — **300 tokens en 38,4 s ≈ 7,8 tok/s** |
| Backend de inferencia | `ollama ps` → **100% CPU** (la iGPU Iris Xe no se usa; coherente con hardware sin GPU soportada) |

> **Hallazgo (importante para §5):** `gemma4:e4b` **razona antes de responder** (el razonamiento va en el campo `thinking` y la respuesta final en `content`). Con el razonamiento activo, un presupuesto de 1000 tokens se consume *íntegramente dentro del razonamiento* y `content` queda vacío (`finish_reason=length`) — comprobado en una tanda exploratoria (evidencia: `results/p2_thinking_exploratorio_20261019.log`). Para obtener respuestas completas y comparables, las pruebas de §5 se ejecutan con **`--no-think`** (`reasoning_effort=none`; soportado por Ollama 0.40.x y verificado en ambos backends) y **`--max-tokens 1600`** como tope de seguridad.
>
> Evidencia cruda: `results/p2_local_deploy_20261019.json`.

### 3.3 Backend remoto (VM de Azure)

| Dato | Valor |
|---|---|
| Acceso | Túnel SSH `-L 21434:127.0.0.1:11434 -L 21435:127.0.0.1:11435` (puertos locales **21434/21435** para no chocar con el Ollama local) |
| Versión | Ollama **0.40.5** en la VM |
| Modelos | `21434` → `gemma4:e4b` y `qwen2:0.5b` · `21435` → `gemma:latest` |
| Propiedad | Instancias de **otros usuarios** de la VM compartida → uso solo-consulta |

---

## 4. El script: `chat_dual.py`

Script Python (solo `requests`) que habla con **dos backends** configurables:

| Backend | URL por defecto | Modelo por defecto |
|---|---|---|
| `local` | `http://127.0.0.1:11434/v1/chat/completions` (Ollama del portátil) | `gemma4:e4b` |
| `remoto` | `http://127.0.0.1:21434/v1/chat/completions` (VM vía túnel SSH) | `gemma4:e4b` |

> El túnel usa el puerto local **21434** (no 11434) para no chocar con el Ollama local:
> `ssh -N -L 21434:127.0.0.1:11434 -L 21435:127.0.0.1:11435 vm-b_ruben_munoz_blanco`

Tres modos:

```bash
python chat_dual.py check                          # ¿responden los dos backends? (versión + modelos)
python chat_dual.py chat                           # chat por consola; pregunta el backend al iniciar
python chat_dual.py chat --backend ambos           # mismo prompt a los dos backends
python chat_dual.py batch                          # los 5 prompts de prompts_5.json a ambos backends
```

- **`chat`**: conversación con historial por backend; comandos `salir` y `reset`. Al iniciar permite elegir **local / remoto / ambos** (o se fija con `--backend`). Ejemplo real transcrito en `results/demo_chat_20261019.txt`.
- **`batch`**: ejecuta los **mismos 5 prompts** contra ambos backends, guarda `results/chat_dual_<ts>.json` y una tabla markdown de métricas en `results/chat_dual_<ts>_tabla.md`.
- **Métricas por respuesta**: latencia (s), tokens de entrada/salida (de `usage` o estimados), **tokens/s**, `finish_reason` (`length` = cortada por `--max-tokens`) y detección de modelos de razonamiento (texto en el campo `reasoning` con `content` vacío).
- **Razonamiento desactivable**: `--no-think` envía `reasoning_effort=none` (API compatible OpenAI) para que los modelos *thinking* respondan directamente, sin gastar el presupuesto en el bloque `thinking` (usado en §5).
- Un fallo de un backend **no rompe la tanda**: se registra `ok:false` + error y se continúa.

---

## 5. Resultados: los mismos 5 prompts en local y remoto

| Prompt | Categoría |
|---|---|
| `p1_razonamiento` | Razonamiento matemático (trenes; solución única y comprobable) |
| `p2_codigo` | Programación (función `primos(n)` + complejidad) |
| `p3_resumen` | Resumen en 2 frases |
| `p4_json` | Salida estructurada (JSON válido con 4 campos) |
| `p5_ingles` | Conocimiento + idioma (respuesta en inglés sobre Gemma 4 E4B) |

> **Ejecución:** 2026-10-19 · mismo modelo `gemma4:e4b` en ambos backends · `--no-think --max-tokens 1600 --temperature 0.2` · 1 muestra por prompt/backend.
> Evidencia: `results/chat_dual_20261019.json` (+ `_tabla.md`); validación objetiva: `results/validacion_calidad_20261019.txt`.

### 5.1 Tiempo y velocidad

| Prompt | Local lat. (s) | Local tok/s | Remoto lat. (s) | Remoto tok/s | Ratio tok/s (local/remoto) |
|---|---:|---:|---:|---:|---:|
| `p1_razonamiento` | 156,6 | 5,98 | 268,1 | 3,65 | 1,64× |
| `p2_codigo` | 28,6 | 6,74 | 53,3 | 3,87 | 1,74× |
| `p3_resumen` | 13,4 | 3,73 | 18,8 | 2,77 | 1,35× |
| `p4_json` | 9,0 | 3,65 | 12,3 | 2,67 | 1,37× |
| `p5_ingles` | 9,1 | 6,02 | 23,0 | 2,44 | 2,47× |
| **Media** | **43,4** | **5,22** | **75,1** | **3,08** | **1,70×** |

- **10/10 respuestas completas** (`finish=stop`): sin errores ni *timeouts*.
- El portátil (CPU; `ollama ps` = 100% CPU) genera de media **~1,7× más rápido** que la VM compartida.
- Los tok/s del local fluctúan (3,6–6,7) porque su CPU se comparte con el resto del sistema; el remoto fue más estable (2,4–3,9).
- Tamaño de las generaciones: p1 ≈950 tokens (el 90% del total de la tanda), p2 ≈200 y p3–p5 entre 33 y 56 tokens.
- En respuestas cortas (p3–p5) pesa más el tiempo fijo (cola/carga del modelo) que la generación: 9–23 s.

### 5.2 Calidad (rúbrica con validación objetiva)

Criterios: **corrección comprobable** (respuesta única / ejecución de código / parseo de JSON), **formato e instrucciones** (nº de frases, solo-JSON, bloque de código) e **idioma**.

| Prompt | Criterio objetivo | Local | Remoto | Observaciones |
|---|---|---|---|---|
| `p1` | Resultado 11:18 con pasos correctos | Sí | Sí | Ambos: 30 km de ventaja → 270 km a 150 km/h → 1,8 h → **11:18**. Bien formateado, con LaTeX |
| `p2` | `primos(n)` ejecutable + complejidad | Sí | Sí | **Ejecutado**: `primos(10)` = `[2,3,5,7,11,13,17,19,23,29]` en ambos. Complejidad citada: O(n·√pₙ). Matiz: la complejidad va como línea de texto tras el bloque, no como comentario `#` (desviación menor, igual en ambos) |
| `p3` | Exactamente 2 frases, fiel al texto | Sí | Sí | Dos frases; cubren los dos lados (límites de hardware vs escala, red y costes) |
| `p4` | Solo JSON, 4 campos y tipos | Sí | Sí | Ambos: mismo JSON válido sin texto extra (`nombre`, `edad`, `ciudad`, `profesion`), validado con *parser* |
| `p5` | Inglés, 2 frases, dato correcto | Parcial | Parcial | Idioma y formato OK, pero **fallo común de conocimiento**: "E" = *"Efficient"* (local) / *"Estimated"* (remoto). Dato oficial: "E" = ***effective*** (parámetros efectivos ≈4B; total ~7,5B). Ambos aciertan el orden (4B) |

**Lectura:** con el mismo modelo, la calidad es **equivalente** en los dos entornos (4/5 correctos en cada lado y el mismo error compartido en p5) → la diferencia local/remoto de esta práctica es de **entorno/hardware (velocidad), no de calidad**. El fallo de p5 no es de formato, sino de conocimiento del modelo pequeño.

---

## 6. Conclusiones: ¿cuándo usar cada opción?

**Desarrollo vs producción — lo que dicen los datos:**

1. **La calidad la fija el modelo, no el entorno.** El mismo `gemma4:e4b` en ambos lados dio 4/5 prompts correctos en cada uno y el mismo error en p5 → mover *el mismo modelo* a otro hardware no cambia lo que sabe.
2. **La velocidad la fija el hardware, no la etiqueta "nube".** El portátil (CPU) fue **~1,7× más rápido** que la VM compartida (también CPU). "Nube" no implica más rápido: solo con GPU dedicada y recursos aislados; una VM multiusuario sin acelerador puede quedar por debajo de un portátil.
3. **Desarrollo → local.** Iterar prompts y scripts con latencias de 9–157 s y respuestas completas, sin túnel, sin coste por uso y con privacidad total es suficiente y cómodo. Al usar Ollama en ambos extremos, lo probado en local se traslada tal cual (mismos prompts, misma API).
4. **Producción → servicio gestionado con GPU.** Lo frágil del "modo VM" no es la nube en sí: es el andamiaje (túnel SSH manual que se cayó una vez, proceso de otro usuario, CPU compartida, sin SLA). Producción exige API estable, aislamiento y concurrencia; el coste pasa a ser por uso/GPU.
5. **Criterio de decisión**: datos sensibles o cargas pequeñas/esporádicas → local; modelos grandes, muchos usuarios o latencias exigentes → nube con GPU. En ambos casos, la misma API (Ollama) permite cambiar de entorno sin reescribir la aplicación.

---

## 7. Supuestos y limitaciones

- Túnel SSH obligatorio para el backend remoto (las APIs de la VM escuchan en su loopback).
- El Ollama remoto pertenece a **otro usuario** de la VM compartida: se usa en modo solo-consulta y su rendimiento puede fluctuar con la carga de otros.
- La comparación es funcional (5 prompts), **no un benchmark**: 1 muestra por prompt y backend.
- El portátil no tiene GPU compatible con Ollama (Iris Xe) → inferencia por **CPU** (confirmado: `ollama ps` = `100% CPU`).
- `gemma4:e4b` **razona por defecto** (bloque `thinking`): con él activo, presupuestos de 1000 tokens se agotan sin llegar a la respuesta (tanda exploratoria). La comparación de §5 se ejecuta con `--no-think` **por igual en ambos backends**: mide respuesta directa, no la traza de razonamiento.
- El **túnel SSH** sufrió un *reset* puntual (*connection reset*) y hubo que reabrirlo; si cae a mitad de una tanda, `chat_dual.py` registra el error por respuesta sin abortar.

---

## 8. Fuentes

| Fuente | Uso |
|---|---|
| `docs.ollama.com/windows` | Instalación en Windows, rutas, requisitos |
| `docs.ollama.com/quickstart` | Flujo básico, ejemplo local, descarga de modelos |
| `docs.ollama.com/api/introduction` | Superficies de API y URLs base |
| `docs.ollama.com/cli` | Referencia de comandos y Modelfile |
| `ollama.com/download` | Instaladores oficiales (`OllamaSetup.exe`, `install.ps1`, zip standalone) |
| `docs.ollama.com/capabilities/thinking` | Capacidad de razonamiento: campo `thinking` y controles (`think`) |
| `docs.ollama.com/api/openai-compatibility` | API compatible OpenAI: parámetro `reasoning_effort` (valor `"none"` desactiva el razonamiento) |
| Práctica 1 (`comparativa.md`) | Contexto de la VM de Azure y de los modelos desplegados |
