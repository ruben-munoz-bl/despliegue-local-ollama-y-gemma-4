#!/usr/bin/env python3
"""Chat dual: Ollama LOCAL (localhost:11434) vs API REMOTA de la VM de Azure (tunel SSH).

Practica 2 - Despliegue Local de LLMs.

Modos:
  check  Comprueba que los dos backends responden (version de Ollama + modelos desplegados).
  chat   Chat por consola con eleccion de backend al iniciar: local / remoto / ambos.
  batch  Ejecuta los MISMOS prompts (por defecto prompts_5.json) contra ambos backends,
         mide latencia y tokens/s y guarda resultados en results/chat_dual_<ts>.json
         y una tabla en results/chat_dual_<ts>_tabla.md.

La parte remota requiere el tunel SSH (puertos locales 21434/21435 para no chocar
con el Ollama local, que usa 11434):
  ssh -N -L 21434:127.0.0.1:11434 -L 21435:127.0.0.1:11435 vm-b_ruben_munoz_blanco

Ejemplos:
  python chat_dual.py check
  python chat_dual.py chat                      # pregunta el backend al iniciar
  python chat_dual.py chat --backend ambos
  python chat_dual.py batch --out results/chat_dual_20261019.json

Depende solo de 'requests'. Codigos de salida: 0 = ok, 1 = algun fallo, 2 = configuracion invalida.

Metricas por respuesta: latencia (s), tokens de entrada/salida (usage de la API o estimacion
chars/4), tokens/s, finish_reason ('length' = cortada por --max-tokens) y deteccion de
modelos de razonamiento cuyo texto llega en el campo 'reasoning' con 'content' vacio.

Modelos con razonamiento: gemma4:e4b razona por defecto y puede gastar todo el presupuesto
en el bloque 'thinking' sin llegar a responder. Con --no-think se desactiva por peticion
(reasoning_effort=none en la API compatible OpenAI de Ollama).
"""

import argparse
import datetime as _dt
import json
import os
import sys
import time

try:
    import requests
except ImportError:  # pragma: no cover
    print("Falta la dependencia 'requests'. Ejecuta: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(2)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_LOCAL_URL = "http://127.0.0.1:11434/v1/chat/completions"
DEFAULT_REMOTE_URL = "http://127.0.0.1:21434/v1/chat/completions"
DEFAULT_MODEL = "gemma4:e4b"
DEFAULT_PROMPTS_FILE = "prompts_5.json"

# Fallback si no existe prompts_5.json (mismos textos que el fichero)
BUILTIN_PROMPTS = [
    {
        "id": "p1_razonamiento",
        "categoria": "razonamiento matematico",
        "prompt": "Un tren sale de la estación a las 9:00 a 60 km/h. Otro tren sale de la misma estación a las 9:30 en sentido contrario a 90 km/h. ¿A qué hora estarán separados por 300 km? Razona el cálculo paso a paso.",
    },
    {
        "id": "p2_codigo",
        "categoria": "programacion",
        "prompt": "Escribe una función Python llamada primos(n) que devuelva una lista con los primeros n números primos. Devuelve solo el código en un bloque ```python``` seguido de una línea de comentario con su complejidad temporal.",
    },
    {
        "id": "p3_resumen",
        "categoria": "resumen",
        "prompt": "Resume el siguiente texto en exactamente 2 frases:\n\n\"La inferencia local de modelos de lenguaje permite ejecutar modelos en el propio ordenador sin enviar datos a servicios externos. A cambio, está limitada por el hardware disponible: memoria RAM o VRAM, velocidad de CPU/GPU y capacidad de disco. La inferencia en la nube, en cambio, ofrece acceso a modelos mucho más grandes y hardware especializado, pero introduce dependencia de la red, costes por uso y consideraciones de privacidad.\"",
    },
    {
        "id": "p4_json",
        "categoria": "formato / salida estructurada",
        "prompt": "Devuelve SOLO un JSON válido, sin texto adicional y sin bloques de código, con estos campos: nombre (string), edad (número), ciudad (string), profesion (string). Inventa los datos de una persona.",
    },
    {
        "id": "p5_ingles",
        "categoria": "conocimiento + idioma (ingles)",
        "prompt": "Answer in English in exactly 2 sentences: what does the 'E' in 'Gemma 4 E4B' stand for, and how many parameters does that model have?",
    },
]


# ---------------------------------------------------------------------------
# Utilidades (misma logica de metricas que inference_azure.py, practica 1)
# ---------------------------------------------------------------------------

def approx_tokens(text: str) -> int:
    """Estimacion grosera de tokens (~4 caracteres por token). Solo si la API no da 'usage'."""
    return max(1, round(len(text) / 4))


def extract_content(data: dict):
    """Devuelve (texto, reasoning, usado_reasoning). Tolerante a modelos de razonamiento:
    gemma4:e4b puede devolver 'content' vacio y el texto completo en 'reasoning'."""
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("respuesta sin 'choices'")
    message = choices[0].get("message") or {}

    def flatten(value):
        if isinstance(value, list):
            value = "".join(part.get("text", "") for part in value if isinstance(part, dict))
        return value if isinstance(value, str) else str(value or "")

    content = flatten(message.get("content", "")).strip()
    reasoning = flatten(message.get("reasoning", "")).strip()
    if content:
        return content, (reasoning or None), False
    if reasoning:
        return reasoning, reasoning, True
    raise ValueError("respuesta con contenido vacio")


def chat_once(target: dict, messages: list, args) -> dict:
    """Una llamada a un backend (estilo OpenAI/chat). Lanza excepcion si falla."""
    payload = {
        "model": target["model"],
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    if getattr(args, "no_think", False):
        payload["reasoning_effort"] = "none"
    started = time.perf_counter()
    resp = requests.post(target["url"], json=payload, timeout=args.timeout)
    elapsed = time.perf_counter() - started
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    content, reasoning, used_reasoning = extract_content(data)

    finish_reason = None
    choices = data.get("choices") or []
    if choices:
        finish_reason = choices[0].get("finish_reason")

    usage = data.get("usage") or {}
    tokens_out = usage.get("completion_tokens")
    tokens_in = usage.get("prompt_tokens")
    source = "api"
    if not isinstance(tokens_out, int):
        tokens_out = approx_tokens(content)
        source = "estimado (chars/4)"
    if not isinstance(tokens_in, int):
        tokens_in = approx_tokens(" ".join(m["content"] for m in messages))

    return {
        "model": target["model"],
        "url": target["url"],
        "ok": True,
        "latency_s": round(elapsed, 3),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_source": source,
        "tokens_per_s": round(tokens_out / elapsed, 2) if elapsed > 0 else None,
        "chars": len(content),
        "finish_reason": finish_reason,
        "reasoning_chars": len(reasoning) if reasoning else 0,
        "used_reasoning_as_answer": used_reasoning,
        "response": content,
        "error": None,
    }


def chat_safe(target: dict, messages: list, args) -> dict:
    """Como chat_once pero devuelve un registro de fallo en vez de lanzar excepcion."""
    try:
        return chat_once(target, messages, args)
    except requests.exceptions.Timeout:
        err = f"timeout tras {args.timeout}s"
    except requests.exceptions.ConnectionError as exc:
        err = f"error de conexion a {target['url']} ({exc.__class__.__name__})"
    except (ValueError, KeyError) as exc:
        err = f"respuesta malformada: {exc}"
    except RuntimeError as exc:
        err = str(exc)
    except Exception as exc:  # noqa: BLE001 - no abortar la tanda completa
        err = f"{exc.__class__.__name__}: {exc}"
    return {
        "model": target["model"],
        "url": target["url"],
        "ok": False,
        "latency_s": None,
        "tokens_in": None,
        "tokens_out": None,
        "tokens_source": None,
        "tokens_per_s": None,
        "chars": None,
        "finish_reason": None,
        "reasoning_chars": None,
        "used_reasoning_as_answer": None,
        "response": None,
        "error": err,
    }


def ollama_base(url: str) -> str:
    """Deriva la base de Ollama a partir del endpoint OpenAI (/v1/chat/completions)."""
    return url.split("/v1/")[0].rstrip("/")


# ---------------------------------------------------------------------------
# Modo check
# ---------------------------------------------------------------------------

def cmd_check(args) -> int:
    ok_all = True
    for side in ("local", "remoto"):
        t = args.targets[side]
        base = ollama_base(t["url"])
        try:
            version = requests.get(base + "/api/version", timeout=5).json().get("version")
            tags = requests.get(base + "/api/tags", timeout=15).json().get("models", [])
            names = [m.get("name") for m in tags]
            print(f"[OK]    {side:<6} {base:<28} Ollama {version} | modelos: {', '.join(names) or '-'}")
        except Exception as exc:  # noqa: BLE001
            ok_all = False
            print(f"[FALLO] {side:<6} {base:<28} {exc.__class__.__name__}: {exc}")
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------
# Modo chat
# ---------------------------------------------------------------------------

def _ask_backend() -> str:
    print("Elige backend para el chat:")
    print("  1) local  (Ollama en localhost:11434)")
    print("  2) remoto (VM de Azure via tunel SSH -> 127.0.0.1:21434)")
    print("  3) ambos  (mismo prompt a los dos)")
    while True:
        try:
            sel = input("Backend [1/2/3]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return "local"
        if sel in ("1", "local", "l"):
            return "local"
        if sel in ("2", "remoto", "r"):
            return "remoto"
        if sel in ("3", "ambos", "a"):
            return "ambos"
        print("Opcion no valida: escribe 1, 2 o 3.")


def cmd_chat(args) -> int:
    targets = args.targets
    backend = args.backend or _ask_backend()
    active = {"local": ["local"], "remoto": ["remoto"], "ambos": ["local", "remoto"]}[backend]

    desc = ", ".join(f"{side} ({targets[side]['model']})" for side in active)
    print(f"\nChat con: {desc}")
    print("Comandos: 'salir' para terminar | 'reset' para borrar el historial.\n")

    histories = {
        side: ([{"role": "system", "content": args.system}] if args.system else [])
        for side in active
    }

    while True:
        try:
            line = input("tu> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        low = line.lower()
        if low in ("salir", "exit", "quit", "q"):
            break
        if low == "reset":
            for side in active:
                histories[side] = ([{"role": "system", "content": args.system}] if args.system else [])
            print("(historial borrado)\n")
            continue
        for side in active:
            t = targets[side]
            msgs = histories[side] + [{"role": "user", "content": line}]
            r = chat_safe(t, msgs, args)
            if r["ok"]:
                histories[side].append({"role": "user", "content": line})
                histories[side].append({"role": "assistant", "content": r["response"]})
                meta = (
                    f"{r['latency_s']} s | {r['tokens_per_s']} tok/s | "
                    f"{r['tokens_out']} tok out | finish={r['finish_reason']}"
                    + (" | [respuesta desde 'reasoning']" if r["used_reasoning_as_answer"] else "")
                )
                print(f"\n--- {side} ({t['model']}) --- {meta}\n{r['response']}\n")
            else:
                print(f"\n--- {side} ({t['model']}) --- FALLO: {r['error']}\n")

    print("Fin del chat.")
    return 0


# ---------------------------------------------------------------------------
# Modo batch
# ---------------------------------------------------------------------------

def load_prompts(path):
    if path is None and os.path.exists(DEFAULT_PROMPTS_FILE):
        path = DEFAULT_PROMPTS_FILE
    if path:
        with open(path, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data = data.get("prompts", [])
        prompts = [p for p in data if isinstance(p, dict) and p.get("prompt")]
        if not prompts:
            raise ValueError(f"'{path}' no contiene prompts validos")
        print(f"Prompts cargados de: {path} ({len(prompts)})")
        return prompts
    print(f"Aviso: no existe {DEFAULT_PROMPTS_FILE}; usando los 5 prompts integrados.")
    return BUILTIN_PROMPTS


def build_markdown_table(prompts: list, results: list) -> str:
    by_key = {(r["prompt_id"], r["side"]): r for r in results}
    lines = [
        "| Prompt | Backend | OK | Latencia (s) | Tok out | Tok/s | finish |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for p in prompts:
        pid = p.get("id", "?")
        for side in ("local", "remoto"):
            r = by_key.get((pid, side))
            if not r:
                continue
            if r["ok"]:
                lines.append(
                    f"| {pid} | {side} | si | {r['latency_s']} | {r['tokens_out']} | "
                    f"{r['tokens_per_s']} | {r['finish_reason']} |"
                )
            else:
                lines.append(f"| {pid} | {side} | NO | - | - | - | - |")
    for side in ("local", "remoto"):
        rs = [r for r in results if r["side"] == side and r["ok"]]
        total = sum(1 for r in results if r["side"] == side)
        if rs:
            lat = round(sum(r["latency_s"] for r in rs) / len(rs), 3)
            tps = round(sum(r["tokens_per_s"] for r in rs) / len(rs), 2)
            out = round(sum(r["tokens_out"] for r in rs) / len(rs), 1)
            lines.append(f"| _media_ | {side} | {len(rs)}/{total} | {lat} | {out} | {tps} | |")
        else:
            lines.append(f"| _media_ | {side} | 0/{total} | - | - | - | |")
    return "\n".join(lines)


def cmd_batch(args) -> int:
    try:
        prompts = load_prompts(args.prompts)
    except (OSError, ValueError) as exc:
        print(f"ERROR: no se pudieron cargar los prompts: {exc}", file=sys.stderr)
        return 2

    sides = ("local", "remoto")
    results = []
    for i, p in enumerate(prompts, 1):
        pid = p.get("id") or f"p{i}"
        for side in sides:
            t = args.targets[side]
            msgs = ([{"role": "system", "content": args.system}] if args.system else []) + [
                {"role": "user", "content": p["prompt"]}
            ]
            print(f"[{i}/{len(prompts)}] {pid} x {side} ({t['model']}) ...", flush=True)
            r = chat_safe(t, msgs, args)
            r.update({"prompt_id": pid, "categoria": p.get("categoria"), "side": side})
            results.append(r)
            if r["ok"]:
                extra = " | desde 'reasoning'" if r["used_reasoning_as_answer"] else ""
                print(
                    f"    ok: {r['latency_s']} s | {r['tokens_out']} tok out | "
                    f"{r['tokens_per_s']} tok/s | finish={r['finish_reason']}{extra}"
                )
            else:
                print(f"    FALLO: {r['error']}", file=sys.stderr)

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join("results", f"chat_dual_{stamp}.json")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    record = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "targets": {s: {"url": args.targets[s]["url"], "model": args.targets[s]["model"]} for s in sides},
        "params": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "timeout": args.timeout,
            "system": args.system,
            "no_think": bool(getattr(args, "no_think", False)),
        },
        "prompts": prompts,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados en: {out_path}")

    table = build_markdown_table(prompts, results)
    table_path = out_path.rsplit(".", 1)[0] + "_tabla.md"
    with open(table_path, "w", encoding="utf-8") as fh:
        fh.write(table + "\n")
    print("\n" + table)
    print(f"\nTabla markdown guardada en: {table_path}")

    return 0 if all(r["ok"] for r in results) else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--local-url", default=DEFAULT_LOCAL_URL, help="endpoint OpenAI-compatible del Ollama LOCAL")
    common.add_argument("--local-model", default=DEFAULT_MODEL, help="modelo del backend local")
    common.add_argument("--remote-url", default=DEFAULT_REMOTE_URL, help="endpoint del Ollama REMOTO (VM via tunel SSH)")
    common.add_argument("--remote-model", default=DEFAULT_MODEL, help="modelo del backend remoto")
    common.add_argument("--max-tokens", type=int, default=1000, help="maximo de tokens de salida por respuesta (incluye el bloque 'thinking' de los modelos razonadores)")
    common.add_argument("--temperature", type=float, default=0.2, help="temperatura de muestreo")
    common.add_argument("--no-think", action="store_true", help="desactiva el razonamiento del modelo (reasoning_effort=none)")
    common.add_argument("--timeout", type=float, default=900.0, help="timeout HTTP en segundos")
    common.add_argument("--system", default=None, help="mensaje de sistema opcional (identico en ambos backends)")

    p = argparse.ArgumentParser(
        description="Chat dual: Ollama local (localhost:11434) vs API remota de la VM de Azure (tunel SSH).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    sc = sub.add_parser("check", parents=[common], help="comprueba que ambos backends responden")
    sc.set_defaults(func=cmd_check)

    sch = sub.add_parser("chat", parents=[common], help="chat interactivo por consola")
    sch.add_argument(
        "--backend",
        choices=("local", "remoto", "ambos"),
        default=None,
        help="backend a usar; si se omite, se pregunta al iniciar",
    )
    sch.set_defaults(func=cmd_chat)

    sb = sub.add_parser("batch", parents=[common], help="mismos prompts contra ambos backends (JSON + tabla)")
    sb.add_argument(
        "--prompts",
        default=None,
        help=f"JSON con [{{'id','categoria','prompt'}}]; por defecto {DEFAULT_PROMPTS_FILE}",
    )
    sb.add_argument("--out", default=None, help="ruta del JSON de salida (por defecto results/chat_dual_<ts>.json)")
    sb.set_defaults(func=cmd_batch)

    args = p.parse_args(argv)
    args.targets = {
        "local": {"name": "local", "url": args.local_url, "model": args.local_model},
        "remoto": {"name": "remoto", "url": args.remote_url, "model": args.remote_model},
    }
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
