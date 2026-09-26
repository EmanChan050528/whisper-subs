"""The model seam. Port of jp-subs' core/backends.js (the Ollama half).

A backend is a callable: (prompt, json=False) -> str. Keeping it that small is
what lets the parity tests drive the pipeline with a fake.

Standard library only: one POST to /api/chat does not justify a dependency.
"""

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable

Backend = Callable[..., str]

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "qwen3.5:9b"


class BackendError(Exception):
    pass


def _host_url(host: str) -> str:
    # OLLAMA_HOST is also the server's own bind setting, so it is often just
    # "127.0.0.1:11434" (no scheme, which urllib needs) or "0.0.0.0" (bind
    # everywhere, which is not an address a Windows client can connect to).
    url = host if re.match(r"^https?://", host) else f"http://{host}"
    return url.replace("://0.0.0.0", "://127.0.0.1")


def check_model(model: str, host: str = DEFAULT_HOST) -> None:
    """Fail in seconds, before a long transcription, if translation cannot run."""
    try:
        with urllib.request.urlopen(_host_url(host).rstrip("/") + "/api/tags", timeout=10) as r:
            names = {m.get("name") for m in json.loads(r.read().decode("utf-8")).get("models", [])}
    except (urllib.error.URLError, TimeoutError, ValueError) as err:
        raise BackendError(
            f"Cannot reach Ollama at {host}. Start the Ollama app (a bare `ollama serve` "
            f"may not see the app's models), or pass --ja-only. ({err})"
        ) from err
    wanted = model if ":" in model else f"{model}:latest"
    if wanted not in names:
        have = ", ".join(sorted(n for n in names if n)) or "none"
        raise BackendError(
            f'Ollama has no model "{model}". Pull it with: ollama pull {model}  (installed: {have})'
        )


def ollama_backend(
    model: str = DEFAULT_MODEL,
    num_ctx: int = 16384,
    num_predict: int = 8192,
    temperature: float = 0.2,
    think: bool = False,
    host: str = DEFAULT_HOST,
    timeout_s: float = 15 * 60,
) -> Backend:
    """Local Qwen, or any Ollama model.

    Two settings are load-bearing, both learned the hard way in jp-subs:

    num_ctx: Ollama's default context silently truncates a chunk carrying
    context units, producing quietly worse output rather than an error.

    think: Qwen3.5 is a reasoning model. Left on, its reasoning consumes the
    whole num_predict budget and `content` comes back EMPTY. Translation does
    not need it, and turning it off is several times faster.
    """
    url = _host_url(host).rstrip("/") + "/api/chat"

    def ollama(prompt: str, json_mode: bool = False) -> str:
        body = {
            "model": model,
            "stream": False,
            "think": think,
            "options": {"temperature": temperature, "num_ctx": num_ctx,
                        "num_predict": num_predict},
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_mode:
            body["format"] = "json"
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as res:
                data = json.loads(res.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")[:300]
            if err.code == 404:
                raise BackendError(
                    f'Ollama has no model "{model}". Pull it first: ollama pull {model}'
                ) from err
            raise BackendError(f"Ollama returned HTTP {err.code}: {detail}") from err
        except TimeoutError as err:
            raise BackendError(
                f"{model} did not answer within {round(timeout_s / 60)} minutes. On a "
                f"slower machine this usually means the model is too large; try a smaller "
                f'one (e.g. qwen3.5:4b). "ollama ps" shows whether it is on GPU or CPU.'
            ) from err
        except urllib.error.URLError as err:
            raise BackendError(
                f"Cannot reach Ollama at {host}. Is it running? Start the Ollama app "
                f"(a bare `ollama serve` may not see the app's models), then "
                f'"ollama pull {model}". ({err.reason})'
            ) from err

        message = data.get("message") or {}
        text = message.get("content")
        if not text:
            thinking = message.get("thinking") or ""
            if thinking:
                raise BackendError(
                    f"{model} produced {len(thinking)} characters of reasoning but no answer: "
                    f"it ran out of output budget while thinking. Raise num_predict "
                    f"(currently {num_predict}) or keep think=False."
                )
            raise BackendError(
                f"Ollama returned an empty message "
                f"(done_reason: {data.get('done_reason') or 'unknown'})."
            )
        return text

    return ollama


def unload(model: str, host: str = DEFAULT_HOST) -> None:
    """Ask Ollama to drop a model from VRAM now rather than after keep_alive."""
    body = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
    req = urllib.request.Request(_host_url(host).rstrip("/") + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=30).close()
    except (urllib.error.URLError, TimeoutError):
        pass


_FENCE_HEAD = re.compile(r"^[\s\S]*?```(?:json)?\s*", re.IGNORECASE)
_FENCE_TAIL = re.compile(r"```[\s\S]*\Z")


def parse_json(text: str, what: str = "response"):
    """Models wrap JSON in prose or code fences no matter how firmly you ask
    them not to. Recover the object rather than failing the whole chunk."""
    attempts = [text, _FENCE_TAIL.sub("", _FENCE_HEAD.sub("", text, count=1), count=1)]
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        attempts.append(text[first:last + 1])

    for candidate in attempts:
        try:
            return json.loads(candidate.strip())
        except ValueError:
            continue

    # Show the TAIL, not the head. These failures are nearly always a reply that
    # ran out of output budget, and it is the end that reveals the cut.
    raise ValueError(
        f"Could not parse JSON from the {what} ({len(text)} chars). "
        f"Last 200 characters:\n…{text[-200:]}"
    )
