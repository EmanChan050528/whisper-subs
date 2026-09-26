"""Translation units -> request chunks. Port of jp-subs' core/chunk.js.

Each chunk carries units the model must translate, plus surrounding units it
may only read. The `after` window is the whole point of working on a finished
recording: in Japanese the thing that disambiguates a dropped subject is very
often what comes next.
"""

DEFAULTS = {
    "size": 20,            # units translated per request
    "context_before": 10,  # units of preceding context, read-only
    "context_after": 6,    # units of following context, read-only
}


def chunk(units: list[dict], options: dict | None = None) -> list[dict]:
    opts = {**DEFAULTS, **(options or {})}
    size, before, after = opts["size"], opts["context_before"], opts["context_after"]
    chunks = []
    for start in range(0, len(units), size):
        end = min(start + size, len(units))
        chunks.append({
            "index": len(chunks),
            "first_unit": start,
            "before": units[max(0, start - before):start],
            "target": units[start:end],
            "after": units[end:min(end + after, len(units))],
        })
    return chunks
