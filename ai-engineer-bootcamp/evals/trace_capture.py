"""Captura de trazas: graba la ejecución de un sistema bajo prueba a JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def capture_trace(
    trace_id: str,
    query: str,
    system_fn: Callable[[str], dict[str, Any]],
    output_dir: str = "evals/traces",
) -> str:
    """Ejecuta `system_fn(query)` y guarda la salida como traza JSON.

    Args:
        trace_id: Identificador único de la traza (usado como nombre de archivo).
        query: Entrada al sistema bajo prueba.
        system_fn: Función a invocar (debe aceptar str y devolver dict).
        output_dir: Carpeta donde se escriben las trazas.

    Returns:
        Ruta absoluta del archivo de traza generado.
    """
    output = system_fn(query)

    trace = {
        "trace_id": trace_id,
        "query": query,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "output": output,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"{trace_id}.json"
    file_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(file_path.resolve())
