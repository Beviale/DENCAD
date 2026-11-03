"""One configuration of a search, in its own interpreter."""

from __future__ import annotations

import json
import sys

from pathlib import Path

from dencad.modeling.search import MODELS

WINDOW = "_window_steps"


def main() -> None:
    model, split_dir, reports_dir, config_file, out_file = sys.argv[1:6]
    config = json.loads(Path(config_file).read_text(encoding="utf-8"))
    out = Path(out_file)
    try:
        report = MODELS[model](Path(split_dir), Path(reports_dir), stage="select",
                               window_steps=config.pop(WINDOW, None), **config)
    except Exception as exc:
        out.write_text(json.dumps({"failed": f"{type(exc).__name__}: {exc}"}),
                       encoding="utf-8")
        raise
    out.write_text(json.dumps(report, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
