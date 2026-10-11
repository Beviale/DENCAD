from __future__ import annotations

from pathlib import Path

from dencad.data.process.process import Processor
from dencad.utils import declared_categorical


class HaiProcessor(Processor):
    def __init__(
        self,
        split_dir: Path = Path("data/interim/hai/split"),
        out_dir: Path = Path("data/processed/hai"),
        name: str = "hai",
        categorical: list[str] | None = None,
        pearson_max: float = 0.99,
        cramer_max: float = 0.99,
    ) -> None:
        if categorical is None:
            categorical = declared_categorical(name)
        super().__init__(split_dir, out_dir, name,
                         categorical=categorical, pearson_max=pearson_max,
                         cramer_max=cramer_max)


def main() -> None:
    HaiProcessor().run(overwrite=True)


if __name__ == "__main__":
    main()


__all__ = ["HaiProcessor"]
