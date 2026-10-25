from __future__ import annotations

from pathlib import Path

from loguru import logger
import pandas as pd

from dencad.data.split.split import Splitter


VALID_START = "2009-09-13"
TEST_START = "2011-11-02"
DIRTY = ("is_anomaly", "is_rare_event")


class EsaSplitter(Splitter):
    def __init__(
        self,
        source: Path = Path("data/interim/esa/full_esa.parquet"),
        out_dir: Path = Path("data/interim/esa/split"),
        valid_start: str = VALID_START,
        test_start: str = TEST_START,
        name: str = "esa",
    ) -> None:
        super().__init__(source, out_dir, valid_start, test_start, name)

    def cut(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        parts = super().cut(df)
        train = parts["train"]
        for c in [c for c in DIRTY if c in train.columns]:
            k = int(train[c].to_numpy().sum())
            logger.info("the training part carries {:,} rows flagged '{}' ({:.2%}); ", k, c, k / len(train))
        return parts



def main() -> None:
    EsaSplitter().run(overwrite=True)


if __name__ == "__main__":
    main()


__all__ = ["EsaSplitter", "VALID_START", "TEST_START"]
