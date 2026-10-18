from __future__ import annotations

from pathlib import Path

from loguru import logger
import pandas as pd

from dencad.data.split.split import Splitter

LEAD_IN_END = "2000-02-28 06:22:30"
VALID_START = "2001-10-01"
TEST_START = "2002-04-03"
DIRTY = ("is_anomaly", "is_rare_event")


class EsaSplitter(Splitter):
    def __init__(
        self,
        source: Path = Path("data/interim/esa/full_esa.parquet"),
        out_dir: Path = Path("data/interim/esa/split"),
        valid_start: str = VALID_START,
        test_start: str = TEST_START,
        name: str = "esa",
        lead_in_end: str = LEAD_IN_END,
    ) -> None:
        super().__init__(source, out_dir, valid_start, test_start, name)
        self.lead_in_end = pd.Timestamp(lead_in_end)

    def load(self) -> pd.DataFrame:
        df = super().load()
        keep = df.index >= self.lead_in_end
        dropped = int((~keep).sum())
        if dropped:
            logger.info("dropping the lead-in: {:,} rows before {} ({:.2%} of the "
                        "table), where 99 of 100 channels have yet to report",
                        dropped, self.lead_in_end, dropped / len(df))
        return df.loc[keep]

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


__all__ = ["EsaSplitter", "LEAD_IN_END", "VALID_START", "TEST_START"]
