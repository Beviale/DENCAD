"""SWaT, Secure Water Treatment: one csv into one table, de-duplicated and ordered."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger
import numpy as np
import pandas as pd

from dencad.data.preprocess.preprocess import Preprocessor

TIME, LABEL = "Timestamp", "Normal/Attack"
ATTACK = "Attack"
TIME_FORMAT = "%d/%m/%Y %I:%M:%S %p"


class SwatPreprocessor(Preprocessor):
    """The merged SWaT csv, de-duplicated, time-ordered and labelled."""

    def __init__(
        self,
        source: Path = Path("data/raw/swat/swat.csv"),
        out_dir: Path = Path("data/interim/swat"),
        name: str = "full_swat",
    ) -> None:
        super().__init__(source, out_dir, name, start=None, end=None)
        self._clean: Optional[pd.DataFrame] = None
        self._instruments: list[str] = []

    # ------------------------------------------------------------------ repair

    @property
    def clean(self) -> pd.DataFrame:
        """The csv with its defects undone, indexed by timestamp."""
        if self._clean is None:
            self._clean = self._repair()
        return self._clean

    def _repair(self) -> pd.DataFrame:
        raw = pd.read_csv(self.source, low_memory=False)
        logger.info("read {:,} rows x {} columns from {}",
                    len(raw), raw.shape[1], self.source)

        raw.columns = [str(c).strip() for c in raw.columns]
        raw[LABEL] = (raw[LABEL].astype(str).str.strip()
                      .str.replace(" ", "", regex=False))
        raw[TIME] = raw[TIME].astype(str).str.strip()
        labels_seen = sorted(raw[LABEL].unique())
        if set(labels_seen) - {"Normal", ATTACK}:
            raise ValueError(f"unexpected labels after normalising: {labels_seen}")

        ts = pd.to_datetime(raw[TIME], format=TIME_FORMAT)
        raw = (raw.drop(columns=[TIME])
                  .set_axis(pd.DatetimeIndex(ts.to_numpy(), name="datetime"))
                  .sort_index(kind="stable"))

        raw = self.drop_duplicate_and_null_rows(raw)

        instruments = [c for c in raw.columns if c != LABEL]
        raw[instruments] = (raw[instruments].apply(pd.to_numeric, errors="coerce")
                            .astype("float32"))
        self._instruments = instruments

        logger.info("{:,} rows from {} to {}, {} instruments",
                    len(raw), raw.index[0], raw.index[-1], len(self._instruments))
        return self._regularise(raw)

    def _regularise(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Report the discontinuities and keep only the instants that were logged."""

        gaps = np.diff(raw.index.to_numpy()).astype("timedelta64[s]").astype("int64")
        broken = np.flatnonzero(gaps != 1)
        if len(broken) == 0:
            logger.info("the series is numeric: every second present")
        else:
            logger.warning("{} discontinuities, {:,} seconds missing from the source",
                           len(broken), int(gaps[broken].sum() - len(broken)))
            for k in broken[:5]:
                logger.warning("  {} -> {}  ({:,} s), left as a gap rather than "
                               "filled", raw.index[k], raw.index[k + 1], int(gaps[k]))
                
        logger.info("{:,} rows kept, all of them instants that were logged", len(raw))
        return raw

    # ------------------------------------------------------------------- hooks

    def load_values(self) -> pd.DataFrame:
        values = self.clean[self._instruments]
        holes = int(values.isna().to_numpy().sum())
        logger.info("{} instruments, {:,} missing cells", values.shape[1], holes)
        return values

    def load_labels(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        is_attack = (self.clean[LABEL].loc[index].to_numpy() == ATTACK).astype("uint8")
        labels = pd.DataFrame({"is_attack": is_attack}, index=index)

        a = labels["is_attack"].to_numpy()
        logger.info("attacks: {:,} of {:,} rows ({:.2%}) in {} segments",
                    int(a.sum()), len(a), a.mean(), self._segments(a))
        # Where the attacks begin-
        hit = np.flatnonzero(a)
        logger.info("first attack {}, last attack {}, nothing before that",
                    index[hit[0]], index[hit[-1]])
        return labels

    @staticmethod
    def _segments(y: np.ndarray) -> int:
        """Contiguous runs of positives, which is what an attack actually is."""
        return int((np.diff(np.r_[0, (y != 0).astype(int), 0]) == 1).sum())


def main() -> None:
    SwatPreprocessor().run(overwrite=True)


if __name__ == "__main__":
    main()


__all__ = ["SwatPreprocessor", "TIME", "LABEL", "ATTACK"]
