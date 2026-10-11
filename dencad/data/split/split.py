"""One interim table into train, validation and test, cut on two dates."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger
import numpy as np
import pandas as pd

PARTS = ("train", "validation", "test")


class Splitter:
    """Cut a time-ordered table into three parts on two timestamps.

    'valid_start' opens the validation part and 'test_start' opens the test part, so
    the training part is everything before the first of them.
    """

    def __init__(
        self,
        source: Path,
        out_dir: Path,
        valid_start: str,
        test_start: str,
        name: Optional[str] = None,
    ) -> None:
        self.source = Path(source)
        self.out_dir = Path(out_dir)
        self.valid_start = pd.Timestamp(valid_start)
        self.test_start = pd.Timestamp(test_start)
        self.name = name or self.source.stem
        if self.valid_start >= self.test_start:
            raise ValueError(
                f"validation starts at {self.valid_start} and test at "
                f"{self.test_start}."
            )

    # ------------------------------------------------------------- machinery

    def path_for(self, part: str) -> Path:
        return self.out_dir / f"{self.name}_{part}.parquet"

    def load(self) -> pd.DataFrame:
        """The table to cut."""
        return pd.read_parquet(self.source)

    def check_order(self, df: pd.DataFrame) -> None:
        """Refuse to cut anything that is not a sorted time index."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                f"the index is {type(df.index).__name__}, not a DatetimeIndex: a "
                "date cannot select rows from it"
            )
        if not df.index.is_monotonic_increasing:
            back = int((df.index.to_series().diff() < pd.Timedelta(0)).sum())
            raise ValueError(
                f"the index goes backwards {back:,} times; a date-based split of an "
                "unordered frame interleaves periods inside every part"
            )
        if not df.index.is_unique:
            raise ValueError(
                f"{int(df.index.duplicated().sum()):,} rows share a timestamp with "
                "another: one instant cannot belong to two parts"
            )

    def cut(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """The three parts, half-open and in order."""
        i = df.index
        parts = {
            "train": df.loc[i < self.valid_start],
            "validation": df.loc[(i >= self.valid_start) & (i < self.test_start)],
            "test": df.loc[i >= self.test_start],
        }
        empty = [k for k, v in parts.items() if v.empty]
        if empty:
            raise ValueError(
                f"{', '.join(empty)} would be empty; the table spans {i[0]} to "
                f"{i[-1]} and the cuts are {self.valid_start} and {self.test_start}"
            )
        total = sum(len(v) for v in parts.values())
        if total != len(df):
            raise ValueError(f"the parts hold {total:,} rows of {len(df):,}")
        return parts

    @staticmethod
    def segments(y: np.ndarray) -> int:
        """Contiguous runs of non-zero values, which is what an event actually is."""
        return int((np.diff(np.r_[0, (y != 0).astype(int), 0]) == 1).sum())

    def check_segments(self, df: pd.DataFrame, parts: dict[str, pd.DataFrame]) -> None:
        """Refuse a cut that falls inside a labelled event."""
        for col in [c for c in df.columns if c.startswith("is_") or c.startswith("label_")]:
            whole = self.segments(df[col].to_numpy())
            pieces = sum(self.segments(parts[p][col].to_numpy()) for p in PARTS)
            if pieces == whole:
                continue
            y, i = df[col].to_numpy(), df.index
            runs = np.flatnonzero(np.diff(np.r_[0, (y != 0).astype(int), 0]) == 1)
            ends = np.flatnonzero(np.diff(np.r_[0, (y != 0).astype(int), 0]) == -1)
            culprits = [(i[a], i[b - 1]) for a, b in zip(runs, ends)
                        if any(a < i.searchsorted(c) <= b
                               for c in (self.valid_start, self.test_start))]
            detail = "; ".join(f"{a} to {b}" for a, b in culprits[:3])
            raise ValueError(
                f"a cut falls inside a run of '{col}': the parts hold {pieces} runs "
                f"where the table holds {whole}. The run(s) cut: {detail}. Move the "
                "boundary outside them, or override check_segments to allow it."
            )
        logger.info("no cut falls inside a labelled run")

    def run(self, overwrite: bool = False) -> dict[str, Path]:
        written = {}
        existing = [p for p in PARTS if self.path_for(p).exists()]
        if existing and not overwrite:
            logger.info("{} already exist, nothing to do",
                        ", ".join(self.path_for(p).name for p in existing))
            return {p: self.path_for(p) for p in PARTS}

        df = self.load()
        self.check_order(df)
        logger.info("{}: {:,} rows x {} columns, {} to {}", self.source.name,
                    len(df), df.shape[1], df.index[0], df.index[-1])
        parts = self.cut(df)
        self.check_segments(df, parts)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        for part in PARTS:
            tmp = self.path_for(part).with_suffix(".parquet.partial")
            parts[part].to_parquet(tmp, compression="snappy")
            tmp.replace(self.path_for(part))
            written[part] = self.path_for(part)
        self.describe(parts)
        return written

    def describe(self, parts: dict[str, pd.DataFrame]) -> None:
        """What each part holds, in the terms that decide whether it is usable."""
        total = sum(len(v) for v in parts.values())
        flags = [c for c in next(iter(parts.values())).columns if c.startswith("is_")]
        logger.success(
            "{} = {} of {:,} rows, cut at {} and {}",
            " / ".join(PARTS),
            " / ".join(f"{len(parts[p]) / total:.1%}" for p in PARTS),
            total, self.valid_start, self.test_start)
        for part in PARTS:
            p = parts[part]
            logger.success("{:<11} {:>9,} rows ({:>5.1%})  {} -> {}",
                           part, len(p), len(p) / total, p.index[0], p.index[-1])
            for col in flags:
                k = int(p[col].sum())
                logger.success("            {:<14} {:>8,} ({:.2%})", col, k,
                               k / len(p))
        for part in PARTS:
            logger.success("  {}", self.path_for(part))


__all__ = ["Splitter", "PARTS"]
