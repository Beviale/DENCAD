"""Turning a raw dataset into one interim table with its labels attached."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq


class Preprocessor(ABC):
    """Base for the dataset-specific preprocessors."""

    def __init__(
        self,
        source: Path,
        out_dir: Path,
        name: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> None:
        self.source = Path(source)
        self.out_dir = Path(out_dir)
        self.name = name
        # Half-open [start, end): the end is excluded.
        self.start = pd.Timestamp(start) if start is not None else None
        self.end = pd.Timestamp(end) if end is not None else None

    # ------------------------------------------------------------------ hooks

    @abstractmethod
    def load_values(self) -> pd.DataFrame:
        """Measurements on a DatetimeIndex, one column per variable."""

    @abstractmethod
    def load_labels(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Label columns on exactly 'index'. Must not reindex or reorder it."""

    def iter_blocks(self) -> Iterator[pd.DataFrame]:
        """Yield the output in row blocks, values and labels already joined."""
        values = self.window(self.load_values())
        if values.empty:
            raise ValueError(f"no rows left in [{self.start}, {self.end})")
        values = self.drop_duplicate_and_null_rows(values)
        labels = self.load_labels(values.index)
        self.check_alignment(values, labels)
        yield pd.concat([values, labels], axis=1)

    def drop_duplicate_and_null_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows that repeat an earlier one, and rows carrying no reading at all."""
        before = len(df)

        blank = df.isna().all(axis=1).to_numpy()
        if blank.any():
            logger.warning("dropping {:,} rows ({:.3%}) with no reading in any of "
                           "the {} feature columns", int(blank.sum()), blank.mean(),
                           df.shape[1])
            df = df.loc[~blank]
        else:
            logger.info("every row carries a reading in at least one of the {} "
                        "feature columns", df.shape[1])

        if df.index.is_unique:
            logger.info("the index is unique: no row can repeat another, "
                        "timestamp included")
        else:
            dup = df.reset_index().duplicated(keep="first").to_numpy()
            if dup.any():
                logger.warning("dropping {:,} rows identical to an earlier one "
                               "({:.3%}), keeping the first of each",
                               int(dup.sum()), dup.mean())
                df = df.loc[~dup]
            else:
                logger.info("no row repeats an earlier one, timestamp included")

        if len(df) != before:
            logger.info("{:,} rows kept of {:,}", len(df), before)
        return df

    # ------------------------------------------------------------- machinery

    @property
    def output_path(self) -> Path:
        return self.out_dir / f"{self.name}.parquet"

    def window(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restrict to [start, end), when either bound is set."""
        if self.start is not None:
            df = df.loc[df.index >= self.start]
        if self.end is not None:
            df = df.loc[df.index < self.end]
        return df

    @staticmethod
    def check_alignment(values: pd.DataFrame, labels: pd.DataFrame) -> None:
        """Refuse to write unless the two frames describe the same instants.

        Equal length is not enough -- two frames of the same height can still be
        offset by a row -- so the index itself is compared, element by element.
        """
        if len(values) != len(labels):
            raise ValueError(
                f"values have {len(values):,} rows and labels {len(labels):,}: "
                "they cannot describe the same instants"
            )
        if not values.index.equals(labels.index):
            first = int((values.index != labels.index).argmax())
            raise ValueError(
                "values and labels share a length but not an index; they first "
                f"differ at position {first}: {values.index[first]} against "
                f"{labels.index[first]}"
            )

    def run(self, overwrite: bool = False) -> Path:
        """Stream the blocks into one parquet file. Returns the path written."""

        if self.output_path.exists() and not overwrite:
            logger.info("{} already exists, nothing to do", self.output_path)
            return self.output_path

        self.out_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.output_path.with_suffix(".parquet.partial")
        writer = None
        rows = 0
        first: Optional[pd.DataFrame] = None
        try:
            for block in self.iter_blocks():
                table = pa.Table.from_pandas(block, preserve_index=True)
                if writer is None:
                    writer = pq.ParquetWriter(tmp, table.schema, compression="snappy")
                    first = block.iloc[:0]
                writer.write_table(table)
                rows += len(block)
                logger.info("  written {:,} rows", rows)
        finally:
            if writer is not None:
                writer.close()
        if writer is None or rows == 0:
            raise ValueError("no rows produced; nothing written")

        tmp.replace(self.output_path)
        assert first is not None
        self.describe(first, rows)
        return self.output_path

    def describe(self, schema_frame: pd.DataFrame, rows: int) -> None:
        """What was written, in the terms that matter when reading it back."""
        label_cols = [c for c in schema_frame.columns
                      if c.startswith("label_") or c.startswith("is_")]
        logger.success("{}: {:,} rows x {} columns ({} values + {} labels)",
                       self.name, rows, schema_frame.shape[1],
                       schema_frame.shape[1] - len(label_cols), len(label_cols))
        logger.success("  {}  ({:.2f} GB on disk)", self.output_path,
                       self.output_path.stat().st_size / 2**30)


__all__ = ["Preprocessor"]
