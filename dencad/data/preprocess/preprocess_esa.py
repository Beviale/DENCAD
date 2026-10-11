"""ESA Anomaly Dataset, Mission 2: 100 independent channels into one table."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Iterator, Optional, Sequence

from loguru import logger
import numpy as np
import pandas as pd

from dencad.data.preprocess.preprocess import Preprocessor

NOMINAL, ANOMALY, RARE_EVENT = 0, 1, 2


class EsaPreprocessor(Preprocessor):
    """Mission 2 of the ESA Anomaly Dataset"""

    def __init__(
        self,
        source: Path = Path("data/raw/esa/ESA-Mission2"),
        out_dir: Path = Path("data/interim/esa"),
        name: str = "full_esa",
        rule: str = "18s",
        channels: Optional[Sequence[str]] = None,
        target_only: bool = False,
        block_rows: int = 400_000,
    ) -> None:
        super().__init__(source, out_dir, name, start=None, end=None)
        # The interval of the common grid every channel is resampled onto, as a pandas
        # frequency string.
        self.rule = rule
        # If True, load only the 47 channels 'channels.csv' marks Target=YES, instead of all 100.
        self.target_only = target_only 
        # Before a channel first reports there is nothing to hold forward, so the
        # grid starts with NaN and keeps it. 
        self._lead: dict[str, int] = {}
        # Rows per parquet row group.
        self.block_rows = block_rows
        self._channels = list(channels) if channels is not None else None
        self._grid: Optional[pd.DatetimeIndex] = None
        self._annotations: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------- selection

    @property
    def channels(self) -> list[str]:
        """Channel names to load, in the order the index file lists them."""
        if self._channels is None:
            meta = pd.read_csv(self.source / "channels.csv")
            if self.target_only:
                meta = meta.loc[meta["Target"] == "YES"]
            available = {p.stem for p in (self.source / "channels").glob("*.zip")}
            self._channels = [c for c in meta["Channel"] if c in available]
            missing = [c for c in meta["Channel"] if c not in available]
            if missing:
                logger.warning("{} channels listed but not on disk: {}",
                               len(missing), ", ".join(missing[:5]))
        return self._channels

    @property
    def grid(self) -> pd.DatetimeIndex:
        """The common time axis every channel is resampled onto."""
        if self._grid is None:
            first, last = self._extent()
            self._grid = pd.date_range(first.floor(self.rule), last,
                                       freq=self.rule, name="datetime")
        return self._grid

    def _extent(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Earliest and latest reading across every channel loaded."""
        first = last = None
        for i, name in enumerate(self.channels, 1):
            df = pd.read_pickle(self.source / "channels" / f"{name}.zip")
            lo, hi = df.index[0], df.index[-1]
            first = lo if first is None or lo < first else first
            last = hi if last is None or hi > last else last
            del df
            if i % 25 == 0 or i == len(self.channels):
                logger.info("  scanned {}/{} channels for their extent",
                            i, len(self.channels))
        logger.info("the archive runs from {} to {}", first, last)
        return first, last

    # ------------------------------------------------------------ resampling

    def _resample(self, name: str, grid: pd.DatetimeIndex) -> np.ndarray:
        df = pd.read_pickle(self.source / "channels" / f"{name}.zip")
        s = df[name]
        if s.dtype == object:
            # Ten channels are flagged Categorical; factorising keeps them in the
            # table as integer codes.
            s = pd.Series(pd.factorize(s)[0], index=s.index, name=name)
        pos = s.index.searchsorted(grid, side="right") - 1
        out = np.full(len(grid), np.nan, dtype=np.float32)
        seen = pos >= 0                      
        vals = s.to_numpy(dtype=np.float32, copy=False)
        out[seen] = vals[pos[seen]]
        self._lead[name] = int((~seen).sum())
        return out

    def _cache_channels(self, cache: Path) -> None:
        """First pass: every channel resampled onto the grid, parked on disk."""
        grid = self.grid
        logger.info("grid: {:,} rows at {} from {} to {}",
                    len(grid), self.rule, grid[0], grid[-1])
        for i, name in enumerate(self.channels, 1):
            np.save(cache / f"{name}.npy", self._resample(name, grid))
            if i % 20 == 0 or i == len(self.channels):
                logger.info("  resampled {}/{} channels", i, len(self.channels))
        late = {k: v for k, v in self._lead.items() if v}
        if late:
            step = pd.Timedelta(self.rule).total_seconds()
            total = sum(late.values())
            logger.info("{} channels report for the first time after the window "
                        "opens; those rows stay NaN ({:,} in all)", len(late), total)
            for name, n in sorted(late.items(), key=lambda kv: -kv[1])[:5]:
                logger.info("    {:<14} {:>9,} NaN rows ({:.1f} days)",
                            name, n, n * step / 86400)

    # ----------------------------------------------------------------- values

    def load_values(self) -> pd.DataFrame:
        grid = self.grid
        mat = np.empty((len(grid), len(self.channels)), dtype=np.float32)
        for i, name in enumerate(self.channels):
            mat[:, i] = self._resample(name, grid)
        full_df = pd.DataFrame(mat, index=grid, columns=self.channels, copy=False)
        holes = int(full_df.isna().to_numpy().sum())
        logger.info("{} instruments, {:,} missing cells", full_df.shape[1], holes)
        return full_df

    # ----------------------------------------------------------------- labels

    @property
    def annotations(self) -> pd.DataFrame:
        """Annotation intervals that touch the window, with their category."""
        if self._annotations is None:
            lab = pd.read_csv(self.source / "labels.csv",
                              parse_dates=["StartTime", "EndTime"])
            kinds = pd.read_csv(self.source / "anomaly_types.csv")[["ID", "Category"]]
            ann = lab.merge(kinds, on="ID", how="left")
            for col in ("StartTime", "EndTime"):
                if isinstance(ann[col].dtype, pd.DatetimeTZDtype):
                    ann[col] = ann[col].dt.tz_convert(None)
            ann = ann[ann["Channel"].isin(set(self.channels))]
            ann = ann[(ann["EndTime"] >= self.grid[0]) & (ann["StartTime"] <= self.grid[-1])]
            logger.info("{} annotation intervals touch the window "
                        "({} anomalies, {} rare events)", len(ann),
                        int((ann["Category"] == "Anomaly").sum()),
                        int((ann["Category"] == "Rare Event").sum()))
            self._annotations = ann
        return self._annotations

    def load_labels(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        cols = {c: np.zeros(len(index), dtype=np.uint8) for c in self.channels}
        for category, value in (("Rare Event", RARE_EVENT), ("Anomaly", ANOMALY)):
            sub = self.annotations[self.annotations["Category"] == category]
            for channel, s, e in zip(sub["Channel"], sub["StartTime"], sub["EndTime"]):
                lo = index.searchsorted(s, side="left")
                hi = index.searchsorted(e, side="right")
                if channel in cols and hi > lo:
                    cols[channel][lo:hi] = value

        labels = pd.DataFrame({f"label_{c}": v for c, v in cols.items()}, index=index)
        stacked = np.stack(list(cols.values()))
        labels["is_anomaly"] = (stacked == ANOMALY).any(axis=0).astype(np.uint8)
        labels["is_rare_event"] = (stacked == RARE_EVENT).any(axis=0).astype(np.uint8)
        return labels

    # ----------------------------------------------------------------- blocks

    def iter_blocks(self) -> Iterator[pd.DataFrame]:
        """Two passes: cache each channel on disk, then assemble row blocks."""
        grid = self.grid
        cache = Path(tempfile.mkdtemp(prefix="esa_resampled_"))
        logger.info("caching resampled channels under {}", cache)
        try:
            self._cache_channels(cache)
            if not grid.is_unique:
                raise ValueError("the grid is not unique; rows could duplicate")
            logger.info("the grid is unique")
            maps = {c: np.load(cache / f"{c}.npy", mmap_mode="r") for c in self.channels}
            n_anom = n_rare = 0
            for lo in range(0, len(grid), self.block_rows):
                hi = min(lo + self.block_rows, len(grid))
                idx = grid[lo:hi]
                values = pd.DataFrame(
                    {c: np.asarray(m[lo:hi]) for c, m in maps.items()}, index=idx)
                holes = int(values.isna().to_numpy().sum())
                logger.info("{} instruments, {:,} missing cells", values.shape[1], holes)
                labels = self.load_labels(idx)
                self.check_alignment(values, labels)
                n_anom += int(labels["is_anomaly"].sum())
                n_rare += int(labels["is_rare_event"].sum())
                yield pd.concat([values, labels], axis=1)
            logger.info("rows flagged: {:,} anomaly ({:.4%}), {:,} rare event ({:.4%})",
                        n_anom, n_anom / len(grid), n_rare, n_rare / len(grid))
        finally:
            shutil.rmtree(cache, ignore_errors=True)


def main() -> None:
    EsaPreprocessor().run(overwrite=True)


if __name__ == "__main__":
    main()


__all__ = ["EsaPreprocessor", "NOMINAL", "ANOMALY", "RARE_EVENT"]
