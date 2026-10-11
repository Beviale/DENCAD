"""HAI 23.05: three recordings and one label file into one table."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger
import numpy as np
import pandas as pd

from dencad.data.preprocess.preprocess import Preprocessor

TIME = "timestamp"
LABEL = "label"
FILES = ("hai-train1", "hai-train2", "hai-test2")
LABEL_FOR = {"hai-test2": "label-test2"}


class HaiPreprocessor(Preprocessor):
    """The four requested HAI 23.05 runs, time-ordered, with one label column."""

    def __init__(
        self,
        source: Path = Path("data/raw/hai-23.05"),
        out_dir: Path = Path("data/interim/hai"),
        name: str = "full_hai",
        files: tuple[str, ...] = FILES,
    ) -> None:
        super().__init__(source, out_dir, name, start=None, end=None)
        self.files = tuple(files)
        self._labels: Optional[pd.Series] = None

    def _read(self, stem: str) -> pd.DataFrame:
        df = pd.read_csv(self.source / f"{stem}.csv", low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        ts = pd.DatetimeIndex(pd.to_datetime(df.pop(TIME)), name="datetime")
        return df.apply(pd.to_numeric, errors="coerce").astype("float32").set_axis(ts)

    def _label_for(self, stem: str, n: int, index: pd.DatetimeIndex) -> pd.Series:
        if stem not in LABEL_FOR:
            logger.info("  {:<11} no label file: attack-free run, label set to 0", stem)
            return pd.Series(np.zeros(n, dtype="uint8"), index=index)
        lf = pd.read_csv(self.source / f"{LABEL_FOR[stem]}.csv")
        if len(lf) != n:
            raise ValueError(
                f"{LABEL_FOR[stem]} has {len(lf):,} rows and {stem} has {n:,}: "
                "they cannot be aligned by position"
            )
        lt = pd.to_datetime(lf[TIME])
        if not lt.is_unique:
            logger.warning("  {:<11} its label file repeats {:,} timestamps and "
                           "cannot be joined on them; aligned by position",
                           stem, int(lt.duplicated().sum()))
        y = (lf[LABEL].to_numpy() != 0).astype("uint8")
        logger.info("  {:<11} {:,} labelled rows, {:,} attack ({:.2%})",
                    stem, n, int(y.sum()), y.mean())
        return pd.Series(y, index=index)

    def load_values(self) -> pd.DataFrame:
        frames, labels, spans = [], [], {}
        reference: Optional[list[str]] = None
        for stem in self.files:
            df = self._read(stem)
            if reference is None:
                reference = list(df.columns)
            elif list(df.columns) != reference:
                raise ValueError(
                    f"{stem} does not carry the same columns, in the same order, as "
                    f"{self.files[0]}"
                )
            spans[stem] = (df.index[0], df.index[-1])
            labels.append(self._label_for(stem, len(df), df.index))
            frames.append(df)

        order = sorted(spans, key=lambda k: spans[k][0])
        for a, b in zip(order, order[1:]):
            if spans[a][1] >= spans[b][0]:
                raise ValueError(f"{a} and {b} overlap in time")
        logger.info("the {} runs are disjoint, in time order: {}",
                    len(order), " -> ".join(order))

        values = pd.concat(frames).sort_index(kind="stable")
        self._labels = pd.concat(labels).sort_index(kind="stable")
       

        step = np.diff(values.index.to_numpy()).astype("timedelta64[s]").astype("int64")
        logger.info("{:,} rows x {} tags, {} to {}", len(values), values.shape[1],
                    values.index[0], values.index[-1])
        broken = np.flatnonzero(step != 1)
        logger.info("  {} breaks in the 1 Hz grid across {} run boundaries",
                    len(broken), len(order) - 1)
        for k in broken:
            logger.info("    {} -> {}  ({})", values.index[k], values.index[k + 1],
                        pd.to_timedelta(int(step[k]), unit="s"))
        holes = int(values.isna().to_numpy().sum())
        if holes:
            logger.warning("  {:,} missing cells", holes)
        else:
            logger.info("  no missing cell")
        return values

    def load_labels(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        assert self._labels is not None
        y = self._labels.loc[index].to_numpy()
        runs = int((np.diff(np.r_[0, (y != 0).astype(int), 0]) == 1).sum())
        logger.info("attacks: {:,} of {:,} rows ({:.2%}) in {} segments",
                    int(y.sum()), len(y), y.mean(), runs)
        return pd.DataFrame({"is_attack": y}, index=index)


def main() -> None:
    HaiPreprocessor().run(overwrite=True)


if __name__ == "__main__":
    main()


__all__ = ["HaiPreprocessor", "FILES", "LABEL_FOR"]
