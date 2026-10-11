from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger
import pandas as pd

from dencad.data.process.process import Processor
from dencad.utils import declared_categorical

ANOMALY, RARE = "is_anomaly", "is_rare_event"
MODES = {"esa_rare_anomalies": True}


def esa_categorical(meta_path: Path) -> list[str]:
    meta = pd.read_csv(meta_path)
    return meta.loc[meta["Categorical"] == "YES", "Channel"].tolist()


class EsaProcessor(Processor):
    def __init__(
        self,
        split_dir: Path = Path("data/interim/esa/split"),
        out_dir: Path = Path("data/processed/esa_rare_anomalies"),
        name: str = "esa",
        categorical: Optional[list[str]] = None,
        pearson_max: float = 0.99,
        cramer_max: float = 0.99,
        rare_counts_as_positive: bool = True,
        meta_path: Path = Path("data/raw/esa/ESA-Mission2/channels.csv"),
    ) -> None:
        if categorical is None:
            categorical = declared_categorical(name)
            if meta_path.exists():
                stated = esa_categorical(meta_path)
                if sorted(stated) != sorted(categorical):
                    raise ValueError(
                        f"{meta_path} marks {len(stated)} channels Categorical but "
                        f"the declaration file names {len(categorical)}: "
                        f"{sorted(set(stated) ^ set(categorical))}")
        super().__init__(split_dir, out_dir, name,
                         categorical=categorical, pearson_max=pearson_max,
                         cramer_max=cramer_max)
        self.rare_counts_as_positive = rare_counts_as_positive

    def prepare_train(self, train: pd.DataFrame) -> pd.DataFrame:
        """Drop the training rows that are positive."""
        pos = train[ANOMALY].to_numpy()
        if self.rare_counts_as_positive:
            pos = pos | train[RARE].to_numpy()
        drop = pos != 0
        kept_rare = int(train.loc[~drop, RARE].to_numpy().sum())
        logger.info("training rows dropped as positive under this label: {:,} of "
                    "{:,} ({:.2%})", int(drop.sum()), len(train), drop.mean())
        if not self.rare_counts_as_positive:
            logger.info("  {:,} rare-event rows are kept: not positives here",
                        kept_rare)
        return train.loc[~drop]

    def transform(self, df: pd.DataFrame, part: str) -> pd.DataFrame:
        out = super().transform(df, part)
        anomaly = out[ANOMALY].to_numpy()
        rare = out[RARE].to_numpy()
        label = (anomaly | rare) if self.rare_counts_as_positive else anomaly
       
        drop = [c for c in out.columns
                if c in (ANOMALY, RARE) or c.startswith("label_")]
        out = out.drop(columns=drop)
        out[ANOMALY] = label.astype("uint8")
        logger.info("{:<11} label = {:<28} {:>8,} positive ({:.2%})", part,
                    "anomaly or rare event" if self.rare_counts_as_positive
                    else "anomaly only", int(label.sum()), label.mean())
        return out


def main() -> None:
    for folder, rare in MODES.items():
        logger.info("=== {} ===", folder)
        EsaProcessor(out_dir=Path("data/processed") / folder,
                     rare_counts_as_positive=rare).run(overwrite=True)


if __name__ == "__main__":
    main()


__all__ = ["EsaProcessor", "esa_categorical", "MODES"]
