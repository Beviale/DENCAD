from __future__ import annotations

from pathlib import Path
from typing import Optional

from dencad.data.split.split import Splitter


class SwatSplitter(Splitter):
    """'full_swat.parquet' cut on the two dates above."""

    def __init__(
        self,
        source: Path = Path("data/interim/swat/full_swat.parquet"),
        out_dir: Path = Path("data/interim/swat/split"),
        valid_start: str = "2015-12-28 10:00:00",
        test_start: str = "2015-12-31 00:29:19",
        name: Optional[str] = "swat",
    ) -> None:
        super().__init__(source, out_dir, valid_start, test_start, name)


def main() -> None:
    SwatSplitter().run(overwrite=True)


if __name__ == "__main__":
    main()


__all__ = ["SwatSplitter"]
