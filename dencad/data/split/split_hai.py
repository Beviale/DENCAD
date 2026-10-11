from __future__ import annotations

from pathlib import Path

from dencad.data.split.split import Splitter

VALID_START = "2022-08-17 00:00:01"
TEST_START = "2022-08-18 08:00:01"


class HaiSplitter(Splitter):
    """'train1' + 'train2' to train, 'test2' halved into validation and test."""

    def __init__(
        self,
        source: Path = Path("data/interim/hai/full_hai.parquet"),
        out_dir: Path = Path("data/interim/hai/split"),
        valid_start: str = VALID_START,
        test_start: str = TEST_START,
        name: str = "hai",
    ) -> None:
        super().__init__(source, out_dir, valid_start, test_start, name)


def main() -> None:
    HaiSplitter().run(overwrite=True)


if __name__ == "__main__":
    main()


__all__ = ["HaiSplitter", "VALID_START", "TEST_START"]
