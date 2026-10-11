"""Data management: bringing sources into data/raw/."""

from dencad.utils import Params

# Re-exported on purpose, so the name is part of this subpackage's public API
# rather than an unused import: every data helper takes a `Params`, and importing
# it from the same place keeps call sites short.
__all__ = ["Params"]
