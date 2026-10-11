"""DENCAST reimplemented in PySpark, and DENCAD built on top of it.

Project layout follows Cookiecutter Data Science; the pipeline is orchestrated
by DVC (see dvc.yaml).
"""

from dencad.utils import Params

__all__ = ["Params"]
__version__ = "0.1.0"
