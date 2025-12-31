"""
The package including the modules of DEARI.

Notes
-----
This implementation follows the PyPOTS module style used by BRITS/CSAI.
"""

# Created by Ao Zhang <ao.zhang@kcl.ac.uk>
# License: BSD-3-Clause

from .backbone import BackboneDEARI, BackboneBDEARI
from .layers import (
    FeatureRegression,
    Decay,
    BayesianGRUCell,
    BayesianLSTMCell,
)

__all__ = [
    "BackboneDEARI",
    "BackboneBDEARI",
    "FeatureRegression",
    "Decay",
    "BayesianGRUCell",
    "BayesianLSTMCell",
]
