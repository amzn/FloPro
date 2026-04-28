from typing import Dict, NewType
from numpy import ndarray
import pandas as pd
from dataclasses import dataclass

PublicVarGroupName = NewType("PublicVarGroupName", str)


@dataclass
class PublicVarGroupMetadata:
    """
    Metadata of a PublicVar group. Each group contains a DataFrame describes
    the metadata of each individual public variable, one row per variable.
    Note that var_metadata can be different among different agent, even
    for the same PublicVar group.
    """

    name: PublicVarGroupName
    var_metadata: pd.DataFrame
    """
       example:
       t, w
       0, 1
       0, 2
    """


PublicVarsMetadata = Dict[PublicVarGroupName, PublicVarGroupMetadata]

"""
We can start with only supporting 1D array. Agent can reshape based on var_metadata, if
needed.
"""
PublicVarValues = Dict[PublicVarGroupName, ndarray]
RhoValues = Dict[PublicVarGroupName, ndarray]
Prices = Dict[PublicVarGroupName, ndarray]


@dataclass
class Residuals:
    """
    Container for primal and dual residuals in ADMM optimization.

    In ADMM, residuals are used to monitor convergence:
    - Primal residual measures constraint violation: ||x_i - z||
    - Dual residual measures change in consensus: ||rho * (z^k - z^{k-1})||
    """

    # TODO: Determine if we want variable-level residuals instead of scalars.
    primal: float
    dual: float
