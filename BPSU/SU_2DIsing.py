import collections
import functools
import itertools
import os
import time
from math import pi, log2, log10

import xyzpy as xyz
import autoray as ar
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.circuit import Gate

from quimb.experimental.belief_propagation.l2bp import (
    L2BP,
    compress_l2bp,
    contract_l2bp,
)
from quimb.experimental.belief_propagation.l1bp import (
    contract_l1bp,
)