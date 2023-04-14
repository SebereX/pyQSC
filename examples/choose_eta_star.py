from qsc import Qsc
import numpy as np
import time

a = 0.23
b = 0.0012
B0 = 1
nfp = 4
rcOr=[1, a, b]
zsOr=[0.0, a, b]
stel = Qsc(rc=rcOr, zs=zsOr, B0 = B0, nfp=nfp, order='r1', nphi = 61)

# Compute eta^* that extremises iota_bar_0
# Different criteria for choosing etabar can be coded (see the file in qs_optim_config.py)
etabar = stel.choose_eta()

# Construct near-axis configuration
stel.etabar = etabar
stel.calculate()

# For more general purpose optimisation there are other functions such as
# optimise_params