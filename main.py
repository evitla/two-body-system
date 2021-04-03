#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from nuclear_systems import *
from bases.gaussian import GaussianBasis
from anc import anc

N_laguerre = 90
parameters = {
  "r0": 1.5,
  "rmax": 22.5,
  "N": 32
}

gaussian_basis = GaussianBasis(parameters, quadrature="geometric_progression")

quantum_numbers = {
  "L": 3,
  "J": 3.5
}

Ca40_n = Calcium40_nucleon(quantum_numbers, nucleon="n")

r = np.arange(0, 15, 0.01)
anc_func = anc(r, Ca40_n, gaussian_basis, N_laguerre)

plt.plot(r, anc_func, lw=2, label=Ca40_n)
plt.legend()
plt.show()
