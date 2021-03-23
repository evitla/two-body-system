import numpy as np
from mpmath import whitw
from solve import solve

def anc(r, nuclear_system, basis, N_laguerre):
  bound_state = solve(nuclear_system, basis, N_laguerre)[0]
  wf = r * basis.wavefunction(r, bound_state.eigenvector, nuclear_system.L)
  
  k = np.sqrt(-2 * nuclear_system.mass * bound_state.energy)
  eta = nuclear_system.Z * nuclear_system.mass / k.real
  whitw_ = np.zeros(r.size)
  for i in range(r.size):
    whitw_[i] = whitw(-eta, nuclear.L + 0.5, 2 * k.real * r[i])

  return wf.real / whitw_
