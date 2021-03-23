import numpy as np
from states import SingleParticleStationaryState

def solve(nuclear_system, basis, N_laguerre):
  H = basis.hamiltonian(nuclear_system, N_laguerre)
  energies, eigenvectors = eigensolve(H, basis.hermitian)
  create_state = lambda en, ev: SingleParticleStationaryState(en, ev,
                                                nuclear_system, basis)
  states = [create_state(en, ev) in zip(energies, eigenvectors.T)]
  return states


def eigensolve(H, hermitian):
  if hermitian:
    return np.linalg.eigh(H)
  
  energies, eigenvectors = np.linalg.eig(H)
  idx_sort = np.argsort(energies)
  energies = np.real_if_close(energies[idx_sort])
  eigenvectors = eigenvectors[:,idx_sort]
  return energies, eigenvectors
