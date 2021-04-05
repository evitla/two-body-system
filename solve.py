import numpy as np
from states import SingleParticleStationaryState

def solve(nuclear_system, basis, N_laguerre):
  H, Hc, Vs, inv_up_matrix = basis.hamiltonian(nuclear_system, N_laguerre)
  energies, eigenvectors = eigensolve(H, basis.hermitian)
  eigenvectors = np.dot(inv_up_matrix, eigenvectors)
  coul_energies, coul_eigenvectors = eigensolve(Hc, basis.hermitian)
  coul_eigenvectors = np.dot(inv_up_matrix, coul_eigenvectors)
  create_state = lambda en, ev: SingleParticleStationaryState(en, ev,
                                                nuclear_system, basis)
  states = [create_state(en, ev) for en, ev in zip(energies, eigenvectors.T)]
  return {
    "energies":             energies,
    "eigenvectors":         eigenvectors,
    "coulomb energies":     coul_energies,
    "coulomb eigenvectors": coul_eigenvectors,
    "states":               states,
    "nuclear potential":    Vs
  }


def eigensolve(H, hermitian):
  if hermitian:
    return np.linalg.eigh(H)
  
  energies, eigenvectors = np.linalg.eig(H)
  idx_sort = np.argsort(energies)
  energies = np.real_if_close(energies[idx_sort])
  eigenvectors = eigenvectors[:,idx_sort]
  return energies, eigenvectors
