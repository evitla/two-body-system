import numpy as np
import matplotlib.pyplot as plt
from solve import solve

def phase_shifts(r, nuclear_system, basis, N_laguerre):
  solution = solve(nuclear_system, basis, N_laguerre)
  energies =          solution["energies"]
  eigenvectors =      solution["eigenvectors"]
  coul_energies =     solution["coulomb energies"]
  coul_eigenvectors = solution["coulomb eigenvectors"]
  Vs =                solution["nuclear potential"]

  bound_energies = energies[energies < 0]
  discretized_continuum_energies = energies[energies >= 0]

  energy_intervals, energy_widths = calculate_intervals_and_widths(discretized_continuum_energies)
  coul_energy_intervals, coul_energy_widths = calculate_intervals_and_widths(coul_energies)

  eig_G = calculate_G_eigenvalues(
    discretized_continuum_energies, energy_intervals, energy_widths,
    coul_energies, coul_energy_intervals, coul_energy_widths
  )

  T_matrix = calculate_T_matrix(
    eigenvectors, coul_eigenvectors, Vs, bound_energies,
    coul_energies, coul_energy_widths, eig_G
  )

  return coul_energies, 180 / np.pi * np.arctan(T_matrix.imag / T_matrix.real)


def calculate_intervals_and_widths(energies):
  energies = np.hstack(([0], energies))   # continuum starts with 0 energy
  intervals = 0.5 * (energies[1:] + energies[:-1])
  widths = intervals[1:] - intervals[:-1]
  return intervals, widths


def calculate_G_eigenvalues(e1, ei1, ew1, e2, ei2, ew2):
  def helper(a, b):
    a = a.reshape(-1,1)
    b = b.reshape(-1,1)
    return (a - b.T) * np.log(np.abs(a - b.T))

  ew1 = ew1.reshape(-1,1)
  ew2 = ew2.reshape(-1,1)
  
  real_G = 1 / (ew2 * ew1.T) * (
    helper(ei2[1:], ei1[:-1]) + helper(ei2[:-1], ei1[1:]) - \
    helper(ei2[:-1], ei1[:-1]) - helper(ei2[1:], ei1[1:])
  )
  
  ei1 = ei1.reshape(-1,1)
  ei2 = ei2.reshape(-1,1)

  _min = np.minimum(ei2, ei1.T)
  _max = np.maximum(ei2, ei1.T)
  min_max = _min[1:,1:] - _max[:-1,:-1]
  
  imag_G = np.pi / (ew2 * ew1.T) * min_max * (min_max > 0)

  return real_G - 1j * imag_G


def calculate_T_matrix(C1, C2, V, be, e, ew, G):
  num_bound_energies = be.size

  be = be.reshape(-1,1)
  e = e.reshape(-1,1)

  tmp = np.dot(C2.T, V.T)
  W = C2 * tmp.T
  B = np.dot(C1.T, tmp.T)
  tmp1 = B[:num_bound_energies, :-1]**2 / (e[:-1,:].T - be)
  tmp2 = B[num_bound_energies:-1, :-1]**2 * G.T

  return (W.sum(axis=0)[:-1] + tmp1.sum(axis=0) + tmp2.sum(axis=0)) / ew
  