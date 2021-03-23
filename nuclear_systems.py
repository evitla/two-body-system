import numpy as np
from potentials import Potential

class NuclearSystem:
  const1 = 41.8016
  const2 = 1.439965

  def __init__(self, quantum_numbers, nucleon, A, Z):
    self.mass = A / (1 + A) / self.const1
    self.Z = Z * self.const2 * nucleon.lower().startswith("p")
    self.L, self.J = quantum_numbers.values()
    self.nucleon = nucleon

  def potential(self, r):
    V = Potential(r)
    return V.Woods_Saxon(self) + V.Coulomb(self)
