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


class Calcium40_nucleon(NuclearSystem):
  A, Z = 40, 20

  def __init__(self, quantum_numbers, nucleon):
    super().__init__(quantum_numbers, nucleon, A=self.A, Z=self.Z)
    self.V0, self.Vso = -54.823, -10
    self.a, self.r0 = 0.65, 1.2361

  def __repr__(self):
    L_letter = ["s", "p", "d", "f", "g", "h", "j"]
    return f"40Ca+{self.nucleon} | {L_letter[self.L]}{int(self.J * 2)}/2"
