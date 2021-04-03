import numpy as np
from potentials import Potential

class NuclearSystem:
  const1 = 41.8016
  const2 = 1.439965

  def __init__(self, quantum_numbers, A1, A2, Z1, Z2):
    self.mass = A1 * A2 / (A1 + A2) / self.const1
    self.Z = Z1 * Z2 * self.const2
    self.L, self.J = quantum_numbers.values()

  def potential(self, r):
    V = Potential(r)
    return V.Woods_Saxon(self) + V.Coulomb(self)


class Calcium40_nucleon(NuclearSystem):
  A1, Z1 = 40, 20
  A2 = 1

  def __init__(self, quantum_numbers, nucleon):
    Z2 = nucleon.lower().startswith("p")
    super().__init__(quantum_numbers, self.A1, self.A2, self.Z1, Z2)
    self.V0, self.Vso = -54.823, -10
    self.a, self.r0 = 0.65, 1.2361
    self.nucleon = nucleon

  def __repr__(self):
    L_letter = ["s", "p", "d", "f", "g", "h", "j"]
    return f"40Ca+{self.nucleon} | {L_letter[self.L]}{int(self.J * 2)}/2"
