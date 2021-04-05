import numpy as np
from potentials import Potential

class NuclearSystem:
  const2 = 1.439965

  def __init__(self, quantum_numbers):
    self.mass = self.A1 * self.A2 / (self.A1 + self.A2) / self.const1
    self.Z = self.Z1 * self.Z2 * self.const2
    self.L, self.J = quantum_numbers.values()

  def potential(self, r):
    V = Potential(r)
    if self.potential_type == "woods-saxon":
      return {
        "nuclear potential": V.Woods_Saxon(self),
        "coulomb potential": V.Coulomb(self)
      }

    if self.potential_type == "gaussian":
      return {
        "nuclear potential": V.Gaussian_potential(self),
        "coulomb potential": V.Coulomb(self)
      }
    
    raise TypeError(
      f"Potential type could be either 'woods-saxon' or 'gaussian', not '{self.potential_type}'"
    )


class Calcium40_nucleon(NuclearSystem):
  const1 = 41.8016
  potential_type = "woods-saxon"
  A1, Z1 = 40, 20
  A2 = 1

  def __init__(self, quantum_numbers, nucleon):
    self.Z2 = nucleon.lower().startswith("p")
    super().__init__(quantum_numbers)
    self.V0, self.Vso = -54.823, -10
    self.a, self.r0 = 0.65, 1.2361
    self.R = self.r0 * (self.A1 + self.A2)**(1/3)
    self.Rc = self.R
    self.nucleon = nucleon

  def __repr__(self):
    L_letter = ["s", "p", "d", "f", "g", "h", "j"]
    return f"40Ca+{self.nucleon} | {L_letter[self.L]}{int(self.J * 2)}/2"


class Alpha_deuteron(NuclearSystem):
  const1 = 41.4686
  potential_type = "gaussian"
  A1, Z1 = 4, 2
  A2, Z2 = 2, 1

  def __init__(self, quantum_numbers):
    super().__init__(quantum_numbers)
    self.V0, self.beta = -76.12, 0.2
    self.Rc = 0
