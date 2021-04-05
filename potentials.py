import numpy as np

class Potential:
  def __init__(self, r):
    self.r = r
  
  def Woods_Saxon(self, nuclear_system):
    f = 1 / (1 + np.exp((self.r-nuclear_system.R) / nuclear_system.a))
    V_central = nuclear_system.V0 * f
    LS = 0.5 * (nuclear_system.J * (nuclear_system.J + 1) - nuclear_system.L * (nuclear_system.L + 1) - 0.75)
    V_LS = 1.414**2 * nuclear_system.Vso * LS * f * (1 - f) / nuclear_system.a / self.r
    return V_central + V_LS

  def Gaussian_potential(self, nuclear_system):
    return nuclear_system.V0 * np.exp(-nuclear_system.beta * self.r**2)
  
  def Coulomb(self, nuclear_system):
    if nuclear_system.Rc == 0:
      return nuclear_system.Z / self.r

    Vc_1 = nuclear_system.Z / 2 / nuclear_system.Rc * (3 - self.r**2 / nuclear_system.Rc**2)
    Vc_2 = nuclear_system.Z / self.r
    return np.hstack((Vc_1[self.r <= nuclear_system.Rc], Vc_2[self.r > nuclear_system.Rc]))
