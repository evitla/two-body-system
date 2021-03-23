class SingleParticleStationaryState:
  def __init__(self, energy, eigenvector, nuclear_system, basis):
    self.energy = energy
    self.eigenvector = eigenvector
    self.nuclear_system = nuclear_system
    self.basis = basis

  def wavefunction(self, r):
    return self.basis.wavefunction(r, self.eigenvector, self.nuclear_system.L)