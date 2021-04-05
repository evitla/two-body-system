import numpy as np
from math import gamma, factorial
from scipy.special import erf

from .decorators import gauss_laguerre_quadrature

class GaussianBasis:
  def __init__(self, parameters, quadrature, hermitian=True):
    if quadrature == "geometric_progression":
      r0, rmax, N = parameters.values()
      r_i = r0 * (rmax / r0)**(np.arange(N) / N)
      alpha = 1 / r_i**2
    elif quadrature == "tchebyshev":
      a0, t, N = parameters.values()
      alpha = a0 * np.tan((2 * np.arange(N) + 1) * np.pi / 4 / N)**t
    else:
      raise ValueError("quadrature must be either 'geometric_progression' or 'tchebyshev'")
    
    self.alpha = alpha.reshape(-1,1)
    self.hermitian = hermitian
  
  def __repr__(self):
    return "Gaussian Basis"

  def overlap_matrix(self, L, eta):
    AL = self.gaussian_integral(2 * L + 2, eta + eta.T)
    BL = self.gaussian_integral(2 * L + 2, eta.conj() + eta.T)
    return 2 * (AL + BL).real

  def kinetic_potential(self, nuclear_system, eta):
    AH0 = 2 * eta * eta.T * self.gaussian_integral(2 * nuclear_system.L + 4, eta + eta.T)
    BH0 = 2 * eta.conj() * eta.T * self.gaussian_integral(2 * nuclear_system.L + 4, eta.conj() + eta.T)
    return 2 / nuclear_system.mass * (AH0 + BH0).real

  def nuclear_potential(self, nuclear_system, eta, N):
    if nuclear_system.potential_type == "gaussian":
      AVs = self.gaussian_integral(2 * nuclear_system.L + 2, eta + eta.T + nuclear_system.beta)
      BVs = self.gaussian_integral(2 * nuclear_system.L + 2, eta.conj() + eta.T + nuclear_system.beta)
      return nuclear_system.V0 * 2 * (AVs + BVs).real
    
    @gauss_laguerre_quadrature(N)
    def potential_integral_function(r, L, eta):
      wf = self.gaussians(r, L, eta)
      tmp_ijr = wf.reshape(-1,1,r.size) * wf.reshape(1,-1,r.size)
      return nuclear_system.potential(r)["nuclear potential"] * tmp_ijr * np.exp(r)
    return potential_integral_function(nuclear_system.L, eta)

  def coulomb_potential(self, nuclear_system, eta, N):
    if nuclear_system.Z == 0:
      return 0

    AVc_0_inf =  0.5 * factorial(nuclear_system.L) / (eta + eta.T)**(nuclear_system.L + 1)
    BVc_0_inf = 0.5 * factorial(nuclear_system.L) / (eta.conj() + eta.T)**(nuclear_system.L + 1)

    if nuclear_system.Rc == 0:
      return nuclear_system.Z * 2 * (AVc_0_inf + BVc_0_inf).real

    from_0_to_Rc = lambda a: 3 * self.integral_for_coulomb(2 * nuclear_system.L + 2, a, nuclear_system.Rc) - \
      1 / nuclear_system.Rc**2 * self.integral_for_coulomb(2 * nuclear_system.L + 4, a, nuclear_system.Rc)
    
    AVc_0_Rc = from_0_to_Rc(eta + eta.T)
    BVc_0_Rc = from_0_to_Rc(eta.conj() + eta.T)
    Vc_0_Rc = (AVc_0_Rc + BVc_0_Rc).real / nuclear_system.Rc
    
    AVc_Rc_inf = AVc_0_inf - self.integral_for_coulomb(2 * nuclear_system.L + 1, eta + eta.T, nuclear_system.Rc)
    BVc_Rc_inf = BVc_0_inf - self.integral_for_coulomb(2 * nuclear_system.L + 1, eta.conj() + eta.T, nuclear_system.Rc)
    Vc_Rc_inf = 2 * (AVc_Rc_inf + BVc_Rc_inf).real

    return nuclear_system.Z * (Vc_0_Rc + Vc_Rc_inf)

  def hamiltonian(self, nuclear_system, N):
    b = np.pi / 2 if nuclear_system.L == 0 else np.pi / nuclear_system.L / 3
    eta = self.alpha * complex(1, b)
    n = nuclear_system.L + 1.5
    N_cos = np.sqrt(4 * (2 * self.alpha * np.sqrt(1 + b**2))**n / \
            gamma(n) / ((1 + b**2)**(n / 2) + np.cos(n * np.arctan(b))))
    N_ij = N_cos * N_cos.T / 4
    L = N_ij * self.overlap_matrix(nuclear_system.L, eta)
    H0 = N_ij * self.kinetic_potential(nuclear_system, eta)
    Vs = N_ij * self.nuclear_potential(nuclear_system, eta, N)
    Vc = N_ij * self.coulomb_potential(nuclear_system, eta, N)
    Hc = H0 + Vc
    H = Hc + Vs

    low_matrix = np.linalg.cholesky(L)
    up_matrix = low_matrix.T
    inv_low_matrix = np.linalg.inv(low_matrix)
    inv_up_matrix = np.linalg.inv(up_matrix)
    new_H = np.dot(np.dot(inv_low_matrix, H), inv_up_matrix)
    new_Hc = np.dot(np.dot(inv_low_matrix, Hc), inv_up_matrix)
    return new_H, new_Hc, Vs, inv_up_matrix

  def wavefunction(self, r, eigenvector, L):
    b = np.pi / 2 if L == 0 else np.pi / L / 3
    eta = self.alpha * complex(1, b)
    n = L + 1.5
    N_cos = np.sqrt(4 * (2 * self.alpha * np.sqrt(1 + b**2))**n / \
            gamma(n) / ((1 + b**2)**(n / 2) + np.cos(n * np.arctan(b))))
    wf = np.dot(eigenvector.T, N_cos / 2 * self.gaussians(r, L, eta))
    change_sign = np.abs(wf.max()) < np.abs(wf.min())
    return -wf if change_sign else wf

  def integral_for_coulomb(self, n, a, rmax):
    '''
    Integrate r^n * \exp(-a * r^2) from 0 to rmax
    '''
    if n == 0:
      return 0.5 * np.sqrt(np.pi / a) * erf(np.sqrt(a) * rmax)
    if n == 1:
      return 0.5 * (1 - np.exp(-a * rmax**2)) / a
    return 0.5 * (n - 1) / a * self.integral_for_coulomb(n-2, a, rmax) - 0.5 * rmax**(n - 1) / a * np.exp(-a * rmax**2)

  @staticmethod
  def gaussians(r, L, eta):
    r = r.reshape(1,-1)
    psi = r**L * np.exp(-eta * r**2)
    return psi.conj() + psi
  
  @staticmethod
  def gaussian_integral(n, a):
    return 0.5 * gamma(n / 2 + 0.5) / a**(n / 2 + 0.5)
