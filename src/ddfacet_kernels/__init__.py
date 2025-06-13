# Copyright (C) 2025 South African Radio Astronomy Observatory,
# Rhodes University, l'Observatoire de Paris

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import dataclasses
import logging
from numbers import Integral
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import numpy.fft as npfft

try:
  import numba
except ImportError:
  numba = None

__version__ = "0.1.2"

log = logging.getLogger(__name__)

LIGHTSPEED = 2.99792458e8
# TEST_DATA_FILENAME = "test-data.pickle.xz"


def gen_numba_coeffs(x: npt.NDArray, y: npt.NDArray, order: int) -> npt.NDArray:
  ncols = (order + 1) ** 2
  coeffs = np.empty((x.size, ncols), dtype=x.dtype)
  c = 0

  for i in range(order + 1):
    for j in range(order + 1):
      for k in range(x.size):
        coeffs[k, c] = x[k] ** i * y[k] ** j

      c += 1

  return coeffs


def gen_numpy_coeffs(x: npt.NDArray, y: npt.NDArray, order: int) -> npt.NDArray:
  i, j = (a.ravel() for a in np.mgrid[: order + 1, : order + 1])
  return x[:, None] ** i[None, :] * y[:, None] ** j[None, :]


def polyfit2d(
  x: npt.NDArray, y: npt.NDArray, z: npt.NDArray, order: int = 3
) -> npt.NDArray:
  """Given ``x`` and ``y`` data points and ``z``, some
  values related to ``x`` and ``y``, fit a polynomial
  of order ``order`` to ``z``.

  Derived from https://stackoverflow.com/a/7997925
  """
  return np.linalg.lstsq(gen_coeffs(x, y, order), z, rcond=-1)[0]


def numba_polyval2d(x: npt.NDArray, y: npt.NDArray, coeffs: int) -> npt.NDArray:
  """Reproduce values from a two-dimensional polynomial fit.

  Derived from https://stackoverflow.com/a/7997925
  """
  order = int(np.sqrt(coeffs.size)) - 1
  assert (order + 1) ** 2 == coeffs.size
  z = np.zeros_like(x)
  c = 0

  for i in range(order + 1):
    for j in range(order + 1):
      a = coeffs[c]
      for k in range(x.shape[0]):
        z[k] += a * x[k] ** i * y[k] ** j

      c += 1

  return z


def numpy_polyval2d(x: npt.NDArray, y: npt.NDArray, coeffs: int) -> npt.NDArray:
  """Reproduce values from a two-dimensional polynomial fit.

  Derived from https://stackoverflow.com/a/7997925
  """
  order = int(np.sqrt(coeffs.size)) - 1
  i, j = (a.ravel() for a in np.mgrid[: order + 1, : order + 1])
  exp = x[None, :, :] ** i[:, None, None] * y[None, :, :] ** j[:, None, None]
  return np.sum(coeffs[:, None, None] * exp, axis=0)


# Spheroidal coefficients derived from
# Rational Approximations to Selected 0-order Spheroidal Functions
# https://library.nrao.edu/public/memos/vla/comp/VLAC_156.pdf
# Table IIIA (c) and Table IIIB (c) respectively

# These values exist for a support width m = 6
# First elements are for |nu| < 0.75 and second for 0.75 <= |nu| <= 1.0

# NOTE(sjperkins)
# The above support width is generally
# much smaller than the filter support sizes:
# In spheroidal_2d the domain is (-1.0, 1.0]

P = np.array(
  [
    [8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
    [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2],
  ]
)

Q = np.array(
  [[1.0000000e0, 8.212018e-1, 2.078043e-1], [1.0000000e0, 9.599102e-1, 2.918724e-1]]
)


def numba_spheroidal_2d(npix: int, factor: float = 1.0) -> npt.NDArray:
  result = np.empty((npix, npix), dtype=np.float64)
  c = np.linspace(-1.0, 1.0, npix)

  for y, yc in enumerate(c):
    y_sqrd = yc**2
    for x, xc in enumerate(c):
      r = np.sqrt(xc**2 + y_sqrd) * factor

      if r >= 0.0 and r < 0.75:
        poly = 0
        end = 0.75
      elif r >= 0.75 and r <= 1.00:
        poly = 1
        end = 1.00
      else:
        result[y, x] = 0.0
        continue

      sP = P[poly]
      sQ = Q[poly]

      nu_sqrd = r**2
      del_nu_sqrd = nu_sqrd - end * end

      top = sP[0]
      del_nu_sqrd_pow = del_nu_sqrd

      for i in range(1, 5):
        top += sP[i] * del_nu_sqrd_pow
        del_nu_sqrd_pow *= del_nu_sqrd

      bot = sQ[0]
      del_nu_sqrd_pow = del_nu_sqrd

      for i in range(1, 3):
        bot += sQ[i] * del_nu_sqrd_pow
        del_nu_sqrd_pow *= del_nu_sqrd

      result[y, x] = (1.0 - nu_sqrd) * (top / bot)

  return result


def numpy_spheroidal_2d(npix: int, factor: float = 1.0) -> npt.NDArray:
  """Numpy implementation of spheroidal_2d"""
  x = np.mgrid[-1 : 1 : 1j * npix] ** 2
  r = np.sqrt(x[:, None] + x[None, :]) * factor

  bin1 = np.logical_and(r >= 0.0, r < 0.75)
  bin2 = np.logical_and(r >= 0.75, r <= 1.00)
  bin3 = np.invert(np.logical_or(bin1, bin2))

  def _eval_spheroid(nu, part, end):
    sP = P[part]
    sQ = Q[part]

    nu_sqrd = nu**2
    del_nu_sqrd = nu_sqrd - end * end
    powers = del_nu_sqrd[:, None] ** np.arange(5)

    top = np.sum(sP[None, :] * powers, axis=1)
    bot = np.sum(sQ[None, :] * powers[:, 0:3], axis=1)

    return (1.0 - nu_sqrd) * (top / bot)

  result = np.empty_like(r)

  result[bin1] = _eval_spheroid(r[bin1], 0, 0.75)
  result[bin2] = _eval_spheroid(r[bin2], 1, 1.00)
  result[bin3] = 0.0

  return result


# Defer to numba variants if available
if numba is not None:
  gen_coeffs = numba.njit(nogil=True, cache=True)(gen_numba_coeffs)
  spheroidal_2d = numba.njit(nogil=True, cache=True)(numba_spheroidal_2d)
  polyval2d = numba.njit(nogil=True, cache=True)(numba_polyval2d)
else:
  gen_coeffs = gen_numpy_coeffs
  spheroidal_2d = numpy_spheroidal_2d
  polyval2d = numpy_polyval2d


def fft2(A) -> npt.NDArray:
  """Shifted fft2 normalised by the size of A"""
  return npfft.fftshift(npfft.fft2(npfft.ifftshift(A))) / np.float64(A.size)


def ifft2(A) -> npt.NDArray:
  """Shifted ifft2 normalised by the size of A"""
  return npfft.fftshift(npfft.ifft2(npfft.ifftshift(A))) * np.float64(A.size)


def zero_pad(img, npix) -> npt.NDArray:
  """Zero pad ``img`` up to ``npix``"""

  if isinstance(npix, Integral):
    npix = (npix,) * img.ndim

  padding = []

  for dim, npix_ in zip(img.shape, npix):
    # Pad and half-pad amount
    p = npix_ - dim
    hp = p // 2

    # Pad the image
    padding.append((hp, hp) if p % 2 == 0 else (hp + 1, hp))

  return np.pad(img, padding, "constant", constant_values=0)


def spheroidal_aa_filter(
  npix: int, support: int = 11, spheroidal_resolution: int = 111
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
  """Computes a spheroidal anti-aliasing filter in the image domain

  Args:
    npix : Number of image pixels in X and Y
    support : Support in the fourier domain
    spheroidal_resolution : Size of the spheroidal
        generated in the image domain.
        Increasing this may increase the accuracy of the kernel
        at the cost of compute and result in an increased
        sampling resolution in the image domain.

  Returns:
    convolution_filter : The original spheroidal
    gridding_kernel : Gridding kernel in the fourier domain
    spheroidal_convolution_filter : The spheroidal in the image domain
        resized to the size of the actual field
  """
  # Convolution filter
  cf = spheroidal_2d(spheroidal_resolution).astype(np.complex128)
  # Fourier transformed convolution filter
  fcf = fft2(cf)

  # Cut the support out
  xc = spheroidal_resolution // 2
  start = xc - support // 2
  end = 1 + xc + support // 2
  fcf = fcf[start:end, start:end].copy()

  # Pad and ifft2 the fourier transformed convolution filter
  # This results in a spheroidal which covers all the pixels
  # in the image
  zfcf = zero_pad(fcf, npix)
  ifzfcf = ifft2(zfcf)
  ifzfcf[ifzfcf < 0] = 1e-10

  return cf, fcf, ifzfcf


def delta_n_coefficients(
  l0: float, m0: float, radius: float = 1.0, order: int = 4
) -> Tuple[Tuple[float, float], npt.NDArray]:
  """Returns polynomical coefficients fitting the difference
  of coordinate n between a grid of (l,m) values centred
  around (l0, m0).

  Args:
    l0 : Facet centre l coordinate
    m0 : Facet centre m coordinate
    radius : Radius of the grid to fit around (l0, m0),
        probably in dimensionless coordinates
    order : The order of the polynomial fit.

  Returns:
    A tuple :code:`((cl, cm), coeffs)` tuple where
    ``cl`` and ``cm`` are the first order linear coefficients
    and ``coeffs`` is an array of the higher order coefficients.
  """

  # Create 100 points around (l0, m0) in the given radius
  Np = 100 * 1j

  l, m = np.mgrid[l0 - radius : l0 + radius : Np, m0 - radius : m0 + radius : Np]

  l = l.ravel()
  m = m.ravel()

  dl = l - l0
  dm = m - m0

  lm = np.sqrt(l**2 + m**2)

  if (max_lm := np.max(lm)) > 1.0:
    raise ValueError(
      f"There are {l} and {m} coordinates that lie "
      f"off the unit sphere. max_lm={max_lm}. "
      f"lm = {lm.min():.2f} => {lm.max():.2f}"
    )

  # Create coefficients fitting the difference between the
  # n coordinates of the lm grid and that of the phase centre.
  dn = np.sqrt(1 - l**2 - m**2) - np.sqrt(1 - l0**2 - m0**2)
  coeff = polyfit2d(dl, dm, dn, order=order).reshape((order + 1, order + 1))

  # See Section 2.2 of the DDFacet paper as wels
  # as the the Kogan & Greisen 2009 reference.
  # The first order coefficients of the polynomial fit
  # are equivalent to a w-dependent (u,v) coordinate scaling.
  # Removing the 1st order coefficients reduces the support size
  # of the fitted w kernels. These coefficients are applied within
  # the gridder itself.
  # Note that Cl and Cm are swapped, compared to the DDFacet code
  # in order to correct a possible naming error.
  Cl = coeff[1, 0]
  Cm = coeff[0, 1]
  coeff[1, 0] = 0
  coeff[0, 1] = 0

  return (Cl, Cm), coeff.ravel()


def reorganise_convolution_filter(cf: npt.NDArray, oversampling: int) -> npt.NDArray:
  """Reorganises the convolution filter so that the kernels
  can be accessed by their oversampling factor, presumably
  for more optimal access patterns in the gridding kernel.

  Args:
    cf : Oversampled convolution filter
    oversampling : Oversampling factor

  Returns:
    Reorganised convolution filter
  """
  support = cf.shape[0] // oversampling
  result = np.empty((oversampling, oversampling, support, support), dtype=cf.dtype)

  for i in range(oversampling):
    for j in range(oversampling):
      result[i, j, :, :] = cf[i::oversampling, j::oversampling]

  return result.reshape(cf.shape)


def find_max_support(radius: float, maxw: float, min_wave: float) -> int:
  """Find the maximum support.

  Args:
    radius: Radius in degrees
    maxw: Maximum w value, in metres
    min_wave: Minimum wavelength, in metres

  Returns:
    The maximum support
  """
  # Assumed maximum support
  max_support = 501

  # spheroidal convolution filter for the maximum support size
  _, _, spheroidal_w = spheroidal_aa_filter(max_support)

  # Compute l, m and n-1 over the area of maximum support
  ex = radius * np.sqrt(2.0)
  l, m = np.mgrid[-ex : ex : max_support * 1j, -ex : ex : max_support * 1j]
  n_1 = np.sqrt(1.0 - l**2 - m**2) - 1.0

  # Multiplying the w term by the spheroidal gives the convolution of both in the fourier domain
  # If the W is larger then the final support will be commensurately larger
  # due to larger W's resulting in higher frequency fringe patterns
  w = np.exp(-2.0 * 1j * np.pi * (maxw / min_wave) * n_1) * spheroidal_w
  fw = fft2(w)

  # NOTE: fw is symmetrical
  # This takes a slice half-way through the first dimension
  fw1d = np.abs(fw[(max_support - 1) // 2, :])
  # Normalise
  fw1d /= np.max(fw1d)
  # Slight optimation: take a half-slice through
  # the 1d shape due to symmetry
  fw1d = fw1d[(max_support - 1) // 2 :]

  # Sort the values and return the interpolated support
  # associated with a small value.

  # NOTE: The reasoning behind why this works:
  # Assume the w term convolved with the prolate spheroidal integrates to 1.0
  # As the support is technically infinite, we find an approximation
  # of it within some error bound that limits the support.
  # This is achieved by throwing away the tails of the function or,
  # in the np.interp function below, 0.1% of the integrated spheroidal
  ind = np.argsort(fw1d)
  x = fw1d[ind]

  if False:
    import matplotlib.pyplot as plt

    plt.figure().add_subplot(111).plot(x)
    plt.show()

  return np.interp(1e-3, x, np.arange(x.size)[ind])


@dataclasses.dataclass
class FacetWKernelData:
  """Stores W kernel related data for a Facet"""

  # LM phase centre
  lm: Tuple[float, float]
  # First order L and M polynomial coefficients
  # applied as scaling factors to UVW coordinates.
  # This naming convention differs from DDFacet which
  # uses Cu and Cv but aligns with the following
  # expression from Kogan & Griesen 2009
  #   U′= U − W · l0
  #   V′= V − W · m0
  clm: Tuple[float, float]
  # The kernel support
  support: int
  # The kernel oversampling factor
  oversampling: int
  # W values for each plane
  w_values: npt.NDArray[np.float64]
  # Flattened W Kernels for each W plane
  # Can be reshaped to (oversampling, oversampling, support, support)
  w_kernels: List[npt.NDArray[np.complex64]]
  # Flattened Conjugate W kernels for each W plane
  # Can be reshaped to (oversampling, oversampling, support, support)
  w_kernels_conj: List[npt.NDArray[np.complex64]]

  @property
  def l(self) -> float:
    """Phase centre l coordinate"""
    return self.lm[0]

  @property
  def m(self) -> float:
    """Phase centre m coordinate"""
    return self.lm[1]

  @property
  def cl(self) -> float:
    """First order L polynomial coefficient"""
    return self.clm[0]

  @property
  def cm(self) -> float:
    """First order M polynomial coefficient"""
    return self.clm[1]

  @property
  def nwplanes(self) -> int:
    """Number of w planes"""
    return len(self.w_kernels)


def facet_w_kernels(
  nwplanes: int,
  cell_size: float,
  support: int,
  maxw: float,
  npix: int,
  oversampling: int,
  lmshift: Tuple[float],
  frequencies: npt.NDArray[np.floating],
) -> FacetWKernelData:
  """Compute Facet gridding kernels and their conjugates

  Args:
    nwplanes : Number of W planes
    cell_size : Cell size in arc seconds
    support : Support, in pixels
    maxw : Maximum W coordinate value
    npix : Number of pixels
    oversampling : oversampling factor, in pixels
    lmshift : (l0, m0) coordinate for this convolution filter.
    frequencies : Array of frequencies in Hz

  Returns:
    Dataclass containing WTerm information
  """

  # Radius in radians, given cell size and number of pixels
  radius = np.deg2rad((npix / 2.0) * cell_size / 3600.0)

  # Minimum wavelength
  min_wave = LIGHTSPEED / frequencies.min()

  # Find the maximum support
  max_support = find_max_support(radius, maxw, min_wave)

  # W values for each plane
  w_values = np.linspace(0, maxw, nwplanes)

  # Create supports for each w plane
  w_supports = np.linspace(support, max(max_support, support), nwplanes, dtype=np.int64)

  # Make any even support odd
  w_supports[w_supports % 2 == 0] += 1

  # Extract lm coordinates if given
  l0, m0 = lmshift

  # Get the first order n coefficients
  # separated from the higher order coefficients
  (cl, cm), dn_coeffs = delta_n_coefficients(l0, m0, 3 * radius, order=5)

  # NOTE: In the single W plane case, no polynomial fit is applied
  # It may be worth just removing this case as it can't hurt to
  # apply the polynomial anyway?
  # Perhaps this case never applies in practice...
  if len(w_values) <= 1:
    # Simplified single kernel case
    _, _, spheroidal_pw = spheroidal_aa_filter(w_support[0])
    w = np.abs(spheroidal_pw)
    zw = zero_pad(w, w.shape[0] * oversampling)
    zw_conj = np.conj(zw)

    fzw = fft2(zw)
    fzw_conj = fft2(zw_conj)

    fzw = reorganise_convolution_filter(fzw)
    fzw_conj = reorganise_convolution_filter(fzw_conj)

    # Ensure complex64, aligned and contiguous
    fzw = np.require(fzw, dtype=np.complex64, requirements=["A", "C"])
    fzw_conj = np.require(fzw_conj, dtype=np.complex64, requirements=["A", "C"])

    return FacetWKernelData(
      (l0, m0), (cl, cm), support, oversampling, w_values, [fzw], [fzw_conj]
    )

  wkernels = []
  wkernels_conj = []

  # For each w plane and associated support
  for plane_w, w_support in zip(w_values, w_supports):
    # Normalise plane w
    norm_plane_w = plane_w / min_wave

    # Calculate the spheroidal for the given support
    _, _, spheroidal_pw = spheroidal_aa_filter(w_support)

    # Fit n-1 for this w plane using
    # delta n polynomial coefficients
    ex = radius - radius / w_support
    l, m = np.mgrid[-ex : ex : w_support * 1j, -ex : ex : w_support * 1j]
    n_1 = polyval2d(l, m, dn_coeffs)

    # Multiply in complex exponential
    # and the spheroidal for this plane
    # Convolution theorem?
    w = np.exp(-2.0 * 1j * np.pi * norm_plane_w * n_1) * np.abs(spheroidal_pw)

    # zero pad w, adding oversampling
    zw = zero_pad(w, w.shape[0] * oversampling)
    zw_conj = np.conj(zw)

    # Now fft2 zero padded w and conjugate
    fzw = fft2(zw)
    fzw_conj = fft2(zw_conj)

    fzw = reorganise_convolution_filter(fzw, oversampling)
    fzw_conj = reorganise_convolution_filter(fzw_conj, oversampling)

    # Ensure complex64, aligned and contiguous
    fzw = np.require(fzw, dtype=np.complex64, requirements=["A", "C"])
    fzw_conj = np.require(fzw_conj, dtype=np.complex64, requirements=["A", "C"])

    wkernels.append(fzw)
    wkernels_conj.append(fzw_conj)

  return FacetWKernelData(
    (l0, m0), (cl, cm), support, oversampling, w_values, wkernels, wkernels_conj
  )
