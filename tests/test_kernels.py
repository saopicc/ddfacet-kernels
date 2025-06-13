import numpy as np
from numpy.testing import (
  assert_array_almost_equal,
  assert_allclose,
)
import pytest

from ddfacet_kernels import (
  numpy_polyval2d,
  numba_polyval2d,
  numpy_spheroidal_2d,
  numba_spheroidal_2d,
  gen_numba_coeffs,
  gen_numpy_coeffs,
  spheroidal_aa_filter,
  facet_w_kernels,
)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_gen_coefficients_allclose(order):
  """Test equivalence of numpy and numba variants"""
  x = np.random.random(100)
  y = np.random.random(100)

  assert_allclose(
    gen_numba_coeffs(x, y, order=order), gen_numpy_coeffs(x, y, order=order)
  )


@pytest.mark.parametrize("order", [2, 3, 4])
def test_polyval2_allclose(order):
  """Test equivalence of numpy and numba variants"""
  x = np.random.random(100)
  y = np.random.random(100)
  z = np.random.random(100)
  coeffs = np.linalg.lstsq(gen_numpy_coeffs(x, y, order), z, rcond=-1)[0]
  xr = x.reshape((10, 10))
  yr = y.reshape((10, 10))
  assert_allclose(numpy_polyval2d(xr, yr, coeffs), numba_polyval2d(xr, yr, coeffs))


@pytest.mark.parametrize("npix", [11, 14, 17, 20, 21, 22, 23, 24, 25])
def test_spheroidal2d_allclose(npix):
  """Test equivalence of numpy and numba variants"""
  assert_allclose(numba_spheroidal_2d(npix), numpy_spheroidal_2d(npix))


def test_ddfacet_allclose(ddf_wkernel_data):
  kw = ddf_wkernel_data["ClassWTermModified-kwargs"]
  Cu = ddf_wkernel_data["Cu"]
  Cv = ddf_wkernel_data["Cv"]

  # The kernels align
  cf, fcf, ifzfcf = spheroidal_aa_filter(kw["Npix"], kw["Sup"])
  (assert_array_almost_equal(cf, ddf_wkernel_data["CF"], decimal=7),)
  assert_array_almost_equal(fcf, ddf_wkernel_data["fCF"], decimal=7)
  assert_array_almost_equal(ifzfcf, ddf_wkernel_data["ifzfCF"], decimal=7)

  wkernel_data = facet_w_kernels(
    nwplanes=kw["Nw"],
    cell_size=kw["Cell"],
    support=kw["Sup"],
    maxw=kw["wmax"],
    npix=kw["Npix"],
    oversampling=kw["OverS"],
    lmshift=kw["lmShift"],
    frequencies=kw["Freqs"],
  )

  assert_array_almost_equal(wkernel_data.cl, Cu, decimal=7)
  assert_array_almost_equal(wkernel_data.cm, Cv, decimal=7)
  assert_array_almost_equal(
    wkernel_data.w_values, ddf_wkernel_data["WValues"], decimal=7
  )
  assert_array_almost_equal(
    wkernel_data.w_values.max(), ddf_wkernel_data["WMax"], decimal=7
  )
  assert_array_almost_equal(wkernel_data.oversampling, kw["OverS"], decimal=7)
  assert_array_almost_equal(wkernel_data.support, kw["Sup"], decimal=7)

  assert len(wkernel_data.w_kernels) == len(ddf_wkernel_data["WPlanes"])
  for this, ddf in zip(wkernel_data.w_kernels_ravel, ddf_wkernel_data["WPlanes"]):
    assert_array_almost_equal(this, ddf, decimal=7)

  for this, ddf in zip(
    wkernel_data.w_kernels_conj_ravel, ddf_wkernel_data["WPlanesConj"]
  ):
    assert_array_almost_equal(this, ddf, decimal=7)
