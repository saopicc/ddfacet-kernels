from collections.abc import Sequence
import lzma
import os.path
import pickle
import warnings


def pytest_generate_tests(metafunc):
  """Hook for producing a ddf_wkernel_data fixture"""
  if "ddf_wkernel_data" not in metafunc.fixturenames:
    return

  if (test_data := metafunc.config.getoption("--ddf-test-data")) is None:
    warnings.warn(
      "--ddf-test-data wasn't supplied, "
      "the ddf_wkernel_data fixture will have an empty parametrization"
    )
    metafunc.parametrize("ddf_wkernel_data", (), scope="session")
    return

  if not os.path.exists(test_data) or not os.path.isfile(test_data):
    raise FileNotFoundError(f"{test_data} doesn't exist or is not a file")

  with lzma.open(test_data, "rb") as f:
    parametrized_data = pickle.load(f)["data"]

  assert isinstance(parametrized_data, Sequence)

  # Generate ids for each parametrization
  ids = [
    ",".join([f"{k}={v}" for k, v in p["ClassWTermModified-kwargs"].items()])
    for p in parametrized_data
  ]

  metafunc.parametrize("ddf_wkernel_data", parametrized_data, ids=ids, scope="session")
