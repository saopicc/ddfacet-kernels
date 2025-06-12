def pytest_addoption(parser):
  parser.addoption(
    "--ddf-test-data",
    action="store",
    default=None,
    help=(
      f"Disk location of pickled test data "
      f"Set to None by default, in which case "
      f"test data will be downloaded"
    ),
  )
