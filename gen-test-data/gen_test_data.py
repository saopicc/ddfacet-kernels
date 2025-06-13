import argparse
from contextlib import ExitStack
from functools import partial
import lzma
from itertools import product
import pickle
from typing import Any, Dict, List, Tuple
from tempfile import TemporaryDirectory
import re

import numpy as np


def parse_lm_pairs(lm_str: str) -> Tuple[float]:
  # Regular expression to capture pairs of floats within comma-separated parentheses
  pattern = re.compile(r"\((.*?),\s*(.*?)\)")

  lm = []

  for match in pattern.findall(lm_str.rstrip("[ ").lstrip(" ]")):
    lm.append((float(match[0]), float(match[1])))

  return lm


def parse_str_list(str_list: str, item_type: type) -> List[Any]:
  item_list = str_list.lstrip("[ ").rstrip(" ]").split(",")
  return [item_type(i.strip()) for i in item_list]


def create_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    description="""
                Generates a LZMA compressed pickle file containing
                members of ClassWTermModified initialised with a
                cartesian product of the supplied arguments."""
  )
  p.add_argument("--npix", default=[101], type=partial(parse_str_list, item_type=int))
  p.add_argument("--nw", default=[11], type=partial(parse_str_list, item_type=int))
  p.add_argument(
    "-f", "--freqs", default=[1e8], type=partial(parse_str_list, item_type=float)
  )
  p.add_argument(
    "-w", "--wmax", default=[30000.0], type=partial(parse_str_list, item_type=float)
  )
  p.add_argument(
    "-c", "--cell", default=[0.2], type=partial(parse_str_list, item_type=float)
  )
  p.add_argument(
    "-s", "--support", default=[15], type=partial(parse_str_list, item_type=int)
  )
  p.add_argument(
    "-o",
    "--oversampling",
    default=[11],
    type=partial(parse_str_list, item_type=int),
  )
  p.add_argument("-lm", default=[(0.0, 0.0)], type=parse_lm_pairs)
  return p


def generate_test_data(args: argparse.Namespace, stack: ExitStack) -> Dict[str, Any]:
  from DDFacet.Imager.ModCF import ClassWTermModified
  from DDFacet.Array.shared_dict import SharedDict

  assert all(isinstance(v, float) for v in args.cell)
  assert all(isinstance(v, float) for v in args.freqs)
  assert all(isinstance(v, int) for v in args.npix)
  assert all(isinstance(v, int) for v in args.nw)
  assert all(isinstance(v, float) for v in args.wmax)
  assert all(isinstance(v, int) for v in args.oversampling)
  assert all(isinstance(v, int) for v in args.support)
  assert all(
    isinstance(t, tuple) and len(t) == 2 and all(isinstance(v, float) for v in t)
    for t in args.lm
  )

  # Prepare ClassWTermModified arguments
  kv_pairs = [
    ("Npix", args.npix),
    ("Nw", args.nw),
    ("wmax", args.wmax),
    ("Freqs", args.freqs),
    ("OverS", args.oversampling),
    ("Cell", args.cell),
    ("Sup", args.support),
    ("lmShift", args.lm),
  ]

  # Decompose into key-values
  keys, values = zip(*kv_pairs)
  data_dict = {}
  data_list = []

  # Iterate over argument products
  for pvalues in product(*values):
    kw = dict(zip(keys, pvalues))
    kw["Freqs"] = np.array([kw["Freqs"]])
    tmp_dir = stack.enter_context(TemporaryDirectory())
    wterm = ClassWTermModified(cf_dict=SharedDict(tmp_dir), compute_cf=True, **kw)
    # Store the arguments that generated wterm and
    # extract members off wterm for comparison
    data_list.append(
      {
        "ClassWTermModified-kwargs": kw,
        "Cu": wterm.Cu,
        "Cv": wterm.Cv,
        "CF": wterm.CF,
        "fCF": wterm.fCF,
        "ifzfCF": wterm.ifzfCF,
        "WMax": wterm.wmax,
        "WValues": wterm.wmap,
        "WPlanes": wterm.Wplanes,
        "WPlanesConj": wterm.WplanesConj,
      }
    )

  # Store the script arguments
  data_dict["arguments"] = dict(kv_pairs)
  # Store the
  data_dict["data"] = data_list

  return data_dict


if __name__ == "__main__":
  args = create_parser().parse_args()
  with ExitStack() as stack:
    test_data = generate_test_data(args, stack)
    f = stack.enter_context(lzma.open("test-data.pickle.xz", "wb"))
    pickle.dump(test_data, f)
