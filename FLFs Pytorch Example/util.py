import os

import numpy as np
import torch
import torch_dct

def logit(y):
  """The inverse of tf.nn.sigmoid()."""
  y = torch.as_tensor(y)
  return -torch.log(1. / y - 1.)


def affine_sigmoid(logits, lo=0, hi=1):
  """Maps reals to (lo, hi), where 0 maps to (lo+hi)/2."""
  if not lo < hi:
    raise ValueError('`lo` (%g) must be < `hi` (%g)' % (lo, hi))
  logits = torch.as_tensor(logits)
  lo = torch.as_tensor(lo)
  hi = torch.as_tensor(hi)
  alpha = torch.sigmoid(logits) * (hi - lo) + lo
  return alpha


def inv_affine_sigmoid(probs, lo=0, hi=1):
  """The inverse of affine_sigmoid(., lo, hi)."""
  if not lo < hi:
    raise ValueError('`lo` (%g) must be < `hi` (%g)' % (lo, hi))
  probs = torch.as_tensor(probs)
  lo = torch.as_tensor(lo)
  hi = torch.as_tensor(hi)
  logits = logit((probs - lo) / (hi - lo))
  return logits
