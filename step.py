import skimage
import math
import numpy as np
import time

from numpy import linalg as LA
from skimage import io
from skimage import filters

def VectorLength(v):
  return np.sqrt(v.dot(v))


def Step(src, dest, a, b, c, tau, border):
  start_time = time.time()

  for j in range(src.shape[0] - border):

    for i in range(src.shape[1] - border):

      dest[i, j] = src[i, j] + tau * (src[i - 1, j + 1] *
                                      ((abs(b[i - 1, j + 1]) - b[i - 1, j + 1]) / 4.0 + (abs(b[i, j]) - b[i, j]) / 4.0) + src[i, j + 1] *
                                      ((c[i, j + 1] + c[i, j]) / 2.0 - (abs(b[i, j + 1]) + abs(b[i, j])) / 2.0) + src[i + 1, j + 1] *
                                      ((abs(b[i + 1, j + 1]) + b[i + 1, j + 1]) / 4.0 + (abs(b[i, j]) + b[i, j]) / 4.0) + src[i - 1, j] *
                                      ((a[i - 1, j] + a[i, j]) / 2.0 - (abs(b[i - 1, j]) + abs(b[i, j])) / 2.0) + src[i, j] *
                                      ((-(a[i - 1, j] + 2 * a[i, j] + a[i + 1, j]) / 2.0 -
                                        (abs(b[i - 1, j + 1]) - b[i - 1, j + 1] + abs(b[i + 1, j + 1]) + b[i + 1, j + 1]) / 4.0 -
                                          (abs(b[i - 1, j - 1]) + b[i - 1, j - 1] + abs(b[i + 1, j - 1]) - b[i + 1, j - 1]) / 4.0 +
                                          (abs(b[i - 1, j]) + abs(b[i + 1, j]) + abs(b[i, j - 1]) + abs(b[i, j + 1]) + 2.0 * abs(b[i, j])) / 2.0 -
                                          (c[i, j - 1] + 2.0 * c[i, j] + c[i, j + 1]) / 2.0)) + src[i + 1, j] *
                                      ((a[i + 1, j] + a[i, j]) / 2.0 - (abs(b[i + 1, j]) + abs(b[i, j])) / 2.0) + src[i - 1, j - 1] *
                                      ((abs(b[i - 1, j - 1]) + b[i - 1, j - 1]) / 4.0 + (abs(b[i, j]) + b[i, j]) / 4.0) + src[i, j - 1] *
                                      ((c[i, j - 1] + c[i, j]) / 2.0 - (abs(b[i, j - 1]) + abs(b[i, j])) / 2.0) + src[i + 1, j - 1] *
                                      ((abs(b[i + 1, j - 1]) - b[i + 1, j - 1]) / 4.0 + (abs(b[i, j]) - b[i, j]) / 4.0))

  print("---- step took %s seconds" % (time.time() - start_time))

def SliceBorder(img, slice_val):
  return img[slice_val: img.shape[0] - slice_val, slice_val:  img.shape[1] - slice_val]
