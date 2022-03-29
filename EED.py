from step import *


def EEDTensor(gradientsX, gradientsY, lmbd, a, b, c):
  start_time = time.time()

  for i in range(gradientsX.shape[0]):
    for j in range(gradientsX.shape[1]):

      eigenVector1 = np.array([gradientsX[i, j], gradientsY[i, j]])
      # vector orthogonal to first eigenvector
      eigenVector2 = np.array(
          [gradientsY[i, j] * (-1), gradientsX[i, j]])
      vecLen = VectorLength(eigenVector1)

      # normalize eigenvector
      if vecLen > 0:
        eigenVector1 = eigenVector1 / vecLen
        eigenVector2 = eigenVector2 / vecLen

      # set eigen values
      eigenValue2 = 1.0

      if vecLen < 1e-5:  # todo equals with precision
        eigenValue1 = 1.0
      else:
        eigenValue1 = 1 - \
            np.exp(-3.31488 / (np.power(vecLen, 4) / np.power(lmbd, 4)))

      R = np.array([
          [eigenVector1[0], eigenVector2[0]],
          [eigenVector1[1], eigenVector2[1]]])

      E = np.array([
          [eigenValue1, 0],
          [0, eigenValue2]])

      tmpMatrix = np.matmul(np.matmul(R, E), np.transpose(R))

      a[i, j] = tmpMatrix[0, 0]
      b[i, j] = tmpMatrix[0, 1]
      c[i, j] = tmpMatrix[1, 1]
  print("---- EED tensor took %s seconds" % (time.time() - start_time))

def EED(src, num_iter, tau, lmbd, sig):
  border = 5 # can be bigger probably
  src = np.pad(src, pad_width=border, mode='edge')  # add Neumann border

  dest = np.zeros(src.shape).astype(np.float64)

  a = np.zeros(src.shape)
  b = np.zeros(src.shape)
  c = np.zeros(src.shape)

  for i in range(num_iter):
    print("iteration %d/%d" % (i + 1, num_iter))
    gaussSrc = filters.gaussian(src, sigma=sig)
    (gradX, gradY) = np.gradient(gaussSrc)

    EEDTensor(gradX, gradY, lmbd, a, b, c)

    Step(src, dest, a, b, c, tau, border)
    src = dest.copy()

  src = SliceBorder(src, border)  # remove border
  return src


