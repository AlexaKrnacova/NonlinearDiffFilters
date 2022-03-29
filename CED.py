from step import *


def CEDTensor(gXX, gXY, gYY, alpha):
  start_time = time.time()
  a = np.zeros(gXX.shape)
  b = np.zeros(gXX.shape)
  c = np.zeros(gXX.shape)
  for i in range(gXX.shape[0]):
    for j in range(gXX.shape[1]):

      J = np.array([
          [gXX[i, j], gXY[i, j]],
          [gXY[i, j], gYY[i, j]]])

      mi, eigvecs = LA.eig(J)
      mi1 = mi[0]
      mi2 = mi[1]
      ev1 = eigvecs[:, 0]
      ev2 = eigvecs[:, 1]
      if mi2 > mi1:
        mi1, mi2 = mi2, mi1
        ev1, ev2 = ev2, ev1

      coherence = mi1 - mi2

      # set eigen values
      mi1 = alpha

      if np.abs(coherence) < 1e-4:
        mi2 = alpha
      else:
        mi2 = alpha + (1 - alpha) * np.exp(-3.31488 / np.power(coherence, 2))

      R = np.array([
          [ev1[0], ev2[0]],
          [ev1[1], ev2[1]]])

      E = np.array([
          [mi1, 0],
          [0, mi2]])

      tmpMatrix = np.matmul(np.matmul(R, E), np.transpose(R))

      gXX[i, j] = tmpMatrix[0, 0]
      gXY[i, j] = tmpMatrix[0, 1]
      gYY[i, j] = tmpMatrix[1, 1]
  print(
      "---- CED tensor took {:.2f} seconds".format((time.time() - start_time)))

def CED(src, num_iter, tau, alpha, sig, rho):

  border = 5
  src = np.pad(src, pad_width=border, mode='edge')  # add Neumann border

  dest = np.zeros(src.shape).astype(np.float64)

  for i in range(num_iter):
    print("iteration %d/%d" % (i + 1, num_iter))

    (gradX, gradY) = np.gradient(filters.gaussian(src, sigma=sig))

    Da = filters.gaussian(np.multiply(gradX, gradX), sigma=rho)
    Db = filters.gaussian(np.multiply(gradX, gradY), sigma=rho)
    Dc = filters.gaussian(np.multiply(gradY, gradY), sigma=rho)

    CEDTensor(Da, Db, Dc, alpha)

    Step(src, dest, Da, Db, Dc, tau, border)
    src = dest.copy()

  src = SliceBorder(src, border) # remove border
  return src


