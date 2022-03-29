from CED import CED 
from EED import EED
from skimage import io
import numpy as np

#input: number of iterations
def RunCED(num_iter, file_name):
    test = io.imread("images/" + file_name).astype(np.float64)
    imgCED = CED(test, num_iter, tau=0.25, alpha=0.01, sig=0.1, rho=2)
    in_str = file_name.split(".")
    io.imsave("output/CED/"+ in_str[0]+ "_" + str(num_iter) + "_iter.png", imgCED)

def RunEED(num_iter, file_name):
    test = io.imread("images/" + file_name).astype(np.float64)
    imgEED = EED(test, num_iter, tau=0.25, lmbd=2, sig=2.5)
    in_str = file_name.split(".")
    io.imsave("output/EED/" + in_str[0]+ "_" + str(num_iter) + "_iter.png", imgEED)

#run with correct filename from images folder
#RunCED(20, "fingerprint.tif") 
#RunEED(20, "orka.tif")