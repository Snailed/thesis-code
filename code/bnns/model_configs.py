from bnns.BNN import BNN, UCI_BNN, UCI_BNN_tanh
from bnns.CBNN import CBNN, UCI_CBNN, UCI_FFT_CBNN
from bnns.Spectral_BNN import UCI_Spectral_BNN


# Synthetic dataset models
def BNN_2_5(X, y=None, D_Y=None, sigma=None):
    return BNN(X, y, depth=2, width=5, D_Y=D_Y, sigma=sigma)

def BNN_2_100(X, y=None, D_Y=None, sigma=None):
    return BNN(X, y, depth=2, width=100, D_Y=D_Y, sigma=sigma)

def CBNN_2_5(X, y=None, D_Y=None, sigma=None):
    return CBNN(X, y, depth=2, width=5, D_Y=D_Y, sigma=sigma)

def CBNN_2_100(X, y=None, D_Y=None, sigma=None):
    return CBNN(X, y, depth=2, width=100, D_Y=D_Y, sigma=sigma)


# UCI dataset models
def UCI_BNN_2_50(X, y=None, D_Y=None, sigma=None):
    return UCI_BNN(X, y, depth=2, width=50, D_Y=D_Y)

def UCI_CBNN_2_50(X, y=None, D_Y=None, sigma=None):
    return UCI_CBNN(X, y, depth=2, width=50, D_Y=D_Y)

def UCI_BNN_2_50_tanh(X, y=None, D_Y=None, sigma=None):
    return UCI_BNN_tanh(X, y, depth=2, width=50, D_Y=D_Y)

def UCI_FFT_CBNN_2_200(X, y=None, D_Y=None, sigma=None):
    return UCI_FFT_CBNN(X, y, depth=2, width=200, D_Y=D_Y)

def UCI_CBNN_2_200(X, y=None, D_Y=None, sigma=None):
    return UCI_CBNN(X, y, depth=2, width=200, D_Y=D_Y)

def UCI_Spectral_BNN_2_50(X, y=None, D_Y=None, sigma=None):
    return UCI_Spectral_BNN(X, y, depth=2, width=50, D_Y=D_Y)