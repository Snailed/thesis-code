from bnns.BNN import BNN, UCI_BNN, UCI_BNN_tanh
from bnns.CBNN import CBNN, UCI_CBNN, UCI_FFT_CBNN, UCI_Full_CBNN, UCI_Full_FFT_CBNN, UCI_Sign_Flipped_CBNN
from bnns.Spectral_BNN import UCI_Spectral_BNN, UCI_Full_Spectral_BNN
from bnns.rasmus_bnn import model as Ola_model


# Synthetic dataset models
def BNN_2_5(X, y=None, D_Y=None, sigma=None):
    return BNN(X, y, width=5, D_Y=D_Y, sigma=sigma)

def BNN_2_100(X, y=None, D_Y=None, sigma=None):
    return BNN(X, y, width=100, D_Y=D_Y, sigma=sigma)

def CBNN_2_5(X, y=None, D_Y=None, sigma=None):
    return CBNN(X, y, width=5, D_Y=D_Y, sigma=sigma)

def CBNN_2_100(X, y=None, D_Y=None, sigma=None):
    return CBNN(X, y, width=100, D_Y=D_Y, sigma=sigma)


# UCI dataset models
def UCI_BNN_50(X, y=None, D_Y=None, sigma=None, subsample=None):
    return UCI_BNN(X, y, width=50, D_Y=D_Y, subsample=subsample)

def UCI_CBNN_50(X, y=None, D_Y=None, sigma=None, subsample=None):
    return UCI_CBNN(X, y, width=50, D_Y=D_Y, subsample=subsample)

def UCI_Full_CBNN_50(X, y=None, D_Y=None, sigma=None, subsample=None):
    return UCI_Full_CBNN(X, y, width=50, D_Y=D_Y, subsample=subsample)

def UCI_Sign_Flipped_CBNN_50(X, y=None, D_Y=None, sigma=None, subsample=None):
    return UCI_Sign_Flipped_CBNN(X, y, width=50, D_Y=D_Y, subsample=subsample)

def UCI_FFT_CBNN_50(X, y=None, D_Y=None, sigma=None, subsample=None):
    return UCI_FFT_CBNN(X, y, width=50, D_Y=D_Y, subsample=subsample)

def UCI_Full_FFT_CBNN_50(X, y=None, D_Y=None, sigma=None, subsample=None):
    return UCI_Full_FFT_CBNN(X, y, width=50, D_Y=D_Y, subsample=subsample)

def UCI_BNN_50_tanh(X, y=None, D_Y=None, sigma=None, subsample=None):
    return UCI_BNN_tanh(X, y, width=50, D_Y=D_Y, subsample=subsample)

def UCI_Spectral_BNN_50(X, y=None, D_Y=None, sigma=None, subsample=None):
    return UCI_Spectral_BNN(X, y, width=50, D_Y=D_Y, subsample=subsample)

def UCI_Full_Spectral_BNN_50(X, y=None, D_Y=None, sigma=None, subsample=None):
    return UCI_Full_Spectral_BNN(X, y, width=50, D_Y=D_Y, subsample=subsample)

def Ola_BNN(X,y=None, D_Y=None, sigma=None, subsample=None):
    return Ola_model(X, y, width=50, D_Y=D_Y, subsample=subsample)