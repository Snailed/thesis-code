from bnns.BNN import BNN, UCI_BNN, UCI_BNN_tanh
from bnns.CBNN import CBNN, FFT_CBNN, UCI_CBNN, UCI_FFT_CBNN, UCI_Full_CBNN, UCI_Full_FFT_CBNN, UCI_Sign_Flipped_CBNN
from bnns.Spectral_BNN import UCI_Spectral_BNN, UCI_Full_Spectral_BNN, Full_Spectral_BNN, Spectral_BNN
from bnns.rasmus_bnn import model as Ola_model
from bnns.ecg.ECG_BNN import ECG_BNN, ECG_CBNN, ECG_Spectral_BNN


# Synthetic dataset models
def BNN_5(X, y=None, D_Y=None, sigma=None, subsample=None):
    return BNN(X, y, width=5, D_Y=D_Y, sigma=sigma, subsample=subsample)

def BNN_100(X, y=None, D_Y=None, sigma=None, subsample=None):
    return BNN(X, y, width=100, D_Y=D_Y, sigma=sigma, subsample=subsample)

def CircBNN_5(X, y=None, D_Y=None, sigma=None, subsample=None):
    return FFT_CBNN(X, y, width=5, D_Y=D_Y, sigma=sigma, subsample=subsample)

def CircBNN_100(X, y=None, D_Y=None, sigma=None, subsample=None):
    return FFT_CBNN(X, y, width=100, D_Y=D_Y, sigma=sigma, subsample=subsample)

def SpectralBNN_5(X, y=None, D_Y=None, sigma=None, subsample=None):
    return Spectral_BNN(X, y, width=5, D_Y=D_Y, sigma=sigma, subsample=subsample)

def SpectralBNN_100(X, y=None, D_Y=None, sigma=None, subsample=None):
    return Spectral_BNN(X, y, width=100, D_Y=D_Y, sigma=sigma, subsample=subsample)


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

def ECG_BNN_128(X, y=None, D_Y=None, sigma=None, subsample=None, prior_probs=None):
    return ECG_BNN(X, y, width=128, subsample=subsample, prior_probs=prior_probs)

def ECG_CBNN_128(X, y=None, D_Y=None, sigma=None, subsample=None, prior_probs=None):
    return ECG_CBNN(X, y, width=128, subsample=subsample, prior_probs=prior_probs)

def ECG_Spectral_BNN_128(X, y=None, D_Y=None, sigma=None, subsample=None, prior_probs=None):
    return ECG_Spectral_BNN(X, y, width=128, subsample=subsample, prior_probs=prior_probs)