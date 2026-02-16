import torch

def fftnc(x, dims=(-4, -3, -2)):
    """
    Centered forward n-dimensional FFT with unitary normalization.
    """
    x = torch.fft.ifftshift(x, dim=dims)
    X = torch.fft.fftn(x, dim=dims, norm="ortho")
    X = torch.fft.fftshift(X, dim=dims)
    return X

def ifftnc(X, dims=(-4, -3, -2)):
    """
    Centered inverse n-dimensional FFT with unitary normalization.
    """
    X = torch.fft.ifftshift(X, dim=dims)
    x = torch.fft.ifftn(X, dim=dims, norm="ortho")
    x = torch.fft.fftshift(x, dim=dims)
    return x
