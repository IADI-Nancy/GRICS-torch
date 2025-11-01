import torch

def fftnc(x, dims=(-4, -3, -2)):
    '''
    Forward n-dimensional FFT with normalization
    '''
    n = 1
    for d in dims:
        n *= torch.sqrt(torch.tensor(x.shape[d], dtype=x.dtype, device=x.device))

    x = torch.fft.ifftshift(x, dim=dims)
    X = torch.fft.fftn(x, dim=dims, norm="ortho")
    X = torch.fft.fftshift(X, dim=dims)

    return X / n

def ifftnc(X, dims=(-4, -3, -2)):
    '''
    Inverse n-dimensional FFT with normalization
    '''
    n = 1
    for d in dims: 
        n *= torch.sqrt(torch.tensor(X.shape[d], dtype=X.dtype, device=X.device))

    X = torch.fft.ifftshift(X, dim=dims)
    x = torch.fft.ifftn(X, dim=dims, norm="ortho")
    x = torch.fft.fftshift(x, dim=dims)
    return x * n