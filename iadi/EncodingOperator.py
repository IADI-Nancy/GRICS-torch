class EncodingOperator:
    """
    MRI encoding operator.

    Methods:
    - __init__(sigmas, masks=None, image_shape=None, device=None)
    - E(x)   : forward operator (image -> k-space)
    - Eh(y)  : adjoint operator (k-space -> image)
    """

    def __init__(self, data, device=None):
        self.data = data
        self.device = device

    def E(self, image, data):
        """
        Apply forward encoding: multiply image by coil sensitivities, FFT -> k-space,
        and apply mask if provided.

        x: real/complex torch tensor with shape (Nx, Ny, Nz) or (Nx, Ny, Nz, 1)
        returns: k-space tensor with shape (Nx, Ny, Nz, nCha)
        """

        

    def Eh(self, kspace, data):
        """
        Apply adjoint (Hermitian) encoding: mask -> iFFT -> multiply by conj(sens) and sum over coils.

        y: k-space tensor shape (Nx, Ny, Nz, nCha)
        returns: image tensor shape (Nx, Ny, Nz) (complex)
        """
        
