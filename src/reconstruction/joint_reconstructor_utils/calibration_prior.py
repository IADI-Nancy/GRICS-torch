"""Multiplicative calibration-image constraint used by GRICS++."""


class CalibrationPriorEncodingOperator:
    """Encode latent image p through the physical image x = C * p."""

    def __init__(self, encoding_operator, calibration_image):
        self.encoding_operator = encoding_operator
        self.calibration_image = calibration_image.flatten()
        self.device = encoding_operator.device

    def forward(self, p):
        return self.encoding_operator.forward(self.calibration_image * p)

    def adjoint(self, data):
        return self.calibration_image.conj() * self.encoding_operator.adjoint(data)

    def normal(self, p):
        return self.calibration_image.conj() * self.encoding_operator.normal(self.calibration_image * p)
