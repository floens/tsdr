class SDRException(Exception):
    """Base exception for all SDR-related errors."""

    pass


class DeviceError(SDRException):
    """Hardware or device communication errors."""

    pass


class ConfigurationError(SDRException):
    """Invalid configuration parameters."""

    pass
