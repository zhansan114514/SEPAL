"""NVML ctypes compatibility patch."""

import ctypes

_orig_cdll_init = ctypes.CDLL.__init__


def apply_nvml_cdll_patch() -> None:
    """Patch ctypes.CDLL to stub missing NVML symbols for older drivers."""

    def _patched_cdll_init(self, name, *args, **kwargs):
        _orig_cdll_init(self, name, *args, **kwargs)
        if name and "libnvidia-ml" in str(name):
            try:
                self.nvmlDeviceGetNvLinkRemoteDeviceType
            except AttributeError:
                _STUB = ctypes.CFUNCTYPE(
                    ctypes.c_int,
                    ctypes.c_void_p,
                    ctypes.c_uint,
                    ctypes.c_void_p,
                )
                setattr(
                    self,
                    "nvmlDeviceGetNvLinkRemoteDeviceType",
                    _STUB(lambda dev, link, dtype: 13),
                )

    ctypes.CDLL.__init__ = _patched_cdll_init
