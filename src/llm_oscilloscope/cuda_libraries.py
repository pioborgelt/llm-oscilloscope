"""Load pip-provided CUDA dependencies before importing bitsandbytes."""

import ctypes
import importlib
from importlib.metadata import PackageNotFoundError, distribution
import logging
from pathlib import Path
import sys


_LIBRARY_HANDLES = {}


def _preload_nvjitlink(cuda_version):
    if sys.platform != 'linux' or not cuda_version:
        return
    major = str(cuda_version).split('.', 1)[0]
    filename = f'libnvJitLink.so.{major}'
    for package in (f'nvidia-nvjitlink-cu{major}', 'nvidia-nvjitlink'):
        try:
            installed = distribution(package)
        except PackageNotFoundError:
            continue
        for entry in installed.files or ():
            if Path(entry).name != filename:
                continue
            path = str(installed.locate_file(entry))
            if path not in _LIBRARY_HANDLES:
                try:
                    _LIBRARY_HANDLES[path] = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError as error:
                    raise RuntimeError(
                        f'Cannot load CUDA dependency {filename}: {error}. '
                        'Reinstall with: python -m pip install -e ".[runtime]"'
                    ) from error
            return
    # System CUDA installations can provide this library without a pip package.


def load_bitsandbytes(cuda_version=None):
    _preload_nvjitlink(cuda_version)
    logger = logging.getLogger('bitsandbytes.cextension')
    previously_disabled = logger.disabled
    try:
        # This module otherwise logs a traceback and silently returns a mock.
        logger.disabled = True
        module = importlib.import_module('bitsandbytes')
        native = importlib.import_module('bitsandbytes.cextension')
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            'Cannot import bitsandbytes. Reinstall with: '
            'python -m pip install -e ".[runtime]"'
        ) from error
    finally:
        logger.disabled = previously_disabled
    if cuda_version and isinstance(native.lib, native.ErrorHandlerMockBNBNativeLibrary):
        raise RuntimeError(
            f'bitsandbytes CUDA library could not load: {native.lib.error_msg}. '
            'Check the runtime installation with: llmosci doctor'
        )
    return module
