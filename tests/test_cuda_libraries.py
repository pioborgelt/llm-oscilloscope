import logging
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from llm_oscilloscope.cuda_libraries import _LIBRARY_HANDLES, _preload_nvjitlink, load_bitsandbytes


class CudaLibraryTests(unittest.TestCase):
    def setUp(self):
        _LIBRARY_HANDLES.clear()

    def tearDown(self):
        _LIBRARY_HANDLES.clear()

    def test_preloads_both_nvidia_wheel_layouts_once(self):
        for layout in ('nvidia/cu13/lib', 'nvidia/nvjitlink/lib'):
            with self.subTest(layout=layout):
                _LIBRARY_HANDLES.clear()
                entry = Path(layout) / 'libnvJitLink.so.13'
                installed = SimpleNamespace(files=[entry], locate_file=lambda f: Path('/venv') / f)
                with patch('llm_oscilloscope.cuda_libraries.sys.platform', 'linux'), \
                     patch('llm_oscilloscope.cuda_libraries.distribution', return_value=installed), \
                     patch('llm_oscilloscope.cuda_libraries.ctypes.CDLL') as load:
                    _preload_nvjitlink('13.0')
                    _preload_nvjitlink('13.0')
                load.assert_called_once()
                self.assertEqual(load.call_args.args[0], str(Path('/venv') / entry))

    def test_missing_native_library_is_reported_without_upstream_traceback(self):
        class NativeError:
            error_msg = 'libnvJitLink.so.13: cannot open shared object file'
        native = SimpleNamespace(lib=NativeError(), ErrorHandlerMockBNBNativeLibrary=NativeError)
        logger = logging.getLogger('bitsandbytes.cextension')
        previous = logger.disabled
        def importing(name):
            self.assertTrue(logger.disabled)
            return native if name.endswith('.cextension') else SimpleNamespace()
        with patch('llm_oscilloscope.cuda_libraries._preload_nvjitlink'), \
             patch('llm_oscilloscope.cuda_libraries.importlib.import_module', side_effect=importing):
            with self.assertRaisesRegex(RuntimeError, 'libnvJitLink.so.13') as error:
                load_bitsandbytes('13.0')
        self.assertNotIn('\n', str(error.exception))
        self.assertIn('llmosci doctor', str(error.exception))
        self.assertEqual(logger.disabled, previous)

    def test_cpu_and_other_platforms_do_not_preload_linux_libraries(self):
        with patch('llm_oscilloscope.cuda_libraries.distribution') as find:
            _preload_nvjitlink(None)
            with patch('llm_oscilloscope.cuda_libraries.sys.platform', 'win32'):
                _preload_nvjitlink('13.0')
        find.assert_not_called()
