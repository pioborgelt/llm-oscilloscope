"""Opt-in experimental appraisal extension of the existing CLI runtime."""
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from .appraisal import Appraisal
from .channels import find_artifacts_root,load_cli_config
from .qwen_runtime import QwenInstrumentedRuntime,_termination_ids


def _check_runtime_versions(expected):
    requirements = {
        'torch': expected['torch'].split('+', 1)[0],
        'transformers': expected['transformers'],
        'bitsandbytes': '0.49.2',
    }
    mismatches = []
    for package, required in requirements.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = 'not installed'
        package_version = installed.split('+', 1)[0] if package == 'torch' else installed
        if package_version != required:
            mismatches.append(f'{package}: installed {installed}, required {required}')
    if mismatches:
        raise ValueError(
            'Appraisal runtime mismatch: ' + '; '.join(mismatches)
            + '. Install the tested runtime with: '
            'python -m pip install -e ".[runtime]"'
        )


def _check_torch_build(installed, required):
    # PyPI metadata omits the CUDA suffix present in torch.__version__.
    if installed != required:
        raise ValueError(
            f'Appraisal Torch build mismatch: installed {installed}, required {required}. '
            'Install the CUDA 13.0 build with: python -m pip install '
            '--force-reinstall "torch==2.11.0" '
            '--index-url https://download.pytorch.org/whl/cu130'
        )


class AppraisalRuntime(QwenInstrumentedRuntime):
    def __init__(self,**kwargs):
        if kwargs.get('quantization','8bit')!='8bit':
            raise ValueError('Appraisal requires the pinned INT8 runtime')
        self.appraisal=Appraisal(find_artifacts_root(kwargs.get('artifacts_root'))/'appraisal_experimental')
        expected=self.appraisal.manifest['expected_runtime']
        _check_runtime_versions(expected)
        import torch,transformers
        _check_torch_build(torch.__version__, expected['torch'])
        torch.set_num_threads(4)
        super().__init__(**kwargs)
        c=load_cli_config();expected=self.appraisal.manifest['expected_runtime']
        actual=dict(model=c['model_id'],revision=c['model_revision'],torch=torch.__version__,
                    transformers=transformers.__version__,device_map=self.model.hf_device_map,
                    attention=self.model.config._attn_implementation,quantization=self.model.config.quantization_config.to_dict())
        mismatch=[k for k,v in actual.items() if v!=expected.get(k)]
        if mismatch or self.subject_layer!=27:
            self.close();raise ValueError('Appraisal runtime mismatch: '+', '.join(mismatch))
    def _step_stream(self,*args,**kwargs):
        self.appraisal.reset()
        forced=kwargs.get('forced_ids')
        if forced is not None and set(forced)&_termination_ids(self.tokenizer,self.model):
            raise ValueError('Appraisal replay must not contain EOS')
        for row in super()._step_stream(*args,**kwargs):
            yield replace(row,channels={**row.channels,**self.appraisal.score(self.capture.take(27))})
    def trace_manifest(self,*args,**kwargs):
        m=super().trace_manifest(*args,**kwargs);m['appraisal']=self.appraisal.manifest
        return m
