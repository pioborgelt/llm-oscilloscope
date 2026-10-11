"""Descriptive signal-strength colors for complete appraisal windows."""

from functools import lru_cache
import hashlib
from importlib.resources import files
import json
import math


@lru_cache(maxsize=1)
def appraisal_color_scale():
    config = json.loads(files('llm_oscilloscope').joinpath('appraisal_color_scale.json').read_text())
    low, high = config['yellow_from'], config['red_from']
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError('Invalid appraisal color limits')
    reference = files('llm_oscilloscope').joinpath(config['reference']['file']).read_bytes()
    if hashlib.sha256(reference).hexdigest() != config['reference']['sha256']:
        raise ValueError('Appraisal color reference checksum mismatch')
    return config


def appraisal_band(value, *, ready=True):
    if not ready or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 'PENDING' if not ready else 'UNAVAILABLE', 'grey62'
    scale = appraisal_color_scale()
    if value < scale['yellow_from']:
        return 'LOW', '#5CCB8A'
    if value < scale['red_from']:
        return 'MEDIUM', '#E6C35C'
    return 'HIGH', '#F07178'


def recorded_appraisal_row(rows, selected):
    """Use the first complete window for the recording's opening positions.

    This look-ahead is presentation-only. Stored readings and live scoring
    retain their original causal values.
    """
    row = rows[selected]
    if selected < 3 and len(rows) >= 4 and rows[3].get('appraisal_window_ready'):
        return dict(row, appraisal_mean4=rows[3]['appraisal_mean4'],
                    appraisal_window_ready=True)
    return row
