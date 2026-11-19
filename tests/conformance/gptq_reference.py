#!/usr/bin/env python3
"""Execute pinned upstream GPTQ on CPU to produce independent fixed vectors.

Requires the model-conformance Torch environment. Upstream sources are external
test dependencies; --upstream must contain the two exact files below. No source
is patched: only CUDA synchronization is bypassed for this CPU-only reference.
The quantizer subclass freezes F16 scales before the original GPTQ feedback.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

REVISION = '2d65066eeb06a5c9ff5184d8cebdf33662c67faf'
SOURCE_SHA256 = {
    'gptq.py': 'bb47468cc2c788577562751de414d76eeb9968ff1c6098e21153557ef3d154c7',
    'quant.py': '211528568b962b868d20fc58b55200b615f41ebd9f36b4d2d7e4a196a095ea5a',
}


def load_source(directory, name):
    path = directory / f'{name}.py'
    expected = SOURCE_SHA256[f'{name}.py']
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'{path} differs from pinned upstream {REVISION}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def generate(directory, supplied_inputs=None):
    import numpy as np
    import torch
    quant = load_source(directory, 'quant')
    gptq = load_source(directory, 'gptq')
    torch.set_num_threads(1)
    if supplied_inputs is None:
        rng = np.random.Generator(np.random.PCG64(271828))
        weights = rng.uniform(-1, 1, (3, 64)).astype(np.float32)
        independent = rng.standard_normal((96, 64)).astype(np.float32)
        activations = (independent * np.float32(.5)
                       + np.roll(independent, 1, axis=1) * np.float32(.7)
                       + independent[:, :1] * np.float32(.2))
    else:
        # Re-run the algorithm against literal fixed inputs, independent of
        # future NumPy changes to its random distribution implementations.
        weights = np.asarray(supplied_inputs['weights'], dtype=np.float32).reshape(64, 3).T.copy()
        activations = np.asarray(supplied_inputs['activations'], dtype=np.float32).reshape(96, 64)
    inputs = torch.from_numpy(activations)
    hessian = 2 * inputs.T @ inputs / len(inputs)

    class StoredGrid(quant.Quantizer):
        def find_params(self, values, weight=False):
            super().find_params(values, weight)
            self.scale = self.scale.half().float().clamp_min(2 ** -24)
            if not torch.isfinite(self.scale).all():
                raise ValueError('scale outside native F16 profile')

    cases = []
    for bits in [4, 8]:
        for group in [32, 64]:
            for activation_order in [False, True]:
                layer = torch.nn.Linear(64, 3, bias=False)
                layer.weight.data.copy_(torch.from_numpy(weights))
                engine = gptq.GPTQ(layer)
                engine.H = hessian.clone()
                engine.quantizer = StoredGrid()
                engine.quantizer.configure(bits, perchannel=True, sym=True)
                with patch('torch.cuda.synchronize', return_value=None), contextlib.redirect_stdout(sys.stderr):
                    engine.fasterquant(blocksize=16, percdamp=.01, groupsize=group,
                                       actorder=activation_order, static_groups=True)
                grids = []
                for start in range(0, 64, group):
                    grid = StoredGrid()
                    grid.configure(bits, perchannel=True, sym=True)
                    grid.find_params(torch.from_numpy(weights[:, start:start+group]), weight=True)
                    grids.extend([grid.scale.flatten()] * (min(group, 64-start) // 32))
                scales = torch.stack(grids, dim=1)
                per_input = scales.repeat_interleave(32, dim=1)
                restored = layer.weight.data
                codes = torch.round(restored / per_input) + 2 ** (bits - 1)
                rtn = per_input * (torch.clamp(torch.round(torch.from_numpy(weights) / per_input)
                                             + 2 ** (bits - 1), 0, 2 ** bits - 1)
                                    - 2 ** (bits - 1))
                target = inputs @ torch.from_numpy(weights.T.copy())
                error = lambda w: ((inputs @ w.T - target) ** 2).mean().item()
                cases.append({'bits': bits, 'group_size': group, 'activation_order': activation_order,
                              'scales_output_rows': scales.flatten().tolist(),
                              'codes_output_rows': codes.to(torch.int32).flatten().tolist(),
                              'projection_mse': error(restored), 'rtn_projection_mse': error(rtn)})
    return {'upstream_revision': REVISION, 'source_sha256': SOURCE_SHA256,
            'profile': 'static symmetric groups; F16 grid; H=2*X.T@X/token_rows; damping=0.01',
            'weights_shape': [64, 3], 'weights': weights.T.flatten().tolist(),
            'activation_shape': list(activations.shape), 'activations': activations.flatten().tolist(),
            'cases': cases}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', type=Path, help='Recompute and check the fixed input/reference fixture')
    args = parser.parse_args()
    golden = json.loads(args.check.read_text()) if args.check else None
    actual = generate(args.upstream, golden)
    args.output.write_text(json.dumps(actual, separators=(',', ':')) + '\n')
    if golden is not None:
        if len(actual['cases']) != len(golden['cases']):
            raise ValueError('reference case count changed')
        for key in actual.keys() - {'cases'}:
            if actual[key] != golden[key]:
                raise ValueError(f'reference {key} changed')
        for i, (reference, expected) in enumerate(zip(actual['cases'], golden['cases'])):
            for key, value in reference.items():
                if key.endswith('mse'):
                    valid = math.isclose(value, expected[key], rel_tol=1e-5, abs_tol=1e-8)
                else:
                    valid = value == expected[key]
                if not valid:
                    raise ValueError(f'upstream reference case {i} {key} changed')
        print(f'Pinned upstream GPTQ: {len(actual["cases"])} code/scale/quality vectors passed')
