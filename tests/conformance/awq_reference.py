#!/usr/bin/env python3
"""Independent Torch check for the native signed F16 AWQ profile.

Use direct token-output products for both searches. The Rust clipping path
uses an activation Gram matrix. The native per-group code grid is intentional.
"""
import argparse
import json
from pathlib import Path

import torch


def fixture():
    i = torch.arange(96 * 3)
    weights = ((i * 17 % 127 - 63).float() / 256).reshape(96, 3)
    weights[31::32] = torch.tensor([3.25, -2.5, 4.75])
    i = torch.arange(23 * 96).reshape(23, 96)
    channel = i % 96
    inputs = (i * 29 % 127 - 63).float() / 64
    inputs *= (channel % 13 + 1).float() / 4
    inputs[:, 31::32] /= 128
    return weights, inputs


def grid(weights, bits):
    peak = weights.abs().amax(dim=0)
    largest = 2 ** (bits - 1) - 1
    scale = (peak.double() / largest).half().float().clamp_min(2 ** -24)
    scale = torch.where(peak == 0, torch.ones_like(scale), scale)
    code = (weights.double() / scale.double()).round().clamp(-largest - 1, largest)
    return code.int(), scale


def quantize(weights, bits, group_size):
    parts = [grid(group, bits) for group in weights.split(group_size)]
    codes = torch.cat([c for c, _ in parts])
    scales = torch.stack([s for _, s in parts]).repeat_interleave(group_size, dim=0)[:len(weights)]
    return codes, scales


def case(weights, x, bits, group_size):
    target = x.double() @ weights.double()
    averages = x.double().abs().mean(dim=0)
    losses, candidates = [], []
    for index in range(20):
        scale = averages.pow(index / 20).clamp_min(1e-4)
        scale = (scale / (scale.min() * scale.max()).sqrt()).float()
        codes, steps = quantize(weights * scale[:, None], bits, group_size)
        effective = (codes.float() * steps) / scale[:, None]
        losses.append(float(((x.double() @ effective.double() - target) ** 2).mean()))
        candidates.append(scale)
    best = min(range(len(losses)), key=losses.__getitem__)
    scale = candidates[best]
    scaled = weights * scale[:, None]
    inputs = x / scale
    chosen, errors_before, errors_after = [], 0.0, 0.0
    clipped = []
    for group, observations in zip(scaled.split(group_size), inputs.split(group_size, dim=1)):
        peak = group.abs().amax(dim=0)
        reference = observations.double() @ group.double()
        bounds, errors = [], []
        for step in range(10):
            bound = peak * float(torch.tensor(1.) - torch.tensor(step, dtype=torch.float32) / 20)
            candidate = group.clamp(-bound, bound)
            code, grid_scale = grid(candidate, bits)
            decoded = code.float() * grid_scale
            error = ((observations.double() @ decoded.double() - reference) ** 2).mean(dim=0)
            bounds.append(bound)
            errors.append(error)
        errors = torch.stack(errors)
        selected = errors.argmin(dim=0)
        bound = torch.stack(bounds)[selected, torch.arange(group.shape[1])]
        errors_before += float(errors[0].sum())
        errors_after += float(errors[selected, torch.arange(group.shape[1])].sum())
        chosen.append(bound)
        clipped.append(group.clamp(-bound, bound))
    codes, scales = quantize(torch.cat(clipped), bits, group_size)
    return {
        "bits": bits, "group_size": group_size, "best_ratio": best / 20,
        "candidate_losses": losses, "channel_scales": scale.tolist(),
        "clipping_bounds": torch.stack(chosen).T.flatten().tolist(),
        "group_loss_before": errors_before, "group_loss_after": errors_after,
        "scales_output_rows": scales[::32].T.flatten().tolist(),
        "codes_output_rows": codes.T.flatten().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    weights, x = fixture()
    actual = {"cases": [case(weights, x, bits, group) for bits in (4, 8) for group in (32, 64)]}
    args.output.write_text(json.dumps(actual, indent=2) + "\n")
    if args.check:
        expected = json.loads(args.check.read_text())
        for a, b in zip(actual["cases"], expected["cases"], strict=True):
            for key, value in a.items():
                if key == "candidate_losses":
                    assert all(abs(x - y) < 1e-10 for x, y in zip(value, b[key], strict=True)), key
                elif key in ("group_loss_before", "group_loss_after"):
                    assert abs(value - b[key]) < 1e-10, key
                else:
                    assert value == b[key], key
        print("Torch AWQ scale search, clipping and native codes match all four vectors")


if __name__ == "__main__":
    main()
