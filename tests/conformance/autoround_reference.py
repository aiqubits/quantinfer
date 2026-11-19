#!/usr/bin/env python3
"""Autograd reference for the signed F16 SignRound projection profile."""
import argparse
import json
from pathlib import Path

import torch


def fixture():
    i = torch.arange(96 * 3)
    weights = ((i * 17 % 127 - 63).double() / 256).reshape(96, 3)
    weights[31::32] = torch.tensor([3.25, -2.5, 4.75], dtype=torch.float64)
    i = torch.arange(23 * 96).reshape(23, 96)
    channel = i % 96
    inputs = (i * 29 % 127 - 63).double() / 64 * (channel % 13 + 1).double() / 4
    inputs[:, 31::32] /= 128
    return weights, inputs


def straight_through(source, discrete):
    return source + (discrete - source).detach()


def optimize(weights, x, bits, group, iterations):
    offsets = torch.zeros_like(weights, requires_grad=True)
    max_code = 2 ** (bits - 1) - 1
    groups = list(weights.split(group))
    sizes = [len(part) for part in groups]
    grouped_peaks = torch.stack([part.abs().amax(dim=0) for part in groups])
    base = torch.where(grouped_peaks > 0, grouped_peaks / max_code, torch.ones_like(grouped_peaks))
    factors = torch.ones_like(base, requires_grad=True)
    best_loss = torch.full((weights.shape[1],), float("inf"), dtype=torch.float64)
    best_codes = torch.zeros_like(weights)
    best_scales = torch.zeros_like(factors)
    repeat = torch.tensor(sizes)
    for i in range(iterations + 1):
        f = base * factors
        steps = straight_through(f, f.half().double().clamp_min(2 ** -24))
        expanded = steps.repeat_interleave(repeat, dim=0)
        pre = weights / expanded + offsets
        codes = straight_through(pre, pre.round()).clamp(-max_code - 1, max_code)
        loss = ((x @ (codes * expanded - weights)) ** 2).mean(dim=0)
        with torch.no_grad():
            if i == 0:
                initial = loss.detach().clone()
                initial_codes = codes.detach().clone()
                initial_scales = steps.detach().clone()
            improved = loss < best_loss
            best_codes[:, improved] = codes[:, improved]
            best_scales[:, improved] = steps[:, improved]
            best_loss[improved] = loss[improved]
        if i == iterations:
            break
        loss.sum().backward()
        step = (1 - i / iterations) / iterations
        with torch.no_grad():
            offsets.sub_(step * offsets.grad.sign()).clamp_(-0.5, 0.5)
            factors.sub_(step * factors.grad.sign()).clamp_(0, 1)
            offsets.grad.zero_()
            factors.grad.zero_()
    return {
        "bits": bits, "group_size": group, "iterations": iterations,
        "initial_mse": float(initial.mean()), "best_mse": float(best_loss.mean()),
        "improved_outputs": int((best_loss < initial).sum()),
        "changed_codes": int((best_codes != initial_codes).sum()),
        "changed_group_scales": int((best_scales != initial_scales).sum()),
        "codes_output_rows": best_codes.T.flatten().int().tolist(),
        "scales_output_rows": best_scales.repeat_interleave(repeat, dim=0)[::32].T.flatten().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    w, x = fixture()
    cases = [optimize(w, x, bits, group, iterations=32) for bits in (4, 8) for group in (32, 64)]
    actual = {"cases": cases}
    args.output.write_text(json.dumps(actual, indent=2) + "\n")
    if args.check:
        expected = json.loads(args.check.read_text())
        for a, b in zip(actual["cases"], expected["cases"], strict=True):
            for key, value in a.items():
                if key.endswith("_mse"):
                    assert abs(value - b[key]) < 1e-10, key
                else:
                    assert value == b[key], key
        print("Torch autograd and SignSGD match all four AutoRound vectors")


if __name__ == "__main__":
    main()
