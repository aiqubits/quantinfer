#!/usr/bin/env python3
"""Independent core-profile QIF verifier, separate from the fixed golden tool."""
import argparse
import json
import sys

from qif_reference.reader import report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--trusted-key', action='append', default=[])
    args = parser.parse_args()
    try:
        for key in args.trusted_key:
            if len(key) != 64 or len(bytes.fromhex(key)) != 32:
                raise ValueError('trusted key must be 32 hex bytes')
        result = report(args.input, {key.lower() for key in args.trusted_key})
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
        print(f'qif-reference: {error}', file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
