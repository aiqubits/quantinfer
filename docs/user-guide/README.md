# User guide

- [Qwen3-0.6B CPU 单机试运行 Runbook](qwen3-production-runbook.md)

## Model packages and profiles

```sh
quantinfer verify-package --input models/smol --json
quantinfer list-profiles --input models/smol --json
```

`verify-package` streams the SHA-256 check for every file declared by
the package manifest. `list-profiles` verifies the profile files and
checks each one against the declared execution variant without loading
model weights. Use a listed profile with `generate --inference-profile NAME`.
