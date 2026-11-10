#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  printf 'Usage: bash benches/real-model.sh QUANTINFER_CLI MODEL_ID REVISION VARIANT MODEL_GGUF\n' >&2
  exit 2
fi

cli=$1
model=$2
revision=$3
variant=$4
gguf=$5
log=$(mktemp)
trap 'unlink "$log"' EXIT

printf 'model_id=%s\nsource_revision=%s\nvariant=%s\n' "$model" "$revision" "$variant"
printf 'gguf_sha256=%s\n' "$(sha256sum "$gguf" | cut -d ' ' -f1)"
printf 'cpu_model=%s\n' "$(awk -F ': *' '/^model name[[:space:]]*:/ && !seen++ {print $2}' /proc/cpuinfo)"
printf 'cpu_arch=%s\n' "$(uname -m)"
printf '%s\n' 'threads=1' 'build_profile=release' 'energy_j_per_token=unmeasured'
"$cli" generate --input "$gguf" --tokens 1,2,3,4,5 \
  --max-new-tokens 8 --max-sequence-length 64 --temperature 0 --profile | tee "$log"
for metric in model_load_ms prefill_ms ttft_ms profile_decode_ms profile_decode_tokens_per_sec \
  peak_rss_kib kv_cache_bytes kv_cache_capacity_bytes; do
  grep -Eq "^${metric}=[0-9]+(\\.[0-9]+)?$" "$log"
done
grep -qx 'decode_tokens=8' "$log"
grep -qx 'max_sequence_length=64' "$log"
