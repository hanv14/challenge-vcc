#!/usr/bin/env bash
# One paired leaderboard probe (REDESIGN.md §5): copy a finished parent run,
# change only the generator setting (and any extra --set), and rebuild the
# submission. Writes runs/<name>/ and runs/<name>/probe.log.
#
#   scripts/probe.sh NAME SEED THRESHOLD EFFECT SHARED [--set key=value ...]
#
#   scripts/probe.sh probe_p1 2 0.05 0.01 2.0
#   scripts/probe.sh probe_l1 2 0.05 0.01 2.0 --set predict.lookup_scale=0.5
#
# Environment: PARENT (default runs/server_shared), CONFIG (default
# configs/server.yaml), PYTHON (default python), STAGES (default
# "predict sanity validate package"), LOOKUP_CACHE (default runs/lookup_cache).
# Run from the repo root, after `source scripts/server_env.sh`. Needs only cp.
#
# The final round (FINAL_ROUND.md) is the same probe on the round's data, with
# `phase3` first so every new context gets its adapter, and its own cache:
#
#   CONFIG=configs/server_final.yaml STAGES="phase3 predict sanity validate package" \
#   LOOKUP_CACHE=runs/lookup_cache_final scripts/probe.sh final_l8 2 0.05 0.01 0.5 \
#       --set predict.lookup_scale=1.0 --set predict.lookup_shrinkage=none
#
# The parent's predictions, submission and sanity reports are not copied;
# everything else is (config, priors, checkpoints, policy, phase3, rehearsal),
# because the four stages below read it. A lookup table (and a low-rank basis)
# built by an earlier probe (runs/lookup_cache) is reused when present: the
# predict stage checks its key and rebuilds it if anything it depends on
# differs, or if a denoiser needs per-entry z that the cached table lacks.
set -euo pipefail

if [ $# -lt 5 ]; then
    sed -n '2,12p' "$0" >&2
    exit 2
fi
name=$1 seed=$2 threshold=$3 effect=$4 shared=$5
shift 5
parent=${PARENT:-runs/server_shared}
config=${CONFIG:-configs/server.yaml}
python=${PYTHON:-python}
stages=${STAGES:-predict sanity validate package}
cache=${LOOKUP_CACHE:-runs/lookup_cache}
run=runs/$name

[ -f "$parent/phase3_policy.json" ] || {
    echo "probe: $parent has no phase3_policy.json; is PARENT a finished run?" >&2
    exit 1
}
mkdir -p "$run"
for entry in "$parent"/* "$parent"/.stages; do
    [ -e "$entry" ] || continue
    case $(basename "$entry") in
        predictions|submission|sanity|sources) continue ;;
        # When this probe adapts its own contexts, the parent's per-context
        # records would only sit beside them, describing other contexts.
        phase3) case " $stages " in *" phase3 "*) continue ;; esac ;;
    esac
    cp -r "$entry" "$run"/
done
if [ -d "$cache" ]; then
    mkdir -p "$run/sources"
    cp "$cache"/* "$run/sources/"
fi

{
    echo "probe $name: parent $parent, config $config, stages: $stages, seed $seed, threshold $threshold, effect $effect, shared $shared, extra: $*"
    for stage in $stages; do
        nice -n 10 ionice -c3 "$python" -m vccp "$stage" --config "$config" --run-name "$name" \
            --seed "$seed" --force --allow-warnings \
            --set predict.override_threshold="$threshold" \
            --set predict.override_scale="$effect" \
            --set predict.override_shared_scale="$shared" "$@"
    done
} 2>&1 | tee "$run/probe.log"

# Refresh the cache with anything this probe built or rebuilt (a table with
# per-entry z, a low-rank basis); files it only copied in are older and skipped.
if [ -d "$run/sources" ]; then
    mkdir -p "$cache"
    cp -u "$run"/sources/* "$cache"/
fi
ls -l "$run/submission/"
