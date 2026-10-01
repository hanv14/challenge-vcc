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
# configs/server.yaml), PYTHON (default python). Run from the repo root,
# after `source scripts/server_env.sh`. Needs only cp, no rsync.
#
# The parent's predictions, submission and sanity reports are not copied;
# everything else is (config, priors, checkpoints, policy, phase3, rehearsal),
# because the four stages below read it. A lookup table built by an earlier
# probe (runs/lookup_cache) is reused when present: the predict stage checks
# its key and rebuilds it if anything it depends on differs.
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
run=runs/$name

[ -f "$parent/phase3_policy.json" ] || {
    echo "probe: $parent has no phase3_policy.json; is PARENT a finished run?" >&2
    exit 1
}
mkdir -p "$run"
for entry in "$parent"/* "$parent"/.stages; do
    [ -e "$entry" ] || continue
    case $(basename "$entry") in
        predictions|submission|sanity) continue ;;
    esac
    cp -r "$entry" "$run"/
done
if [ -d runs/lookup_cache ]; then
    mkdir -p "$run/sources"
    cp runs/lookup_cache/* "$run/sources/"
fi

{
    echo "probe $name: parent $parent, seed $seed, threshold $threshold, effect $effect, shared $shared, extra: $*"
    for stage in predict sanity validate package; do
        nice -n 10 ionice -c3 "$python" -m vccp "$stage" --config "$config" --run-name "$name" \
            --seed "$seed" --force --allow-warnings \
            --set predict.override_threshold="$threshold" \
            --set predict.override_scale="$effect" \
            --set predict.override_shared_scale="$shared" "$@"
    done
} 2>&1 | tee "$run/probe.log"

if [ -d "$run/sources" ] && [ ! -d runs/lookup_cache ]; then
    mkdir -p runs/lookup_cache
    cp "$run"/sources/* runs/lookup_cache/
fi
ls -l "$run/submission/"
