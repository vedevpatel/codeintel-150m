#!/usr/bin/env bash
# Dev localization for a model trained by scripts/run_train.sh with NO_DEVLOC=1: rebuild its interpolation variants
# from the downloaded export and score them on the dev localization cases over the fusion-weight grid. Dev data only.
#
#   OUTROOT=artifacts/run CAP=60 scripts/run_dev_localization.sh granite-r2 > devloc.log
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="$1"; CAP="${CAP:-90}"; ROOTDIR="${OUTROOT:-artifacts/run}"; ALPHAS="${ALPHAS:-0.6,0.8}"
RUN=(scripts/runpod_job.sh)
TAR="$ROOTDIR/$MODEL/$MODEL-ft.tar"; PARTS="$ROOTDIR/parts-$MODEL-ft"
if [ ! -d "$PARTS" ]; then mkdir -p "$PARTS"; split -b 24m -d -a 3 "$TAR" "$PARTS/$MODEL-ft.tar.part"; fi
SHA="$(shasum -a 256 "$TAR" | cut -d' ' -f1)"
SWEEP=/content/job/sweep
MODES="dense,bm25,hybrid@0.25,hybrid@0.5,hybrid@0.75,hybrid,hybrid@1.5,hybrid@2,hybrid@3,hybrid@4"
LIST=""; for a in ${ALPHAS//,/ }; do LIST="$LIST${LIST:+,}$SWEEP/$MODEL-a$a.yaml"; done; LIST="$LIST,$SWEEP/$MODEL-ft.yaml"
UPLOADS=(); for f in "$PARTS"/*; do UPLOADS+=(--upload "$f" "/content/job/$(basename "$f")"); done
exec "${RUN[@]}" --max-minutes "$CAP" --allow-others "${UPLOADS[@]}" \
  --upload artifacts/jobs/sweep_job.py /content/job/sweep_job.py \
  --upload artifacts/jobs/dev_eval_job.py /content/job/eval_job.py \
  --env JOB_SCRIPT=/content/job/sweep_job.py --env "JOB_MODEL=$MODEL" --env "JOB_FT_SHA256=$SHA" --env "JOB_ALPHAS=$ALPHAS" \
  --env "JOB_OUT=$SWEEP" --exec scripts/job_launch.py \
  --env JOB_SCRIPT=/content/job/eval_job.py --env "JOB_ARGS=--models $LIST --variants clean --modes $MODES" --exec scripts/job_launch.py
