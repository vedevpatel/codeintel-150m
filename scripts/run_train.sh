#!/usr/bin/env bash
# Train one base model on a GPU pod, download the export, build its interpolation variants and score them on the
# development data only (dev retrieval tasks and dev localization over the fusion-weight grid).
#
#   CONFIG=codeintel DATA=artifacts/train/groups.json.gz OUTROOT=artifacts/run CAP=240 scripts/run_train.sh granite-r2 > train.log
#
# Options: SMOKE=1 (30 steps, a few dev cases), C2C=1 (also code-to-code dev tasks), BASE_DEV=1 (also the zero-shot
# base), MINI=<n> (GradCache mini-batch), TEACHER_CACHE=<file> (precomputed teacher), NO_DEVLOC=1 (skip dev
# localization; run scripts/run_dev_localization.sh instead).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="$1"
CAP="${CAP:-280}"
DATA="${DATA:-artifacts/train/groups.json.gz}"
# Smoke test: SMOKE=1 DATA=<small packed file> runs 30 steps and scores a few dev cases of two repositories.
SMOKE="${SMOKE:-0}"
ALPHAS="${ALPHAS:-0.6,0.8}"
OUT=/content/job/run
SWEEP=/content/job/sweep
MODES="dense,bm25,hybrid@0.25,hybrid@0.5,hybrid@0.75,hybrid,hybrid@1.5,hybrid@2,hybrid@3,hybrid@4"
LIST=""
for a in ${ALPHAS//,/ }; do LIST="$LIST${LIST:+,}$SWEEP/$MODEL-a$a.yaml"; done
LIST="$LIST,$SWEEP/$MODEL-ft.yaml"
# BASE_DEV=1 also scores the zero-shot base in the retrieval step (code-to-code dev reference rows).
[ "${BASE_DEV:-0}" = "1" ] && RLIST="$MODEL,$LIST" || RLIST="$LIST"
TASKS="cosqa=data/dev/retention/cosqa,docstrings-python=data/dev/retention/docstrings-python,docstrings-typescript=data/dev/retention/docstrings-typescript"
EVAL_EXTRA=""
STEP_ENV=()
if [ "$SMOKE" = "1" ]; then
  TASKS="cosqa=data/dev/retention/cosqa"; EVAL_EXTRA=" --only-repos poetry,trpc --limit-per-repo 3"; STEP_ENV=(--env JOB_STEPS=30)
fi
PARTS="artifacts/v2/parts-$(basename "$DATA" .json.gz)"
if [ ! -d "$PARTS" ]; then mkdir -p "$PARTS"; split -b 24m -d -a 3 "$DATA" "$PARTS/train_data.json.gz.part"; fi
SHA="$(shasum -a 256 "$DATA" | cut -d' ' -f1)"
UPLOADS=()
for f in "$PARTS"/*; do UPLOADS+=(--upload "$f" "/content/job/$(basename "$f")"); done
# C2C=1 also scores the code-to-code dev tasks in the retrieval step.
DATA_TAR_ENV=()
if [ "${C2C:-0}" = "1" ]; then
  for f in artifacts/jobs/dev_code2code_parts/*; do UPLOADS+=(--upload "$f" "/content/job/$(basename "$f")"); done
  TASKS="$TASKS,code2code-tests-python=data/dev/code2code/tests-python,code2code-calls-python=data/dev/code2code/calls-python,code2code-calls-typescript=data/dev/code2code/calls-typescript"
  DATA_TAR_ENV=(--env JOB_DATA_TAR=/content/job/dev_code2code.tar.gz)
fi
OUTDIR="${OUTROOT:-artifacts/v2}/$MODEL"; [ "$SMOKE" = "1" ] && OUTDIR="$OUTDIR-smoke" || true
mkdir -p "$OUTDIR"
# NO_DEVLOC=1 skips the dev localization step (run it elsewhere with scripts/run_dev_localization.sh).
DEVLOC_STEP=(--env JOB_SCRIPT=/content/job/eval_job.py --env "JOB_ARGS=--models $LIST --variants clean --modes $MODES$EVAL_EXTRA" --exec scripts/job_launch.py)
[ "${NO_DEVLOC:-0}" = "1" ] && DEVLOC_STEP=() || true
RUN=(scripts/runpod_job.sh)
# MINI=<n>: GradCache mini-batch (memory and speed only; the gradients are the same).
MINI_ENV=(); [ -n "${MINI:-}" ] && MINI_ENV=(--env "JOB_MINI=$MINI")
# TEACHER_CACHE=<file>: distil from teacher embeddings precomputed elsewhere (uploaded in parts) instead of the base.
TC_STEPS=()
if [ -n "${TEACHER_CACHE:-}" ]; then
  TCP="$(dirname "$TEACHER_CACHE")/parts-$(basename "$TEACHER_CACHE")"
  [ -d "$TCP" ] || { mkdir -p "$TCP"; split -b 24m -d -a 3 "$TEACHER_CACHE" "$TCP/teacher_cache.pt.part"; }
  for f in "$TCP"/*; do TC_STEPS+=(--upload "$f" "/content/job/$(basename "$f")"); done
  TC_STEPS+=(--env JOB_TEACHER_CACHE=/content/job/teacher_cache.pt --env "JOB_TEACHER_CACHE_SHA256=$(shasum -a 256 "$TEACHER_CACHE" | cut -d' ' -f1)")
fi
exec "${RUN[@]}" --max-minutes "$CAP" --allow-others \
  "${UPLOADS[@]}" \
  --upload artifacts/jobs/train_job.py /content/job/train_job.py \
  --upload artifacts/jobs/sweep_job.py /content/job/sweep_job.py \
  --upload artifacts/jobs/retrieval_job.py /content/job/retrieval_job.py \
  --upload artifacts/jobs/dev_eval_job.py /content/job/eval_job.py \
  --collect "$OUT/train_log.jsonl" "$OUTDIR/train_log.jsonl" \
  ${TC_STEPS[@]+"${TC_STEPS[@]}"} \
  --env JOB_SCRIPT=/content/job/train_job.py --env "JOB_MODEL=$MODEL" --env "JOB_CONFIG=${CONFIG:-v2}" --env JOB_EXPORT=final \
  --env "JOB_TEACHER=$MODEL" \
  --env "JOB_OUT=$OUT" --env "JOB_DATA_SHA256=$SHA" ${STEP_ENV[@]+"${STEP_ENV[@]}"} ${MINI_ENV[@]+"${MINI_ENV[@]}"} \
  --exec scripts/job_launch.py \
  --download "$OUT/$MODEL-ft.tar" "$OUTDIR/$MODEL-ft.tar" \
  --env JOB_SCRIPT=/content/job/sweep_job.py --env "JOB_MODEL=$MODEL" --env "JOB_FT_TAR=$OUT/$MODEL-ft.tar" \
  --env "JOB_ALPHAS=$ALPHAS" --env "JOB_OUT=$SWEEP" \
  --exec scripts/job_launch.py \
  --env JOB_SCRIPT=/content/job/retrieval_job.py ${DATA_TAR_ENV[@]+"${DATA_TAR_ENV[@]}"} --env "JOB_ARGS=--models $RLIST --tasks $TASKS" --exec scripts/job_launch.py \
  ${DEVLOC_STEP[@]+"${DEVLOC_STEP[@]}"}
