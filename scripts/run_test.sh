#!/usr/bin/env bash
# Test scoring of one trained model, run once after its dev selection: build the chosen interpolation, score the
# held-out test repositories at the chosen fusion weight, then the code-to-code test tasks.
#
#   OUTROOT=artifacts/run CAP=120 DOWNLOAD_TAR=1 scripts/run_test.sh granite-r2 1.0 2 > test.log
#
# Score the downloaded rankings locally with scripts/collect_job_log.py and `codeintel evaluate`.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="$1"; ALPHA="$2"; WEIGHT="$3"; CAP="${CAP:-150}"
ROOTDIR="${OUTROOT:-artifacts/run}"
TAR="$ROOTDIR/$MODEL/$MODEL-ft.tar"
PARTS="$ROOTDIR/parts-$MODEL-ft"
if [ ! -d "$PARTS" ]; then mkdir -p "$PARTS"; split -b 24m -d -a 3 "$TAR" "$PARTS/$MODEL-ft.tar.part"; fi
SHA="$(shasum -a 256 "$TAR" | cut -d' ' -f1)"
UPLOADS=()
for f in "$PARTS"/*; do UPLOADS+=(--upload "$f" "/content/job/$(basename "$f")"); done
for f in artifacts/jobs/test_retrieval_parts/*; do UPLOADS+=(--upload "$f" "/content/job/$(basename "$f")"); done
if [ "$ALPHA" = "1" ] || [ "$ALPHA" = "1.0" ]; then NAME="$MODEL-ft"; ALPHAS_ENV=0.5; else NAME="$MODEL-a$ALPHA"; ALPHAS_ENV="$ALPHA"; fi
YAML="/content/job/sweep/$NAME.yaml"
TASKS="code2code-tests-python=data/eval/code2code-tests-python,code2code-calls-python=data/eval/code2code-calls-python,code2code-calls-typescript=data/eval/code2code-calls-typescript"
mkdir -p "$ROOTDIR/test/$MODEL"
# REPOS=a,b,... scores only those test repositories (a second VM sharing the work); SKIP_CODE2CODE=1 drops the
# code-to-code step.
TAIL_STEPS=(--env JOB_SCRIPT=/content/job/retrieval_job.py --env JOB_DATA_TAR=/content/job/test_retrieval_data.tar.gz
  --env "JOB_ARGS=--models $YAML --tasks $TASKS --modes dense,bm25,hybrid" --exec scripts/job_launch.py)
[ "${SKIP_CODE2CODE:-0}" = "1" ] && TAIL_STEPS=() || true
# DOWNLOAD_TAR=1 also brings the built model back (needed to aggregate the test checkpoints locally).
DOWNLOAD_STEP=()
[ "${DOWNLOAD_TAR:-0}" = "1" ] && DOWNLOAD_STEP=(--download "/content/job/sweep/$NAME.tar" "$ROOTDIR/test/$MODEL/$NAME.tar") || true
RUN=(scripts/runpod_job.sh)
exec "${RUN[@]}" --max-minutes "$CAP" --allow-others \
  "${UPLOADS[@]}" \
  --upload artifacts/jobs/sweep_job.py /content/job/sweep_job.py \
  --upload artifacts/jobs/test_eval_job.py /content/job/eval_job.py \
  --upload artifacts/jobs/retrieval_job.py /content/job/retrieval_job.py \
  --env JOB_SCRIPT=/content/job/sweep_job.py --env "JOB_MODEL=$MODEL" --env "JOB_FT_SHA256=$SHA" --env "JOB_ALPHAS=$ALPHAS_ENV" --env JOB_TAR=1 \
  --exec scripts/job_launch.py \
  ${DOWNLOAD_STEP[@]+"${DOWNLOAD_STEP[@]}"} \
  --env JOB_SCRIPT=/content/job/eval_job.py \
  --env "JOB_ARGS=--models $YAML --variants clean --modes dense,bm25,hybrid@$WEIGHT --only-repos ${REPOS:-$(cat artifacts/jobs/test_eval_repos.txt)}" \
  --exec scripts/job_launch.py \
  ${TAIL_STEPS[@]+"${TAIL_STEPS[@]}"}
