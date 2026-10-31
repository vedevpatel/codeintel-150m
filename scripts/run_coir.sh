#!/usr/bin/env bash
# The six CoIR tasks for one trained model (OUTROOT/<model>/<model>-ft.tar), dense retrieval.
#
#   OUTROOT=artifacts/run CAP=90 scripts/run_coir.sh granite-r2 > coir.log
#
# ALPHA=<a> scores an interpolated variant; COIR_TASKS=name=dir,... scores a subset.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=(scripts/runpod_job.sh)
MODEL="$1"; CAP="${CAP:-120}"; ROOTDIR="${OUTROOT:-artifacts/run}"
# ALPHA=<a> scores the interpolated model (alpha < 1) built by the sweep step instead of the fine-tuned one.
ALPHA="${ALPHA:-1}"; if [ "$ALPHA" = "1" ] || [ "$ALPHA" = "1.0" ]; then NAME="$MODEL-ft"; ALPHAS_ENV=0.5; else NAME="$MODEL-a$ALPHA"; ALPHAS_ENV="$ALPHA"; fi
TAR="$ROOTDIR/$MODEL/$MODEL-ft.tar"; PARTS="$ROOTDIR/parts-$MODEL-ft"
if [ ! -d "$PARTS" ]; then mkdir -p "$PARTS"; split -b 24m -d -a 3 "$TAR" "$PARTS/$MODEL-ft.tar.part"; fi
SHA="$(shasum -a 256 "$TAR" | cut -d' ' -f1)"
# COIR_TASKS=name=dir,... scores a subset (e.g. tasks a lost session never reported).
COIR="${COIR_TASKS:-coir-apps=data/eval/coir/apps,coir-stackoverflow-qa=data/eval/coir/stackoverflow-qa,coir-csn-python=data/eval/coir/csn-native-python,coir-csn-javascript=data/eval/coir/csn-native-javascript,coir-csn-ccr-python=data/eval/coir/csn-ccr-python,coir-csn-ccr-javascript=data/eval/coir/csn-ccr-javascript}"
UPLOADS=()
for f in "$PARTS"/* artifacts/jobs/public_parts/*; do UPLOADS+=(--upload "$f" "/content/job/$(basename "$f")"); done
exec "${RUN[@]}" --max-minutes "$CAP" --allow-others "${UPLOADS[@]}" \
  --upload artifacts/jobs/sweep_job.py /content/job/sweep_job.py --upload artifacts/jobs/retrieval_job.py /content/job/retrieval_job.py \
  --env JOB_SCRIPT=/content/job/sweep_job.py --env "JOB_MODEL=$MODEL" --env "JOB_FT_SHA256=$SHA" --env "JOB_ALPHAS=$ALPHAS_ENV" --exec scripts/job_launch.py \
  --env JOB_SCRIPT=/content/job/retrieval_job.py --env JOB_DATA_TAR=/content/job/public_data.tar.gz \
  --env "JOB_ARGS=--models /content/job/sweep/$NAME.yaml --tasks $COIR --modes dense,bm25,hybrid" --exec scripts/job_launch.py
