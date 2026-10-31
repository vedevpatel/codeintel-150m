#!/usr/bin/env bash
# Run an ordered list of steps on ONE fresh RunPod GPU pod and guarantee the pod is terminated.
#
#   scripts/runpod_job.sh [--gpu "NVIDIA H100 80GB HBM3"] [--max-minutes 240] [--disk 120] STEP...
#
# Steps run in the order given (the run_*.sh drivers build these lists):
#   --upload LOCAL REMOTE        copy a local file to the pod
#   --env KEY=VALUE              environment variable for the NEXT --exec (repeatable)
#   --exec SCRIPT                run a local Python script on the pod (stdout streams back)
#   --download REMOTE LOCAL      copy a pod file back (a failure stops the run)
#   --collect REMOTE LOCAL       best-effort copy back on EVERY exit path
#
# The pod is Secure Cloud, on-demand, with only 22/tcp exposed (key-only SSH, the key registered on the account) and
# no pod volume, so nothing bills once it is gone. It is terminated on success, failure, Ctrl-C, SIGTERM and at the
# wall-clock cap; termination is confirmed by a read-back. Steps run detached on the pod (nohup) and are followed by
# polling, so a dropped SSH connection does not kill a step. A second cap runs ON the pod (runpodctl stop, 10 min after
# the local cap), so a pod whose controller (this Mac) went away still stops billing for GPU time.
# Needs RUNPOD_API_KEY (read from .env if not set; never printed).
set -uo pipefail
cd "$(dirname "$0")/.."
[[ -n "${RUNPOD_API_KEY:-}" ]] || { set -a; . ./.env >/dev/null 2>&1; set +a; }
[[ -n "${RUNPOD_API_KEY:-}" ]] || { echo "RUNPOD_API_KEY is not set" >&2; exit 2; }

GPU="NVIDIA H100 80GB HBM3"
MAX_MIN=240
DISK=120
# Runpod's default "Runpod Pytorch 2.8.0" template image (runpod-torch-v280), the one hosts most likely have cached:
# 4 of 6 pods on runpod/pytorch:1.4.1-cu1281-torch2130-ubuntu2204 (template 2srslt74cv) never started their container.
IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404}"
KEY="$HOME/.ssh/runpod_codeintel"
STEPS=()
COLLECT=()
PENDING_ENV=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --max-minutes) MAX_MIN="$2"; shift 2 ;;
    --disk) DISK="$2"; shift 2 ;;
    --allow-others) shift ;;   # accepted for compatibility; every pod is independent
    --upload) STEPS+=("upload|$2|$3"); shift 3 ;;
    --env) PENDING_ENV+=("$2"); shift 2 ;;
    --exec)
      env_joined=""
      for e in ${PENDING_ENV[@]+"${PENDING_ENV[@]}"}; do env_joined+="${env_joined:+;;}$e"; done
      STEPS+=("exec|$2|$env_joined"); PENDING_ENV=(); shift 2 ;;
    --download) STEPS+=("download|$2|$3"); shift 3 ;;
    --collect) COLLECT+=("$2|$3"); shift 3 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ ${#STEPS[@]} -ge 1 ]] || { echo "usage: $0 [--gpu TYPE] [--max-minutes N] --upload L R --exec SCRIPT --download R L ..." >&2; exit 2; }

NAME="codeintel-$(date +%Y%m%d-%H%M%S)-$$"
START=$(date +%s)
CAP_SECONDS=$((MAX_MIN * 60))
POD=""
SSH_HOST=""
SSH_PORT=""
WATCH_PID=""
WORK="$(mktemp -d "/tmp/$NAME.XXXX")"
KNOWN="$WORK/known_hosts"

with_limit() { local secs="$1"; shift; perl -e 'alarm shift; exec @ARGV' "$secs" "$@"; }
log() { echo "[runpod $(date +%H:%M:%S)] $*" >&2; }
api() {  # api METHOD PATH [JSON]; the key goes through a header file so it never appears in the process list
  local hdr="$WORK/auth"; [[ -f "$hdr" ]] || { umask 077; printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY" > "$hdr"; }
  local data=(); [[ $# -ge 3 ]] && data=(-d "$3")
  curl -sS --max-time 60 -X "$1" "https://rest.runpod.io/v1$2" -H @"$hdr" -H "Content-Type: application/json" ${data[@]+"${data[@]}"}
}
pod_field() { python3 -c "import json,sys; d=json.load(sys.stdin); v=d$1; print('' if v is None else v)" 2>/dev/null; }
ssh_pod() { with_limit "${LIMIT:-120}" ssh -i "$KEY" -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KNOWN" \
            -n -o ConnectTimeout=20 -o ServerAliveInterval=30 -o BatchMode=yes -o LogLevel=ERROR "root@$SSH_HOST" "$@"; }
scp_pod() { with_limit "${LIMIT:-1800}" scp -q -i "$KEY" -P "$SSH_PORT" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KNOWN" \
            -o ConnectTimeout=20 -o BatchMode=yes -o LogLevel=ERROR "$@"; }
remaining() { echo $((CAP_SECONDS - ($(date +%s) - START))); }

terminate_pod() {  # delete a pod and confirm by read-back (404)
  local id="$1" code=""
  for _ in 1 2 3 4 5; do
    api DELETE "/pods/$id" >/dev/null 2>&1
    sleep 5
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 -H @"$WORK/auth" "https://rest.runpod.io/v1/pods/$id")
    [[ "$code" == "404" ]] && { log "pod $id terminated (read-back: 404)."; return 0; }
  done
  echo "!! COULD NOT CONFIRM POD $id IS TERMINATED - delete it in the console (https://console.runpod.io/pods)" >&2
  return 1
}

release() {
  trap - EXIT INT TERM
  [[ -n "$WATCH_PID" ]] && kill "$WATCH_PID" 2>/dev/null
  if [[ -n "$POD" ]]; then
    if [[ -n "$SSH_HOST" ]]; then
      for item in ${COLLECT[@]+"${COLLECT[@]}"}; do   # best effort, bounded, before the pod disappears
        remote="${item%%|*}"; local_path="${item##*|}"
        mkdir -p "$(dirname "$local_path")"
        LIMIT=600 scp_pod "root@$SSH_HOST:$remote" "$local_path" 2>/dev/null && log "collected $remote -> $local_path"
      done
    fi
    terminate_pod "$POD" || true
  fi
  rm -f "$WORK/auth"
  log "runtime $(( ($(date +%s) - START) / 60 )) min"
}
trap 'release' EXIT
trap 'exit 130' INT TERM

PUB="$(cat "$KEY.pub")"
BODY=$(python3 - "$NAME" "$IMAGE" "$GPU" "$DISK" "$PUB" <<'EOF'
import json, sys
name, image, gpu, disk, pub = sys.argv[1:]
print(json.dumps({
    "name": name, "imageName": image, "computeType": "GPU", "gpuTypeIds": [gpu], "gpuCount": 1,
    "cloudType": "SECURE", "interruptible": False, "containerDiskInGb": int(disk), "volumeInGb": 0,
    "ports": ["22/tcp"], "supportPublicIp": True,
    # Data centers with H100 SXM stock and a short path from the US (AP-IN-1 never answered SSH in 20 min).
    "dataCenterIds": ["US-GA-2", "US-MO-1", "CA-MTL-1", "EU-FR-1", "EUR-IS-3"],
    "dataCenterPriority": "availability",
    "env": {"PUBLIC_KEY": pub},
}))
EOF
)
create_pod() {
  log "creating $NAME ($GPU, Secure Cloud, cap ${MAX_MIN} min)"
  for try in $(seq 1 "${CREATE_TRIES:-15}"); do   # "no instances available" is common at Low stock; failed creates are free
    RESP="$(api POST /pods "$BODY")"
    POD="$(echo "$RESP" | pod_field "['id']")"
    [[ -n "$POD" ]] && break
    echo "$RESP" | grep -q "no instances currently available" || break
    log "no $GPU available (attempt $try), retrying in 60 s"; sleep 60
  done
  [[ -n "$POD" ]] || { echo "pod creation failed: $(echo "$RESP" | head -c 400)" >&2; exit 4; }
  log "pod $POD: $(echo "$RESP" | pod_field "['costPerHr']") credits/hr, data center $(echo "$RESP" | pod_field ".get('machine',{}).get('dataCenterId')")"
}

wait_ssh() {  # image pull + boot, with the state logged every 2 minutes; healthy pods answer in about 5 minutes
  local n INFO err
  for n in $(seq 1 "$((BOOT_MIN * 3))"); do
    INFO="$(api GET "/pods/$POD")"
    SSH_HOST="$(echo "$INFO" | pod_field "['publicIp']")"
    SSH_PORT="$(echo "$INFO" | pod_field "['portMappings']['22']")"
    err=""
    if [[ -n "$SSH_HOST" && -n "$SSH_PORT" ]]; then
      err="$(LIMIT=30 ssh_pod true 2>&1)" && return 0
    fi
    (( n % 6 == 0 )) && log "waiting: status=$(echo "$INFO" | pod_field "['desiredStatus']") ip=${SSH_HOST:-none} port22=${SSH_PORT:-none} ${err:+ssh: ${err:0:160}}"
    SSH_HOST=""; sleep 20
  done
  return 1
}

BOOT_MIN="${BOOT_MIN:-10}"
START=$(date +%s)   # the cap counts from the first creation, boot retries included
( sleep "$CAP_SECONDS"; log "${MAX_MIN} min cap hit, stopping."; kill -TERM $$ 2>/dev/null ) &
WATCH_PID=$!; disown "$WATCH_PID"
for boot in $(seq 1 "${BOOT_TRIES:-3}"); do
  create_pod
  wait_ssh && break
  # Some hosts never start the container (2 of the first 5 pods); replace the pod rather than wait on it.
  log "pod $POD not reachable after $BOOT_MIN min (boot attempt $boot), replacing it"
  terminate_pod "$POD" || exit 5
  POD=""
done
[[ -n "$SSH_HOST" ]] || { log "no pod became reachable over SSH"; exit 5; }
log "ssh ready: root@$SSH_HOST -p $SSH_PORT"
# Find the image's Python with torch (non-login SSH shells may not have it on PATH); install torch if there is none.
cat > "$WORK/setup.sh" <<'SETUP'
set -e
mkdir -p /content/job/_steps
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
PY=""
for c in python3 python /opt/venv/bin/python /opt/conda/bin/python /usr/local/bin/python3 /usr/bin/python3 \
         $(bash -lc 'command -v python3 python' 2>/dev/null); do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import torch' 2>/dev/null; then PY="$(command -v "$c")"; break; fi
done
if [ -z "$PY" ]; then
  PY="$(command -v python3)"
  echo "no torch in the image; installing torch (cu128) into $PY"
  PIP_BREAK_SYSTEM_PACKAGES=1 "$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
fi
"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("torch", torch.__version__, torch.cuda.get_device_name(0))'
echo "@@PY $PY"
SETUP
LIMIT=900 scp_pod "$WORK/setup.sh" "root@$SSH_HOST:/tmp/setup.sh" && LIMIT=900 ssh_pod "bash /tmp/setup.sh" > "$WORK/setup.out" 2>&1
setup_code=$?
grep -v '^@@PY ' "$WORK/setup.out" >&2
PY="$(grep '^@@PY ' "$WORK/setup.out" | tail -1 | cut -d' ' -f2)"
[[ $setup_code -eq 0 && -n "$PY" ]] || { log "pod setup failed"; exit 5; }
log "python: $PY"
# Pod-side cap: stops the pod (GPU billing ends) even if this controller disappears. The container's own pod-scoped
# key (in PID 1's environment, not in SSH sessions) is accepted by the GraphQL API only.
POD_CAP=$((CAP_SECONDS + 600))
cat > "$WORK/podcap.sh" <<PODCAP
eval "\$(tr '\\0' '\\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
[ -n "\${RUNPOD_POD_ID:-}" ] && [ -n "\${RUNPOD_API_KEY:-}" ] || { echo unavailable; exit 0; }
curl -s https://api.runpod.io/graphql -H "Content-Type: application/json" -H "Authorization: Bearer \$RUNPOD_API_KEY" \\
  -d "{\\"query\\":\\"{ pod(input:{podId:\\\\\\"\$RUNPOD_POD_ID\\\\\\"}) { id } }\\"}" | grep -q '"id"' || { echo "unavailable (key rejected)"; exit 0; }
setsid nohup sh -c "sleep $POD_CAP; curl -s https://api.runpod.io/graphql -H 'Content-Type: application/json' -H \\"Authorization: Bearer \$RUNPOD_API_KEY\\" -d '{\\"query\\":\\"mutation { podStop(input:{podId:\\\\\\"\$RUNPOD_POD_ID\\\\\\"}) { id desiredStatus } }\\"}'" >/content/job/_steps/podcap.log 2>&1 </dev/null &
echo "armed (podStop after $((POD_CAP / 60)) min)"
PODCAP
scp_pod "$WORK/podcap.sh" "root@$SSH_HOST:/tmp/podcap.sh" && ssh_pod "bash /tmp/podcap.sh" | sed 's/^/[runpod] pod-side cap: /' >&2

i=0
for step in "${STEPS[@]}"; do
  i=$((i + 1))
  kind="${step%%|*}"; rest="${step#*|}"; a="${rest%%|*}"; b="${rest#*|}"
  left=$(remaining)
  [[ $left -gt 30 ]] || { log "out of time before step: $kind $a"; exit 6; }
  case "$kind" in
    upload)
      log "upload $a -> $b ($(du -h "$a" | cut -f1))"
      ssh_pod "mkdir -p '$(dirname "$b")'" && LIMIT=$left scp_pod "$a" "root@$SSH_HOST:$b" || { log "upload failed"; exit 7; } ;;
    download)
      log "download $a -> $b"
      mkdir -p "$(dirname "$b")"
      LIMIT=$left scp_pod "root@$SSH_HOST:$a" "$b" || { log "download failed"; exit 8; } ;;
    exec)
      dir="/content/job/_steps/$i"
      envf="$WORK/env.$i"; echo "export PIP_BREAK_SYSTEM_PACKAGES=1" > "$envf"   # Ubuntu 24.04 images mark Python as system-managed
      if [[ -n "$b" ]]; then
        IFS=$'\n' read -r -d '' -a pairs <<<"${b//;;/$'\n'}" || true
        for p in "${pairs[@]}"; do [[ -n "$p" ]] && printf 'export %s=%q\n' "${p%%=*}" "${p#*=}" >> "$envf"; done
      fi
      log "exec $a"
      ssh_pod "mkdir -p $dir" && scp_pod "$envf" "root@$SSH_HOST:$dir/env.sh" && scp_pod "$a" "root@$SSH_HOST:$dir/step.py" || { log "exec setup failed"; exit 9; }
      ssh_pod "cd /content/job; setsid nohup bash -c '. $dir/env.sh; $PY -u $dir/step.py > $dir/out.log 2>&1; echo \$? > $dir/exit' >/dev/null 2>&1 </dev/null &" \
        || { log "exec launch failed"; exit 9; }
      off=0; misses=0
      while :; do   # follow the step's log; tolerate dropped connections (10 misses in a row = 5 min)
        # exit status first, then the log: a status seen before the tail means the tail is complete
        if code="$(LIMIT=60 ssh_pod "cat $dir/exit 2>/dev/null || true")" \
           && LIMIT=120 ssh_pod "tail -c +$((off + 1)) $dir/out.log 2>/dev/null" > "$WORK/chunk"; then
          misses=0
          cat "$WORK/chunk"
          off=$((off + $(wc -c < "$WORK/chunk")))
          if [[ -n "$code" ]]; then
            [[ "$code" == "0" ]] || { log "exec failed with $code"; exit 9; }
            break
          fi
        else
          misses=$((misses + 1)); log "poll failed ($misses)"
          [[ $misses -ge 10 ]] && { log "pod unreachable for 5 min, stopping."; exit 10; }
        fi
        sleep 30
      done ;;
  esac
done
log "all steps finished"
exit 0
