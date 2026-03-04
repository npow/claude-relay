#!/usr/bin/env bash
set -euo pipefail

TARGET_HOST="${1:-hetzner-recon}"
SSH_LOCAL_PORT="${SSH_LOCAL_PORT:-3100}"
LOG_DIR="${HOME}/.claude-relay-observability/logs"
POS_DIR="${HOME}/.claude-relay-observability/promtail"
LOG_FILE="${LOG_DIR}/relay.log"
PROMTAIL_CONTAINER="${PROMTAIL_CONTAINER:-claude-relay-promtail}"
TUNNEL_PID_FILE="${POS_DIR}/ssh-tunnel.pid"

mkdir -p "${LOG_DIR}" "${POS_DIR}"
touch "${LOG_FILE}"

cleanup() {
  if [[ -f "${TUNNEL_PID_FILE}" ]]; then
    pid="$(cat "${TUNNEL_PID_FILE}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" || true
    fi
    rm -f "${TUNNEL_PID_FILE}"
  fi
  docker rm -f "${PROMTAIL_CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

start_tunnel() {
  # Keep this loop alive; it auto-reconnects if internet/ssh drops.
  (
    while true; do
      ssh \
        -N \
        -L "${SSH_LOCAL_PORT}:127.0.0.1:3100" \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        "${TARGET_HOST}" || true
      sleep 2
    done
  ) &
  echo $! >"${TUNNEL_PID_FILE}"
}

start_tunnel

echo "Shipping ${LOG_FILE} to Loki via ssh tunnel on localhost:${SSH_LOCAL_PORT}"
echo "Target host: ${TARGET_HOST}"
echo "Stop with Ctrl+C"

docker rm -f "${PROMTAIL_CONTAINER}" >/dev/null 2>&1 || true
exec docker run --rm \
  --name "${PROMTAIL_CONTAINER}" \
  -v "$(pwd)/ops/promtail/local-promtail.yml:/etc/promtail/config.yml:ro" \
  -v "${LOG_DIR}:/var/log/claude-relay:ro" \
  -v "${POS_DIR}:/positions" \
  grafana/promtail:3.2.1 \
  -config.file=/etc/promtail/config.yml
