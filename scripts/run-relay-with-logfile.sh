#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${HOME}/.claude-relay-observability/logs"
LOG_FILE="${LOG_DIR}/relay.log"
mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}"

echo "Writing relay logs to ${LOG_FILE}"
echo "Start Promtail shipper in another terminal:"
echo "  ./scripts/run-promtail-shipper.sh hetzner-recon"

export AGENT_RELAY_LOG_LEVEL="${AGENT_RELAY_LOG_LEVEL:-INFO}"

# Preserve logs locally so Promtail can catch up after internet outages.
exec agent-relay serve 2>&1 | tee -a "${LOG_FILE}"
