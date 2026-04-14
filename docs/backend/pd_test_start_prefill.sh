#!/bin/sh
set -eu

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x /opt/conda310/bin/python ]; then
    PYTHON_BIN=/opt/conda310/bin/python
  elif command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python3.10)
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python3)
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python)
  else
    PYTHON_BIN=
  fi
fi

CHECKPOINT_PATH=${CHECKPOINT_PATH:-/root/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct}
TOKENIZER_PATH=${TOKENIZER_PATH:-}
MODEL_TYPE=${MODEL_TYPE:-qwen_2}
PREFILL_HOST=${PREFILL_HOST:-127.0.0.1}
PREFILL_PORT=${PREFILL_PORT:-8090}
DECODE_HOST=${DECODE_HOST:-127.0.0.1}
DECODE_PORT=${DECODE_PORT:-27001}
USE_LOCAL=${USE_LOCAL:-1}
RDMA_CONNECT_RETRY_TIMES=${RDMA_CONNECT_RETRY_TIMES:-5000}

if [ "${USE_LOCAL}" = "1" ] && [ -n "${MODEL_SERVICE_CONFIG:-}" ]; then
  if printf '%s' "${MODEL_SERVICE_CONFIG}" | grep -q '"services"'; then
    echo "Detected legacy MODEL_SERVICE_CONFIG, replacing it with local single-service route config" >&2
    unset MODEL_SERVICE_CONFIG
  fi
fi

if [ "${USE_LOCAL}" = "1" ] && [ -z "${MODEL_SERVICE_CONFIG:-}" ]; then
  MODEL_SERVICE_CONFIG=$(printf '%s' '{"service_id":"pd_test.service","use_local":true,"role_endpoints":[{"group":"default","prefill_endpoint":{"type":"SpecifiedIpPortList","address":"'"${PREFILL_HOST}:${PREFILL_PORT}"'","protocol":"http","path":"/"},"decode_endpoint":{"type":"SpecifiedIpPortList","address":"'"${DECODE_HOST}:${DECODE_PORT}"'","protocol":"http","path":"/"}}]}')
fi

if [ -z "${PYTHON_BIN}" ]; then
  echo "python3/python not found in current shell environment" >&2
  exit 1
fi

PYTHON_VERSION=$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "${PYTHON_VERSION}" != "3.10" ]; then
  echo "RTP-LLM PD test requires Python 3.10, but current interpreter is ${PYTHON_VERSION}: ${PYTHON_BIN}" >&2
  echo "You can override explicitly, for example: PYTHON_BIN=/opt/conda310/bin/python sh docs/backend/pd_test_start_prefill.sh" >&2
  exit 1
fi

export RDMA_CONNECT_RETRY_TIMES
export MODEL_SERVICE_CONFIG

echo "Starting PREFILL node with:"
echo "  ${PYTHON_BIN} -m rtp_llm.start_server --checkpoint_path=${CHECKPOINT_PATH} --model_type=${MODEL_TYPE} --role_type=PREFILL --start_port=${PREFILL_PORT} --use_local=${USE_LOCAL} --remote_rpc_server_ip=${DECODE_HOST}:${DECODE_PORT}${TOKENIZER_PATH:+ --tokenizer_path=${TOKENIZER_PATH}}"

if [ -n "${TOKENIZER_PATH}" ]; then
  exec "${PYTHON_BIN}" -m rtp_llm.start_server \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --model_type="${MODEL_TYPE}" \
    --role_type=PREFILL \
    --start_port="${PREFILL_PORT}" \
    --use_local="${USE_LOCAL}" \
    --remote_rpc_server_ip="${DECODE_HOST}:${DECODE_PORT}" \
    --tokenizer_path="${TOKENIZER_PATH}"
fi

exec "${PYTHON_BIN}" -m rtp_llm.start_server \
  --checkpoint_path="${CHECKPOINT_PATH}" \
  --model_type="${MODEL_TYPE}" \
  --role_type=PREFILL \
  --start_port="${PREFILL_PORT}" \
  --use_local="${USE_LOCAL}" \
  --remote_rpc_server_ip="${DECODE_HOST}:${DECODE_PORT}"