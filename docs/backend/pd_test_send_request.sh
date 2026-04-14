#!/bin/sh
set -eu

: "${PREFILL_HOST:=127.0.0.1}"
: "${PREFILL_PORT:=8090}"
: "${REQUEST_MODEL:=Qwen/Qwen2.5-0.5B-Instruct}"
: "${REQUEST_TEXT:=你是谁。}"
: "${TEMPERATURE:=0}"
: "${MAX_TOKENS:=64}"

curl -s "http://${PREFILL_HOST}:${PREFILL_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON
{
  "model": "${REQUEST_MODEL}",
  "messages": [
    {
      "role": "user",
      "content": "${REQUEST_TEXT}"
    }
  ],
  "temperature": ${TEMPERATURE},
  "max_tokens": ${MAX_TOKENS}
}
JSON