# PD 手动测试步骤

本文只讲一件事：在单机上手动启动一个 Decode 进程、一个 Prefill 进程，然后向 Prefill 发一条请求，确认 PD 分离链路能通。

这里的 PD 分离指：

- Prefill 负责处理输入和首轮计算。
- Decode 负责后续 token 生成。
- 请求入口默认在 Prefill。

本文不覆盖 `DECODE_ENTRANCE=1`、多机部署和性能压测。

## 前提

开始前先确认：

- 已经能在 Python 3.10 环境里运行 `python -m rtp_llm.start_server`
- 机器上有可用模型目录
- 当前机器推荐直接使用 conda 环境 `rtp-llm-py310`
- 最好有两张空闲 GPU，分别给 Decode 和 Prefill

如果你只是做最小验证，推荐先用小模型：

- `CHECKPOINT_PATH=/root/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct`
- `MODEL_TYPE=qwen_2`
- `REQUEST_MODEL=Qwen/Qwen2.5-0.5B-Instruct`

## 先准备一组公共变量

下面这段在两个启动终端里都要用到，建议先复制一份：

```bash
export PYTHON_BIN=/root/miniconda3/envs/rtp-llm-py310/bin/python
export CHECKPOINT_PATH=/root/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct
export MODEL_TYPE=qwen_2
export REQUEST_MODEL=Qwen/Qwen2.5-0.5B-Instruct
export PREFILL_HOST=127.0.0.1
export PREFILL_PORT=8090
export DECODE_HOST=127.0.0.1
export DECODE_PORT=27001
export USE_LOCAL=1
export RDMA_CONNECT_RETRY_TIMES=5000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MODEL_SERVICE_CONFIG='{"service_id":"pd_test.service","use_local":true,"role_endpoints":[{"group":"default","prefill_endpoint":{"type":"SpecifiedIpPortList","address":"'"${PREFILL_HOST}:${PREFILL_PORT}"'","protocol":"http","path":"/"},"decode_endpoint":{"type":"SpecifiedIpPortList","address":"'"${DECODE_HOST}:${DECODE_PORT}"'","protocol":"http","path":"/"}}]}'
```

说明：

- `MODEL_SERVICE_CONFIG` 这段不能省。单机手动测试时，Prefill 和 Decode 要靠它互相发现。
- 不是任意 Python 环境都可以，至少要满足 Python 3.10，并且已经装好 RTP-LLM 运行所需依赖。
- 如果只有一张卡，也可以先试，但更容易 OOM。
- 多卡机器建议把 Decode 和 Prefill 分开跑在两张卡上。

## 终端 1：启动 Decode

先开第一个终端，粘贴上面的公共变量，然后执行：

```bash
export CUDA_VISIBLE_DEVICES=0

${PYTHON_BIN} -m rtp_llm.start_server \
  --checkpoint_path="${CHECKPOINT_PATH}" \
  --model_type="${MODEL_TYPE}" \
  --role_type=DECODE \
  --start_port="${DECODE_PORT}" \
  --use_local="${USE_LOCAL}" \
  --remote_rpc_server_ip=${PREFILL_HOST}:${PREFILL_PORT}
```

预期：

- Decode 进程保持运行
- `DECODE_PORT` 开始监听
- 日志里没有模型路径错误、端口占用错误、`GET_HOST_FAILED`

## 终端 2：启动 Prefill

再开第二个终端，同样先粘贴公共变量，然后执行：

```bash
export CUDA_VISIBLE_DEVICES=1

${PYTHON_BIN} -m rtp_llm.start_server \
  --checkpoint_path="${CHECKPOINT_PATH}" \
  --model_type="${MODEL_TYPE}" \
  --role_type=PREFILL \
  --start_port="${PREFILL_PORT}" \
  --use_local="${USE_LOCAL}" \
  --remote_rpc_server_ip=${DECODE_HOST}:${DECODE_PORT}
```

预期：

- Prefill 进程保持运行
- `PREFILL_PORT` 开始监听
- Decode 和 Prefill 都不退出

## 终端 3：发测试请求

第三个终端执行：

```bash
export PREFILL_HOST=127.0.0.1
export PREFILL_PORT=8090
export REQUEST_MODEL=Qwen/Qwen2.5-0.5B-Instruct

curl -s http://${PREFILL_HOST}:${PREFILL_PORT}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'"${REQUEST_MODEL}"'",
    "messages": [
      {"role": "user", "content": "列出 3 个国家及其首都。"}
    ],
    "temperature": 0,
    "max_tokens": 32
  }'
```

请求一定要打到 Prefill 端口，不要打到 Decode 端口。

## 怎么判断成功

满足下面几点，就可以认为基础 PD 测试通过：

1. Decode 启动成功并一直在跑。
2. Prefill 启动成功并一直在跑。
3. 请求发到 Prefill 后能返回正常文本，而不是超时或报错。
4. 返回结果里的 `aux_info.pd_sep` 为 `true`。

## 最小排障

如果启动失败或请求不通，优先看这几项：

- Python 版本是不是 3.10。
- `CHECKPOINT_PATH` 和 `MODEL_TYPE` 是否匹配。
- Prefill 和 Decode 是否用了不同的 GPU。
- `MODEL_SERVICE_CONFIG` 是否已经导出。
- 请求是否发到了 `PREFILL_PORT`。

如果报 `8200_GET_HOST_FAILED`，通常就是本地服务发现配置没带上。

如果报 OOM，先把两个角色分到不同 GPU，再重试。

## 相关材料

如果你后面想看更完整的说明，可以继续参考：

- [docs/backend/pd_disaggregation.ipynb](docs/backend/pd_disaggregation.ipynb)
- [docs/backend/pd_entrance_transpose.md](docs/backend/pd_entrance_transpose.md)
- [docs/backend/pd_test_start_decode.sh](docs/backend/pd_test_start_decode.sh)
- [docs/backend/pd_test_start_prefill.sh](docs/backend/pd_test_start_prefill.sh)
- [docs/backend/pd_test_send_request.sh](docs/backend/pd_test_send_request.sh)