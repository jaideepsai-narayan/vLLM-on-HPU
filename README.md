# Run vLLM on Habana Gaudi2 with Docker

### Prerequisites
- Habana Gaudi2 drivers installed on the host.
- Docker configured with Habana runtime.
- Hugging Face Hub token (for gated models).

### Steps

- **Clone the vLLM Repository**
  
  ```
  git clone https://github.com/vllm-project/vllm.git
  cd vllm/docker
  ```
  
- **Build the Docker Image**

  ```
  docker build . -t vllm_hpu -f Dockerfile.hpu
  ```
  
- **Run the Container**
  
  ```
  docker run -it \
  --runtime=habana \
  -e VLLM_SKIP_WARMUP=true \
  -e HUGGING_FACE_HUB_TOKEN="hf_xxxxxxxxxxxx" \
  -e HABANA_VISIBLE_DEVICES=all \
  --cap-add=sys_nice \
  --net=host \
  --rm \
  vllm_hpu \
  --model="<model>" \
  --tensor-parallel-size=<number_of_devices>
  ```

- **Test the API**

  Wait until the server logs show:
  ```
  INFO:     Application startup complete.
  ```

  Then, in a new terminal, run:

  ```
  curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "tell me about yourself ?"}
    ]
  }'
  ```

### Note

- Ensure the model is compatible with Gaudi2 (check [Habana Model Zoo](https://github.com/huggingface/optimum-habana/tree/main?tab=readme-ov-file#validated-models)).

- Adjust --tensor-parallel-size based on your HPU configuration (e.g., 8 for 8 HPUs).

- For private/gated models, replace hf_xxxxxxxxxxxx with your Hugging Face token.


