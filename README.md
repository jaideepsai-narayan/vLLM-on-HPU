# Run vLLM on Habana Gaudi2 with Docker (Online serving)

### Prerequisites
- Habana Gaudi2 drivers installed on the host.

<img src="https://github.com/user-attachments/assets/fec852b7-cb72-4233-bb6b-d8294ce142b8" width="500"/>
  
- Docker configured with Habana runtime.
- Hugging Face Hub token (for gated models).

### Steps

- **Clone the vLLM Repository**
  
  ```
  git clone https://github.com/HabanaAI/vllm-fork.git
  cd vllm
  ```
  
- **Build the Docker Image**

  ```
  docker build . -t vllm_hpu -f ./docker/Dockerfile.hpu
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
  
  <img src="https://github.com/user-attachments/assets/bf397d7c-413a-4a27-90f1-d9a5dec11002" width="500"/>

  Wait until the server logs show:
  ```
  INFO:     Application startup complete.
  ```

  Then, in a new terminal, run:

  ```
  curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
  ```

  <img src="https://github.com/user-attachments/assets/abcbef50-055e-481b-ae3b-68770b57d949" width="900"/>


### Note

- Ensure the model is compatible with Gaudi2 (check [Habana Model Zoo](https://github.com/huggingface/optimum-habana/tree/main?tab=readme-ov-file#validated-models)).

- Adjust --tensor-parallel-size based on your HPU configuration (e.g., 8 for 8 HPUs).

- For private/gated models, replace hf_xxxxxxxxxxxx with your Hugging Face token.
  
- [Offline serving](https://docs.vllm.ai/en/stable/serving/offline_inference.html ) 
