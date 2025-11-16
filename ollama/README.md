# Deployment to Ollama

Deploying to Ollama provides the models via an API, which is used by frontends
such as Open WebUI to 

Note: As with the other scripts, it's currently essential to have a writable
directory at `/opt/shared`. Feel free to `rg /opt/shared` and replace all
occurances with another folder of your choosing.
-[ ] TODO: Implement an environment variable override.

## Setting up the Ollama server

```sh
# Install ollama using your package manager of choice
sudo dnf install -y epel-release
sudo dnf install -i ollama

# Activate the server, listens on localhost:11434 by default
sudo systemctl enable ollama
sudo systemctl start ollama
```

## The base model

Nothing fancy needs to be done here. Ollama hosts a corresponding GGUF model.

```sh
# Pulls the GGUF model for Qwen2.5-7B-Instruct

ollama pull qwen2.5:7b
```


## The fine-tuned model

Unfortunately Ollama and Llama.cpp don't support Safetensors models.
The training script is build using the `transformers` library which
works with Safetensors models, so conversion is needed.

```sh
# Merge the LoRA adapter with the base model. Llama.cpp's support for converting
# adapters to GGUF is not good enough at time of writing.
uv run merge_lora.py

# Use llama.cpp to convert the merged model to GGUF.
./llamacpp_convert.sh

# Use the ollama-cli to create a new model with respect to `Modelfile.qwen2-5-7b-lora`
./ollama_create.sh
```

And now the custom model should be uploaded to Ollama successfully.

