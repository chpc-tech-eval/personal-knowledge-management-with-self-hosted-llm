# Personal Knowledge Management with Self-Hosted LLM

## What is this?

A system providing a natural-language query and reponse interface for a test-based
knowledge base.

In plain terms: "Chat with your documents."

## How is this achieved?

1. Split the documents, generate embeddings, store them in a vector database for RAG.
2. Chunk the text, load a base model, and train a LoRA adapter on the new data. (Unsupervised.)
3. Optionally automatically trigger these with CI/CD with a local GitHub Actions server.
4. Use a basic chat interface with [`src/chat.ipynb`](src/chat.ipynb).
	
	OR lunch a Open WebUI instance, Ollama server, & Pipelines server for a pretty interface
	that does the same thing.

## Get Started

Clone this repository onto the machine you want to use for training and inference.
(If it has a lot of VRAM, it's probably that one.)

### Package Management

If `uv` isn't installed for your user, install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Use or experiment with the Jupyter Notebooks

- `cd` into a clone of this repository
- `uv run jupyter lab`
    - Take note of what port it starts on, it won’t be 8888 if another user has a JupyterLab server running. If it’s different, replace subsequent uses of `8888` with the appropriate port.
- Now open up another shell session on your computer
- `ssh -i ~/.ssh/<yourprivatekey> -L 8888:localhost:8888 <username>@<node-ip>`
- Copy the command given in the JupyterLab output with the token
    - It starts with `http://localhost:8888/lab?token=`
- Paste into a browser on your computer.
- Start chatting by running the `src/chat.ipynb` notebook. Note the step where you select the model you would like to use. A result with and without RAG is generated.

## Fine-tuning

Check out the [`src/finetuning.ipynb`](src/finetuning.ipynb) notebook.

## RAG

Check out the [`src/retrieval_db_update.ipynb`](src/retrieval_db_update.ipynb)
for the setup, and [`src/retrieval.py`](src/retrieval.py) for the runtime implementation.

## How to start a Open WebUI client

Check out [ollama/README.md](ollama/README.md) and then [webui/README.md](webui/README.md).
