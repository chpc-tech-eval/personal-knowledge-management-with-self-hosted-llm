# Personal Knowledge Management with Self-Hosted LLM

## Get Started

- SSH into the GPU node.
- If `uv` isn't installed for your user, install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- `cd` into a clone of this repository
- `uv run jupyter lab`
    - Take note of what port it starts on, it won’t be 8888 if another user has a JupyterLab server running. If it’s different, replace subsequent uses of `8888` with the appropriate port.
- Now open up another shell session on your computer
- `ssh -i ~/.ssh/<yourprivatekey> -L 8888:localhost:8888 <username>@<node-ip>`
    - For Shaun it's `ssh -i ~/.ssh/oc -L 8888:localhost:8888 shaun@scc-gpu`
    - For Glen it's `ssh -i ~/.ssh/id_ed25519 -L 8888:localhost:8888 glen@196.24.241.72`
    - For Tam it's `ssh -i ~/.ssh/capstone_key -L 8888:localhost:8888 tamryn@192.24.241.72`
- Copy the command given in the JupyterLab output with the token
    - It starts with `http://localhost:8888/lab?token=`
- Paste into a browser on your computer.
- Start chatting by running the `src/chat.ipynb` notebook. Note the step where you select the model you would like to use. A result with and without RAG is generated.

## How does it work?

TODO

## How did we develop this?

TODO

## How to start a Open WebUI client

TODO?
