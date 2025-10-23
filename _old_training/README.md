# Training

_Note: don't actually run this right now; we'll run out of disk space._

---

Create a virtual environment

```sh
cd training/
python3.11 -m venv venv

source venv/bin/activate.sh
```

Install the required packages

```sh
pip install -r requirements.txt
```

Run the script

```sh
python train.py
```

Currently it just asks for a prompt and generates output using LLaMA-3.2-3B-Instruct
