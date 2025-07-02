# Handwriting Neural‑Net (Python)

A minimalist, pure‑Python re‑implementation of the feed‑forward neural‑network example from Michael Nielsen’s *Neural Networks and Deep Learning*.  It recognises handwritten digits from the classic **MNIST** dataset and is intentionally small—great for learning or tinkering.

---

## Features

* Single or Multi hidden‑layer network (`784‑N‑10`, default `N = {64, 32}`)
* Mini‑batch stochastic gradient descent
* Sigmoid activations & mean‑squared‑error cost (faithful to the book)
* Quick‑start **entry point** (`entry.py`) so you can train and benchmark in one command

---

## Requirements

| Tool       | Version                  |
| ---------- | ------------------------ |
| **Python** | ≥ 3.8 (tested on 3.13.1) |
| **NumPy**  | ≥ 2.1.0                  |

Install NumPy with:

```bash
python -m pip install --upgrade "numpy>=2.1.0"
```

> **Note for Windows users** If `pip` warns that the scripts are not on PATH, just keep using `python -m pip …`.

---

## Getting the data

Data file is included but you can also - Download **`mnist.pkl.gz`** from any MNIST mirror (e.g. Yann LeCun’s site) and drop it into the project root—or into a `data/` sub‑directory if you’ve edited `mnist_loader.py` accordingly.

```
wget https://storage.googleapis.com/cvdf-datasets/mnist/mnist.pkl.gz
```

---

## Quick start

```bash
python entry.py                # uses defaults: 30 epochs, η=3.0, 30 hidden units
```

Typical console output:

```console
Loading MNIST …
Building fresh network …
Before training: 1014 / 10000

Training for 12 epoch(s) with mini‑batch size 10 and η=3.0 …

Epoch 0 : 9199 / 10000
Epoch 1 : 9335 / 10000
Epoch 2 : 9379 / 10000
…
```

To customise hyper‑parameters:

```python
from entry import run
run(epochs=12, mini_batch_size=20, eta=1.0, sizes=(784, 50, 10))
```

---

## Project structure

```
.
├── entry.py         # convenience wrapper (main script)
├── mnist_loader.py  # dataset utilities (unchanged from Nielsen repo)
├── network.py       # core neural‑network implementation
└── README.md        # you are here
```

---

## Licence

This project contains code from Michael Nielsen’s
[neural‑networks‑and‑deep‑learning](https://github.com/mnielsen/neural-networks-and-deep-learning)
repository and is distributed under the **MIT Licence**.  See `LICENSE` for full text.

© 2025 Brian Lynch — modifications and entry script.
