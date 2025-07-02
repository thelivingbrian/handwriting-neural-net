# entry.py
"""Convenience entry point to run a simple MNIST experiment.

Usage
-----
python entry.py                 # run with default hyper‑params
python -m entry                 # same, because file is executable as a module

Within another script or REPL:

from entry import run
run(epochs=10, eta=1.0)          # override defaults
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from mnist_loader import load_data_wrapper
from network import Network  # assumes network.py lives alongside mnist_loader.py


def run(
    epochs: int = 12,
    mini_batch_size: int = 10,
    eta: float = 3.0,
    sizes: Sequence[int] = (784, 64, 32, 10),
) -> None:
    """Spin up a fresh network, report baseline accuracy, and train.

    Parameters
    ----------
    epochs
        Number of full passes over the 50 000‑image training set.
    mini_batch_size
        Images per gradient‑descent step.
    eta
        Learning‑rate (step size).
    sizes
        Tuple/list giving the layer sizes.  Default (784, 30, 10) is one hidden
        layer with 30 neurons for MNIST.
    """
    print("Loading MNIST …")
    training_data, _, test_data = load_data_wrapper()
    test_data = list(test_data)  # reuse across evaluations

    print("Building fresh network …")
    net = Network(list(sizes))

    # Baseline (un‑trained) accuracy
    baseline = net.evaluate(test_data)
    print(f"Before training: {baseline} / {len(test_data)}")

    # Training loop provided by the book's Network.SGD implementation
    print(
        f"\nTraining for {epochs} epoch(s) with mini‑batch size {mini_batch_size} and η={eta} …\n"
    )
    net.SGD(
        training_data,
        epochs=epochs,
        mini_batch_size=mini_batch_size,
        eta=eta,
        test_data=test_data,
    )


# Allow running with ``python entry.py``
if __name__ == "__main__":
    run()
