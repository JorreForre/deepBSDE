#!/usr/bin/env python3
import torch
import numpy as np
from sys import argv

from libs import equations, model, settings

if __name__ == "__main__":
    if len(argv) != 2:
        print(f"\nusage: {argv[0]} <EQUATION>")
        exit()

    # Set default tensor type
    torch.set_default_dtype(settings.DTYPE)

    eq_name = argv[1]
    if not hasattr(equations, eq_name):
        print(f"Unknown equation '{eq_name}'. Available options: {', '.join([n for n in dir(equations) if not n.startswith('_')])}")
        exit(1)

    eq = getattr(equations, eq_name)()
    model = model.DeepBSDE(eq, log_name=eq_name)
    model.train()
