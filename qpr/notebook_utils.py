import matplotlib.pyplot as plt
import numpy as np


def make_ax_grid(n, ax_h=4, ax_w=6, ncols=4):
    nrows = int(np.ceil(n / ncols))
    fig_h = nrows * ax_h
    fig_w = ncols * ax_w
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))
