import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def dataset_length(data_loader):
    """Return the full length of the dataset from the DataLoader alone.

    Calling len(data_loader) only shows the number of mini-batches.
    Requires data to be located at.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        The data_loader for the data.

    Returns
    -------
    int
        The total length of the `data_loader`.

    Raises
    ------
    KeyError
        If each entry in the `data_loader` is a dictionary, key for the label is expected to be 'label'.

    """
    sample = next(iter(data_loader))
    batch_size = None

    if isinstance(sample, dict):
        try:
            if isinstance(sample["label"], torch.Tensor):
                batch_size = sample["label"].shape[0]
            else:
                # in case of sequence of inputs use first input
                batch_size = sample["label"][0].shape[0]
        except:
            KeyError("Expects key to be 'label'.")
    else:
        if isinstance(sample[1], torch.Tensor):
            batch_size = sample[1].shape[0]
        else:
            # in case of sequence of inputs use first input
            batch_size = sample[1][0].shape[0]
    return len(data_loader) * batch_size


def count_parameters(model):
    """Returns the number of adjustable parameters of the input.

    Parameters
    ----------
    model
        The model.

    Returns
    -------
    int
        The number of adjustable parameters of `model`.

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.

    Can be used for checking for possible gradient vanishing / exploding problems.

    Parameters
    ----------
    named_parameters : list
        A list of tuples. First entry is the name, second the value.

    Notes
    -----
        Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow


    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n:
            layers.append(n.replace(".weight", ""))
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    

def is_bad_grad(grad_output):
    """Checks if gradient is too big

    Parameters
    ----------
    grad_output
        The gradient you got back during back-propagation.

    Returns
    -------
    bool
        True if gradient is bad, False otherwise.

    """
    grad_output = grad_output.data
    return grad_output.ne(grad_output).any() or grad_output.gt(1e6).any()
