# Initialize weights
from torch.nn import init, Conv3d, BatchNorm3d, Linear


def xavier(x):
    """Wrapper for torch.nn.init.xavier method.

    Parameters
    ----------
    x : torch.tensor
        Input tensor to be initialized. See torch.nn.init.py for more information

    Returns
    -------
    torch.tensor
        Initialized tensor

    """
    return init.xavier_normal_(x)


def xavier_uniform(x):
    """Wrapper for torch.nn.init.xavier_uniform method.

    Parameters
    ----------
    x : torch.tensor
        Input tensor to be initialized. See torch.nn.init.py for more information

    Returns
    -------
    torch.tensor
        Initialized tensor

    """
    return init.xavier_uniform_(x)


def he(x):
    """Wrapper for torch.nn.init.kaiming_normal_ method.

    Parameters
    ----------
    x : torch.tensor
        Input tensor to be initialized. See torch.nn.init.py for more information

    Returns
    -------
    torch.tensor
        Initialized tensor

    """
    return init.kaiming_normal_(x)


def he_uniform(x):
    """Wrapper for torch.nn.init.kaiming_uniform_ method.

    Parameters
    ----------
    x : torch.tensor
        Input tensor to be initialized. See torch.nn.init.py for more information

    Returns
    -------
    torch.tensor
        Initialized tensor

    """
    return init.kaiming_uniform_(x)


def weights_init(m, func=he_uniform):
    """Performs weight initialization for a layer.

    Parameters
    ----------
    m
        The layer which weights should be initialized.
    func
        The function to use to initialize weights.

    Returns
    -------
    m
        Weight initialized layer.

    """
    if isinstance(m, Conv3d):
        func(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, BatchNorm3d):
        m.reset_parameters()
    elif isinstance(m, Linear):
        m.reset_parameters()
