import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import recall_score, roc_curve, auc


def prepare_values(y_true, y_pred):
    """Converts the input values to numpy.ndarray.

    Parameters
    ----------
    y_true : torch.tensor
        Either a CPU or GPU tensor.
    y_pred : torch.tensor
        Either a CPU or GPU tensor.

    Returns
    -------
    y_true, y_pred : numpy.ndarray
        Numpy.ndarray of the input tensor data.

    """
    if isinstance(y_true, torch.Tensor):
        if y_true.is_cuda:
            y_true = y_true.to(torch.device("cpu"))
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        if y_pred.is_cuda:
            y_pred = y_pred.to(torch.device("cpu"))
        y_pred = y_pred.numpy()
    return y_true, y_pred


def specificity(y_true, y_pred):
    """Gets the specificity of labels and predictions.

    Parameters
    ----------
    y_true : torch.tensor
        The true labels.
    y_pred : torch.tensor
        The prediction.

    Returns
    -------
    numpy.ndarray
        The specificity.

    """
    y_true, y_pred = prepare_values(y_true, y_pred)
    return recall_score(y_true, y_pred, pos_label=0)


def sensitivity(y_true, y_pred):
    """Gets the sensitivity of labels and predictions.

    Parameters
    ----------
    y_true : torch.tensor
        The true labels.
    y_pred : torch.tensor
        The prediction.

    Returns
    -------
    numpy.ndarray
        The sensitivity.

    """
    y_true, y_pred = prepare_values(y_true, y_pred)
    return recall_score(y_true, y_pred, pos_label=1)


def binary_balanced_accuracy(y_true, y_pred):
    """Gets the binary balanced accuracy of labels and predictions.

    Parameters
    ----------
    y_true : torch.tensor
        The true labels.
    y_pred : torch.tensor
        The prediction.

    Returns
    -------
    numpy.ndarray
        The binary balanced accuracy.

    """
    y_true, y_pred = prepare_values(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    sens = sensitivity(y_true, y_pred)
    return (spec + sens) / 2


def auc_score(y_true, y_pred):
    """Gets the auc score of labels and predictions.

    Parameters
    ----------
    y_true : torch.tensor
        The true labels.
    y_pred : torch.tensor
        The prediction.

    Returns
    -------
    numpy.ndarray
        The auc score.

    """

    y_true, y_pred = prepare_values(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)


def classif_accuracy(labels, preds):
    """Gets the percentage of correct classified predictions.

    Parameters
    ----------
    labels : torch.tensor
        The true labels.
    preds : torch.tensor
        The prediction.

    Returns
    -------
    float
        Percentage of correct classified labels.

    """
    # care should be taken that both predictions and labels are class numbers and not one-hot vectors
    correct = labels.int().eq(preds.int()).sum()
    return (correct.float() / (len(labels))).item()


def regression_accuracy(labels, preds):
    """Gets the regression accuracy.

    Notes
    -----
    label values have to be in range [0,1]

    Parameters
    ----------
    labels : torch.tensor
        The true labels.
    preds : torch.tensor
        The prediction.

    Returns
    -------
    float
        Regression accuracy.

    """
    # TODO: currently, accuracy scores are sensible only if label value ranges within [0,1], 
    # convert this func to a class and let user set the range of possible values
    return (1. - F.l1_loss(preds.float(), target=labels)).item()
