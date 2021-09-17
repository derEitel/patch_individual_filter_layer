import os
import sys
import numpy as np

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader, ConcatDataset

from scipy import stats

NITORCH_DIR = os.getcwd()
sys.path.insert(0, NITORCH_DIR)
# nitorch
from nitorch.trainer import Trainer
from nitorch.initialization import weights_init
from nitorch.metrics import *
from nitorch.loss import *


class syntheticDataset(Dataset):
    '''A dataset of 3D synthetic data '''

    def __init__(self, n_classes=2, multitask=False):

        self.multitask = multitask

        data_classes = []
        data_labels = []
        if self.multitask:
            data_means = []
        n_samples = 1000
        mean_idxs = np.random.choice(n_classes, size=n_classes, replace=False)

        for c, m in zip(range(n_classes), mean_idxs):
            # keep the mean at safe distances such that it can be easily seperable
            signs = np.random.choice([-1, 1], size=3)
            mean = np.array([5 * m, 5 * m, 5 * m], dtype=np.float) * signs
            covariance = np.array([[10 * (m + 1), 0, 0], [0, 10 * (m + 1), 0], [0, 0, 10 * (m + 1)]])
            data = np.random.multivariate_normal(mean, covariance, n_samples)
            # for labels create one-hot vectors
            # for binary classification, the label is a single value 
            if n_classes <= 2:
                labels = np.tile([c], reps=(n_samples, 1))
            else:
                labels = np.zeros((n_samples, n_classes))
                labels[np.arange(n_samples), c] = 1

            if self.multitask:
                data_means.append(np.tile(mean / 5, reps=(n_samples, 1)))

            data_classes.append(data)
            data_labels.append(labels)

        self.X = np.vstack(data_classes)
        self.y = np.vstack(data_labels)
        if self.multitask:
            self.y2 = np.vstack(data_means)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        y = self.y[idx].astype(float)
        if self.multitask:
            y = [y, self.y2[idx]]
        return self.X[idx], y


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes=2, task2_outsize=None):
        super(NeuralNet, self).__init__()
        self.out_size = 1 if n_classes <= 2 else n_classes

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, self.out_size)
        if task2_outsize:
            self.mutitask = True
            self.fc2_task2 = nn.Linear(hidden_size, task2_outsize)

    def forward(self, x):
        fc1out = self.relu(self.fc1(x))

        fc2out = self.fc2(fc1out)

        if (self.out_size > 1):
            out = F.softmax(fc2out, dim=-1)
        else:
            out = torch.sigmoid(fc2out)

        if self.mutitask:
            out_task2 = self.fc2_task2(fc1out)
            out = [out, out_task2]

        return out


if __name__ == "__main__":

    print("Starting tests ...")
    if (torch.cuda.is_available()):
        device = torch.device('cpu')
        print("Running on GPU ...")
    else:
        device = torch.device('cpu')
        print("Running on CPU ...")

    #############################################
    # Test 1 : Multi-task classifications
    #############################################
    print(
        "-----------------------------------------------\nTesting nitorch.loss.Multihead_loss() and nitorch.trainer.Trainer() : Multi-task training - Regression + classification using the same network")
    n_classes = 5
    BATCH_SIZE = 64
    EPOCHS = 20
    # the first task is to do classification of 2 classes and the second task is to predict the mean of the 3-D gaussian data
    data = syntheticDataset(n_classes=n_classes, multitask=True)

    # shuffle and split into test and train
    train_size = int(0.75 * len(data))
    train_data, val_data = random_split(data, (train_size, len(data) - train_size))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    # NETWORK
    net = NeuralNet(3, 10, n_classes=n_classes, task2_outsize=3).to(device).double()

    net.apply(weights_init)
    criterion = Multihead_loss([nn.BCELoss(), nn.MSELoss()], weights=[1, 0.01]).to(device)
    optimizer = optim.Adam(net.parameters(),
                           lr=1e-3,
                           weight_decay=1e-5)

    metrics = [[classif_accuracy], [regression_accuracy]]

    trainer = Trainer(
        net,
        criterion,
        optimizer,
        metrics=metrics,
        device=device,
        prediction_type=["classification", "regression"],
        multitask=True)

    # train model and store results
    net, report = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=EPOCHS
    )

    # a negative slope on the valdation dataset
    print(report.keys())
    y = (report["val_metrics"]["loss"])
    x = np.arange(len(y))
    slope, intercept, _, _, _ = stats.linregress(x, y)

    assert slope < 0, "Test 1-a : The loss is not reducing with training - TEST FAILED"
    print("Test 1-a 'binary classification': The loss on the valdation dataset reduces with training - TEST PASSED")

    assert (report["val_metrics"]['classif_accuracy'][0] < report["val_metrics"]['classif_accuracy'][-1]) or (
                report["val_metrics"]['classif_accuracy'][0] < report["val_metrics"]['classif_accuracy'][
            -2]), "Test 1-b : The accuracy did not improve with training - TEST FAILED"
    print("Test 1-b 'task1 classif_accuracy': metric improved with training - TEST PASSED")

    assert (report["val_metrics"]['task2 regression_accuracy'][0] < report["val_metrics"]['task2 regression_accuracy'][
        -1]) or (report["val_metrics"]['task2 regression_accuracy'][0] <
                 report["val_metrics"]['task2 regression_accuracy'][
                     -2]), "Test 1-b : task 2 accuracy did not improve with training - TEST FAILED"
    print("Test 1-c 'task2 regression_accuracy': metric improved with training - TEST PASSED")
