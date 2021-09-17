import os
import sys
import numpy as np

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader

from scipy import stats

NITORCH_DIR = os.getcwd()
sys.path.insert(0, NITORCH_DIR)
#nitorch
from nitorch.trainer import Trainer
from nitorch.initialization import weights_init
from nitorch.metrics import *

class syntheticDataset(Dataset):
    '''A dataset of 3D synthetic data '''
    def __init__(self, n_classes=2):
        
        data_classes = []
        data_labels = []
        n_samples = 1000
        means = np.random.choice(n_classes, size=n_classes, replace=False)
        for c,m in zip(range(n_classes), means):
            # keep the mean at safe distances such that it can be easily seperable
            signs = np.random.choice([-1,1],size=3)
            mean = np.array([5*m, 5*m, 5*m])*signs
            covariance = np.array([[10*(m+1), 0, 0],[0, 10*(m+1), 0],[0, 0, 10*(m+1)]])
            data = np.random.multivariate_normal(mean, covariance, n_samples)
            # for labels create one-hot vectors
            # for binary classification, the label is a single value 
            if(n_classes<=2):
                labels = np.tile([c],reps=(n_samples,1))
            else:
                labels = np.zeros((n_samples, n_classes))
                labels[np.arange(n_samples), c]=1

            data_classes.append(data)
            data_labels.append(labels)
        # Visualize synthetic dataset
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         for i, data_class in enumerate(data_classes):
#             ax.scatter(data_class[:,0],data_class[:,1],data_class[:,2], label="Class "+str(i))
#         plt.legend()
#         plt.show()        
        self.X = np.vstack(data_classes)
        self.y = np.vstack(data_labels)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx].astype(float)



# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes=2):
        super(NeuralNet, self).__init__()
        self.out_size = 1 if n_classes<=2 else n_classes
        self.fc1 = nn.Linear(input_size, hidden_size).float() 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, self.out_size).float()   

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        if(self.out_size >1):
            out = F.softmax(out)
        else:
            out = torch.sigmoid(out)
        return out


if __name__ == "__main__":

    print("Starting tests ...")
    if(torch.cuda.is_available()):
        device = torch.device('cpu')
        print("Running on GPU ...")
    else:
        device = torch.device('cpu')
        print("Running on CPU ...")

    #############################################
    # Test 1 : Binary classification
    #############################################
    print("-----------------------------------------------\nTesting nitorch.trainer.Trainer() : Binary classification")
    n_classes = 2 
    BATCH_SIZE = 64
    EPOCHS = 20    
#     random.seed(42)
#     np.random.seed(42)
#     torch.manual_seed(42)

    data = syntheticDataset(n_classes=n_classes)
    # shuffle and split into test and train
    train_size = int(0.75*len(data))
    train_data, val_data = random_split(data, (train_size, len(data) - train_size))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    # NETWORK
    net = NeuralNet(3, 10).to(device).double()
    net.apply(weights_init)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(net.parameters(),
                           lr=1e-3,
                           weight_decay=1e-5)
    
    metrics = [specificity, sensitivity, binary_balanced_accuracy]
    
    trainer = Trainer(
    net,
    criterion,
    optimizer,
    # scheduler=None,
    metrics=metrics,
    # callbacks=callbacks,
    device=device,
    prediction_type="binary")

    # train model and store results
    net, report = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=EPOCHS,
        show_train_steps=5,
        show_validation_epochs=5
        )
    
    # Test 1-a: Check if the loss reduces with   
    # a negative slope on the valdation dataset
    y = (report["val_metrics"]["loss"])
    x = np.arange(len(y))
    slope, intercept, _, _, _ = stats.linregress(x,y)
    # plt.scatter(x, y)
    # plt.plot((slope*x + intercept))
    # plt.show()
    assert slope<0, "Test 1-a 'binary classification': The loss is not reducing with training - TEST FAILED"
    print("Test 1-a 'binary classification': The loss on the valdation dataset reduces with training - TEST PASSED")
    
    # Test 1-b: Check if the metrics are calculated correctly and improve with training
    assert (report["val_metrics"]['binary_balanced_accuracy'][0]<report["val_metrics"]['binary_balanced_accuracy'][-1]) or (report["val_metrics"]['binary_balanced_accuracy'][0] < report["val_metrics"]['binary_balanced_accuracy'][-2]),  "Test 1-b : The accuracy did not improve with training - TEST FAILED"
    print("Test 1-b 'binary classification': 'binary_balanced_accuracy' metric improved with training - TEST PASSED")
    
     #############################################
    # Test 2 : Multi-class classification
    #############################################
    print("-----------------------------------------------\nTesting nitorch.trainer.Trainer() : Multi-class classification")
    n_classes = 4
    data = syntheticDataset(n_classes=n_classes)

    # shuffle and split into test and train
    train_size = int(0.75*len(data))
    train_data, val_data = random_split(data, (train_size, len(data) - train_size))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    
    # NETWORK
    net = NeuralNet(3, 10, n_classes=n_classes).to(device).double()
    net.apply(weights_init)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(net.parameters(),
                           lr=1e-3,
                           weight_decay=1e-5)
    
    metrics = [classif_accuracy] 

    trainer = Trainer(
    net,
    criterion,
    optimizer,
    # scheduler=None,
    metrics=metrics,
    # callbacks=callbacks,
    device=device,
    prediction_type="classification")

    # train model and store results
    net, report = trainer.train_model(
        train_loader,
        val_loader,
        num_epochs=EPOCHS,
        show_train_steps=5,
        show_validation_epochs=5
        )
    # Test 2-a: Check if the loss reduces with a negative slope on the valdation dataset
    y = (report["val_metrics"]["loss"])
    x = np.arange(len(y))
    slope, intercept, _, _, _ = stats.linregress(x,y)
    # plt.scatter(x, y)
    # plt.plot((slope*x + intercept))
    # plt.show()
    assert slope<0, "Test 1-a 'Multi-class classification': The loss is not reducing with training - TEST FAILED"
    print("Test 1-a 'Multi-class classification': The loss on the valdation dataset reduces with training - TEST PASSED")
    
    # Test 2-b: Check if the metrics are calculated correctly and improve with training
    assert (report["val_metrics"]['classif_accuracy'][0]<report["val_metrics"]['classif_accuracy'][-1]) or (report["val_metrics"]['classif_accuracy'][0] < report["val_metrics"]['binary_balanced_accuracy'][-2]),  "Test 1-b : The accuracy did not improve with training - TEST FAILED"
    print("Test 1-b 'Multi-class classification': 'classif_accuracy' metric improved with training - TEST PASSED")