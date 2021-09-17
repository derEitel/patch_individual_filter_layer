import numpy as np
import nibabel as nib
import pandas as pd
import torch
import os
import sys
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# set nitorch path
NITORCH_DIR = os.getcwd()

# load functions from nitorch
sys.path.insert(0, os.path.join(NITORCH_DIR))
from nitorch.data import DataBunch
from nitorch.trainer import Trainer
from nitorch.initialization import weights_init
from nitorch.metrics import binary_balanced_accuracy


def load_nifti(
        file_path: str,
        dtype=np.float32,
        incl_header: bool = False,
        mask: np.ndarray = None,
        remove_nan: bool = True):
    """Routine to load a nifti file.

    Parameters
    ----------
    file_path : str
        The path to the nifti, including its filename.
    dtype
        A desired type the data gets transformed to.
    incl_header : bool
        Flag to return either data only (False) or data and image header (True).
    mask : np.ndarray
        A mask of the same shape of the data the data gets multiplied with.
    remove_nan : bool
        Flag to indicate to remove nan-values (True) or not (False).

    Returns
    -------
    struct_arr : np.ndarray
        The numpy array of the image.
    img
        The complete image.
    """
    # load the actual image
    img = nib.load(file_path)

    # get data transforming into a certain type (less memory needed)
    struct_arr = img.get_data().astype(dtype)

    # replace infinite values with 0
    if np.inf in struct_arr:
        struct_arr[struct_arr == np.inf] = 0.0

    # remove nan
    if remove_nan:
        struct_arr = np.nan_to_num(struct_arr)

    # apply a mask (e.g. binary mask)
    if mask is not None:
        struct_arr *= mask

    # return either the data only, or the data and the image
    if incl_header:
        return struct_arr, img
    else:
        return struct_arr


class ToTensor(object):
    """Convert ndarrays to Tensors."""

    def __call__(self, ndarray):
        """Calling routine

        Parameters
        ----------
        image : np.ndarray/np.int
            The array/int to transform to Tensor
        Returns
        -------
        image : torch.tensor
            The tensor object of the ndarray/int
        """
        if isinstance(ndarray, np.ndarray):
            t = torch.from_numpy(ndarray).unsqueeze(0)
            t = t.float()
        elif isinstance(ndarray, np.number):
            t = torch.tensor(ndarray, dtype=torch.float)
        else:
            raise TypeError
        return t


class SimpleLoaderClass(Dataset):
    """A simple data loader class."""

    def __init__(self, data_df,
                 modality,
                 label,
                 image_transform=None,
                 label_transform=None):
        """Initialization routine for the class object.

        Parameters
        ----------
        data_df : pandas.Dataframe
            The data in form of a panda dataframe.
        modality : str
            Either "T1" or "T2". Columns mus exist in dataframe and must provide a path to the data.
        label : str
            The label column. Column must exist in dataframe and  must provide a label for the data.
        image_transform
            Transformation to be applied to the data.
        label_transform
            Transformation to be applied to the label.
        """

        self.data = data_df
        self.X_modality = modality
        self.label = label
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __len__(self):
        """Obtain the length of the dataset.

        Returns
        -------
        length : int
            The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Loads an item from the dataset.

        Parameters
        ----------
        idx : int
            The index to load.

        Returns
        -------
        image and label : dict
            The key "image" holds the data, the key "label" holds the label.
        """
        # check validity of index
        if idx >= len(self.data):
            raise IndexError

        # loading the image
        img = nib.load(self.data[self.X_modality][idx])
        struct_arr = img.get_data().astype(np.float32)

        # load the label
        y = self.data[self.label][idx]

        # image transformation
        if self.image_transform:
            struct_arr = self.image_transform(struct_arr)

        # label transformation
        if self.label_transform:
            y = self.label_transform(y)

        return {"image": struct_arr, "label": y}


def main1():
    """Main routine."""

    # load csv describing the data
    participants = pd.read_csv("/analysis/ritter/data/ADNI/ADNI_BIDS_3T/participants.csv", sep=",")

    # loop over the participants
    for index, row in participants.iterrows():
        print("Loading image {} from path {}".format(index, row["T1"]))

        # load the data with the aid of the path in the csv
        img = load_nifti(row["T1"])

        # do something with the data, e.g. predict its outcome with a pretrained model
        print(type(img))

        # disable this line to load every datapoint in the csv!
        break


def main2():
    """Main routine."""

    # load csv describing the data
    participants = pd.read_csv("/analysis/ritter/data/ADNI/ADNI_BIDS_3T/participants.csv", sep=",")

    # factorize labels
    labels, uniques = pd.factorize(participants["GROUP"])
    participants["GROUPFACTORIZED"] = labels

    print("Found labels: {}".format(uniques))

    # using an object of our class to load data
    slc = SimpleLoaderClass(participants, "T1", "GROUPFACTORIZED",
                            image_transform=ToTensor(), label_transform=ToTensor())

    # accessing an index
    test_img = slc[0]
    print(type(test_img["image"]))
    print(type(test_img["label"]))

    # using the torch class DataLoader to load data batch wise
    slc_dl = DataLoader(slc, batch_size=8, num_workers=1, shuffle=True)
    print(type(slc_dl))

    # DataLoader class can be used for example to train Networks


class BaseModels(nn.Module):
    """A base class for pytorch modules.

    This class allows easy setup of complex pytorch models.
    Further, its functionality allows to start and stop the forward process of data from any arbitrary layer
    of the model.

    Attributes
    ----------
    return_after : str
        The name of the layer after which to stop the forward process and prematurely return modeled data.
        Can only be a name listed in attribute 'operation_sequence'.
        Default behavior is to forward data without prematurely returning.
        (Forward till end of model architecture) Default: None
    start_from : str
        The name of the layer at which to start the forward process.
        Can only be a name listed in attribute 'operation_sequence'.
        Default behavior is to forward data beginning in the
        first layer of the model architecture. Default: None
    operation_sequence : list
        The sequence in which the layers will be used in the forward process.
    debug : int
        A flag to enable debugging. Prints additional information about data size.

    Methods
    -------
    forward()
        Defines what actually happens to the data in the forward process.
    get_num_params()
        Get the number of parameters which require a gradient (trainable parameters).
    """
    def __init__(self):
        nn.Module.__init__(self)
        self.return_after = None
        self.start_from = None
        self.operation_sequence = []
        self.debug = 0

    def _update_operation_sequence(self):
        start = 0
        if self.start_from is not None:
            start = self.operation_sequence.index(self.start_from)
        stop = len(self.operation_sequence)
        if self.return_after is not None:
            stop = self.operation_sequence.index(self.return_after)
        updated_operation_sequence = self.operation_sequence[start:stop + 1]
        return updated_operation_sequence

    def forward(self, x):
        """Forward function of the model. Defines what actually "happens" with the data during modelling.

        Parameters
        ----------
        x : torch.tensor
            The data to forward.

        Returns
        -------
        torch.tensor
            The forwarded (modeled) data.

        """
        op_sqn = self._update_operation_sequence()
        for operation in op_sqn:
            x = self.__getattr__(operation)(x)
            if self.debug:
                print(x.shape)
        return x

    def get_num_params(self):
        """Get the number of parameters which require a gradient (trainable parameters).

        Returns
        -------
        int
            Number of trainable (can be optimized) parameters.

        """
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        num_trainable = sum([np.prod(p.size()) for p in trainable_params])
        return num_trainable


class Flatten(nn.Module):
    """Layer which flattens the data when called.

    Methods
    -------
    forward()
        Defines what actually happens to the data in the forward process.

    """
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        return x.view(x.size(0), -1)


class DemonstationModel(BaseModels):
    """Simple Convolutional Neuronal Network

    Parameters
    ----------
    return_after : str
        The name of the layer after which to stop the forward process and prematurely return modeled data.
        Can only be a name listed in attribute 'operation_sequence'.
        Default behavior is to forward data without prematurely returning.
        (Forward till end of model architecture) Default: None
    start_from : str
        The name of the layer at which to start the forward process.
        Can only be a name listed in attribute 'operation_sequence'.
        Default behavior is to forward data beginning in the
        first layer of the model architecture. Default: None
    debug : int
        A flag to enable debugging. Prints additional information about data size.

    Attributes
    ----------
    Conv_1 : nn.Conv3d
        3D Convultion layer.
    Conv_1_mp : nn.MaxPool3d
        3D Maxpool layer.
    dense_1 : nn.Linear
        Linear layer.
    flatten : Flatten
        Flatten layer.
    get_class : nn.Sigmoid
        a sigmoid layer.
    return_after : str
        The name of the layer after which to stop the forward process and prematurely return modeled data.
        Can only be a name listed in attribute 'operation_sequence'.
        Default behavior is to forward data without prematurely returning.
        (Forward till end of model architecture) Default: None
    start_from : str
        The name of the layer at which to start the forward process.
        Can only be a name listed in attribute 'operation_sequence'.
        Default behavior is to forward data beginning in the
        first layer of the model architecture. Default: None
    operation_sequence : list
        The sequence in which the layers will be used in the forward process.
    debug : int
        A flag to enable debugging. Prints additional information about data size.

    """
    def __init__(self, debug=False, return_after=None, start_from=None):
        super(BaseModels, self).__init__()
        self.Conv_1 = nn.Conv3d(1, 8, 3)
        self.Conv_1_mp = nn.MaxPool3d(3, return_indices=False)
        self.dense_1 = nn.Linear(22528, 1)
        self.activation_1 = nn.ReLU()
        self.debug = debug
        self.flatten = Flatten()
        self.get_class = nn.Sigmoid()
        self.return_after = return_after
        self.start_from = start_from
        self.operation_sequence = ["Conv_1",
                                   "activation_1",
                                   "Conv_1_mp",
                                   "flatten",
                                   "dense_1",
                                   "get_class"]


def main3():
    """Main routine."""

    # set a source_dir to the data
    source_dir = os.path.join(NITORCH_DIR, "data/OASIS_BIDS_example_data/")

    # configure the DataBunch
    oasis_dbunch = DataBunch(
        source_dir=source_dir,
        image_dir="",  # necessary for a relative path in the csv
        table=f"OASIS_example.csv",
        separator="\t",
        path=os.path.join(source_dir, "results"),
        file_col="path",
        label_col="labels_simplified",
        ptid_col="subjectID",
        labels_to_keep=["clinical dementia", "healthy"],
        z_factor=0.2  # large zooming for demonstration purposes
    )
    # built the DataLoader
    oasis_dbunch.build_dataloaders(bs=2)
    oasis_dbunch.show_sample()

    # use a DemonstationModel CNN suited for our needs
    my_net = DemonstationModel()

    # applys function "weights_init" for each layer
    my_net = my_net.apply(weights_init)

    # choose a criterion and an optimizer and metric
    criterion = nn.BCELoss()
    optimizer = optim.SGD(my_net.parameters(), lr=0.001)
    metric = [binary_balanced_accuracy]

    # define the Trainer
    trainer = Trainer(
        my_net,
        criterion,
        optimizer,
        metrics=metric,
        prediction_type="binary",
        device=torch.device("cpu")  # default device is "cuda", but for demonstration purposes CPU is sufficient!
    )

    # simply run the trainer
    net, report = trainer.train_model(
        train_loader=oasis_dbunch.train_dl,
        val_loader=oasis_dbunch.val_dl,
        num_epochs=10,
    )

    # let's look at what happened
    trainer.visualize_training(report)

    # we could use evaluate_model function with a dataset different than those used for training to evaluate our model
    # Since our dataset is very small and serves as demonstration only we will use the same data twice.
    # Notice: Do not use the same data twice!
    trainer.evaluate_model(oasis_dbunch.val_dl)


# execute only when called as script via command line
if __name__ == '__main__':
    main3()
    print("Done")
