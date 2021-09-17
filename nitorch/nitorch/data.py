import os
import warnings
import logging
import numpy as np
import pandas as pd
import pickle
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom
import multiprocessing
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm_notebook as tqdm
import nibabel
import nibabel as nib
import nilearn.image as nilimg
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# pytorch
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

# nitorch
from nitorch.transforms import *

logger = logging.getLogger(__name__)


def balanced_subsample(y, size=None):
    """Balanced sample according to y.

    Balances samples either equally or according to a certain size.

    Parameters
    ----------
    y
        Labels to balance over.
    size
        size each label should have. e.g. 0.2 means 20% off all samples. Default: None

    Returns
    -------
    list
        List of indices belonging to a balanced subsample.

    """
    subsample = []
    if size is None:
        n_smp = y.value_counts().min()
    else:
        n_smp = int(size / len(y.value_counts().index))

    for label in y.value_counts().index:
        samples = y[y == label].index.values
        index_range = range(samples.shape[0])
        indexes = np.random.choice(index_range, size=n_smp, replace=False)
        subsample += samples[indexes].tolist()
    return subsample


def load_nifti(
        file_path: str,
        dtype=np.float32,
        incl_header: bool = False,
        z_factor=None,
        mask: np.ndarray = None,
        force_to_shape: list = None,
        remove_nan: bool = True,
):
    """Loads a volumetric image in nifti format (extensions .nii, .nii.gz etc.) as a 3D numpy.ndarray.

    Parameters
    ----------
    file_path : str
        Absolute path to the nifti file.
    dtype
        Datatype of the loaded numpy.ndarray. Default: np.float32
    incl_header : bool
        If True, the nifTI object of the image is also returned. Default: False
    z_factor
        The zoom factor along the axes. If a float, zoom is the same for each axis. If a sequence,
        zoom should contain one value for each axis. Default: None
    mask : np.ndarray
        A mask with the same shape as the original image. If provided then the mask is element-wise
        multiplied with the image ndarray. Default: None
    force_to_shape : list
        3-dimensional desired shape of the nifti. if specified, nifti will be forced to have these dimension by
        cutting the edges of the images, evenly on both sides of each dimension. Default: None
    remove_nan : bool
        If True, NaN values will be converted to num. Default: None

    Returns
    -------
    np.ndarray
        3D numpy.ndarray with axis order (saggital x coronal x axial)

    Raises
    ------
    TypeError
        If z_factor is not float.

    """

    img = nib.load(file_path)

    if force_to_shape:
        img = _force_to_shape(img, force_to_shape)

    struct_arr = img.get_data().astype(dtype)

    # replace infinite values with 0
    if np.inf in struct_arr:
        struct_arr[struct_arr == np.inf] = 0.0

    if remove_nan:
        struct_arr = np.nan_to_num(struct_arr)

    # replace NaN values with 0
    if np.isnan(struct_arr).any() == True:
        struct_arr[np.isnan(struct_arr)] = 0.0

    if z_factor is not None:
        if isinstance(z_factor, float):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                struct_arr = np.around(zoom(struct_arr, z_factor), 0)
        else:
            TypeError("z_factor has to be one of None or tuple")
    if mask is not None:
        struct_arr *= mask

    if incl_header:
        return struct_arr, img
    else:
        return struct_arr


def _force_to_shape(img_nii_X, shape, rtol=1e-8, copy=True, info=False):
    """Forces image to have a certain shape. Retuns cropped image.

    Parameters
    ----------
    img_nii_X : Niimg-like object
    shape : list
        desired shape of the nifti. 3-Dimensions -> 3 entries in the list
    rtol : float
        Dont keep values under these threshold. They only make calculation unnecessary difficult! Default: 1e-8
    copy : bool
        Specifies whether cropped data is to be copied or not. Default: True
    info : bool
        Specifies whether to print additional information or not. Default: False

    Returns
    -------
    numpy.ndarray
        3D numpy.ndarray with axis order (saggital x coronal x axial)

    Raises
    ------
    AssertionError
        shape in any dimension larger than the shape of the data.

    """
    data = img_nii_X.get_data()
    infinity_norm = max(-data.min(), data.max())
    passes_threshold = np.logical_or(
        data < -rtol * infinity_norm, data > rtol * infinity_norm
    )

    if data.ndim == 4:
        passes_threshold = np.any(passes_threshold, axis=-1)
    coords = np.array(np.where(passes_threshold))
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1

    # pad with one voxel to avoid resampling problems
    start = np.maximum(start - 1, 0)
    end = np.minimum(end + 1, data.shape[:3])

    # add as many voxel as neccessary to archive shape
    missing_diff_start = []
    missing_diff_end = []
    if shape:
        assert shape < list(
            data.shape
        ), "Shape of the data smaller than shape it should be transformed to!"
        diff = end - start
        missing_diff = shape - diff
        for dim in missing_diff:
            if dim % 2 == 0:
                missing_diff_start.append(int(dim / 2))
                missing_diff_end.append(int(dim / 2))
            else:
                missing_diff_start.append(
                    int(dim / 2)
                )  # conversion to int always floors the value
                # to archive desired shape we have to add what we floored before
                missing_diff_end.append(int(dim / 2) + 1)

        for idx, s in enumerate(start):
            if (s - missing_diff_start[idx]) > 0:
                start[idx] = s - missing_diff_start[idx]
            else:
                start[idx] = 0
                # add rest of the pixels to end
                diff = missing_diff_start[idx] - s
                end[idx] = end[idx] + diff
        end = end + missing_diff_end

    slices = [slice(s, e) for s, e in zip(start, end)]

    new_img = _crop_img_to(img_nii_X, slices, copy=copy)

    if info:
        print(
            "Input img shape: {}. Desired shape: {}. New img shape: {}".format(data.shape,
                                                                               shape, new_img.get_data().shape))

    return new_img


def _crop_img_to(img, slices, copy=True):
    """Crops image to a smaller size.

    Crops img to size indicated by slices and adjust affine accordingly.

    Parameters
    ----------
    img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Img to be cropped. If slices has less entries than img
        has dimensions, the slices will be applied to the first len(slices)
        dimensions
    slices : list
        Defines the range of the crop.
        E.g. [slice(20, 200), slice(40, 150), slice(0, 100)]
        defines a 3D cube.
    copy : bool
        Specifies whether cropped data is to be copied or not. Default: True

    Returns
    -------
    cropped_img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Cropped version of the input image.

    """

    data = img.get_data()
    affine = img.affine

    cropped_data = data[tuple(slices)]
    if copy:
        cropped_data = cropped_data.copy()

    linear_part = affine[:3, :3]
    old_origin = affine[:3, 3]
    new_origin_voxel = np.array([s.start for s in slices])
    new_origin = old_origin + linear_part.dot(new_origin_voxel)

    new_affine = np.eye(4)
    new_affine[:3, :3] = linear_part
    new_affine[:3, 3] = new_origin

    return nilimg.new_img_like(img, cropped_data, new_affine)


class H5pyDataset(Dataset):
    """Dataset for data X and label y.

    Abstract class for a dataset with data X and label y. Optionally applies transformation, masking,
    and zooming to the data. If specified, normalization will be executed.

    Parameters
    ----------
    X : ndarray
        Data matrix, favourably a numpy array
    y : ndarray
        Label matrix, favourably a numpy array
    transform
        Transformation to the data. See nitorch.transforms for example classes. Default: None
    mask
        Mask applicable to the data. Should have same size as 'X'. Default: None
    z_factor : float
        Zooming factor. Value between 0 and 1. Default: None
    dtype
        Data type. Data will be converted to the specified format. See numpy dtype for mor information.
        Default: np.float32

    Attributes
    ----------
    X : ndarray
        Data matrix, favourably a numpy array
    y : ndarray
        Label matrix, favourably a numpy array
    transform
        Transformation to the data. See nitorch.transforms for example classes.
    mask
        Mask applicable to the data. Should have same size as 'X'.
    z_factor : float
        Zooming factor. Value between 0 and 1.
    dtype
        Data type. Data will be converted to the specified format. See numpy dtype for mor information.
    mean : float
        The mean of the dataset.
    std : float
        The standard deviation of the dataset.

    Methods
    -------
    fit_normalization(num_sample=None, show_progress=False)
        Sets normalization parameters 'mean' and 'std'.

    """

    def __init__(self, X, y, transform=None, mask=None, z_factor=None, dtype=np.float32):
        """Sets data and label, transformation, mask, zooming and data-type."""
        self.X = np.copy(X)
        self.y = np.copy(y)
        self.transform = transform
        self.mask = mask
        self.z_factor = z_factor
        self.dtype = dtype
        self.mean = None
        self.std = None

    def __len__(self):
        """The length of the Dataset.

        Returns
        -------
        int
            Length of the Dataset.

        """
        return len(self.X)

    def __getitem__(self, idx):
        """Accesses the data and the label.

        Parameters
        ----------
        idx: int
            The index to access the data from.

        Returns
        -------
        image : any
            The data (usually an image) with optionally applied transformation, zooming and normalization.
        label : any
            The unchanged label of the data.

        """
        image = self.X[idx]
        label = self.y[idx]

        # normalize by max
        image /= np.max(image)

        if self.transform:
            image = self.transform(image)

        # todo: implement zooming or remove from parameter list

        if self.mean is not None:
            image = (image - self.mean) / (self.std + 1e-10)

        return image, label

    def fit_normalization(self, num_sample=None, show_progress=False):
        """Calculate the voxel-wise mean and std across the dataset for normalization.

        Calculate the voxel-wise mean and std across the dataset for normalization thereby setting attributes mean
        and std of the Dataset object.

        Parameters
        ----------
        num_sample : int
            If None, calculate the values across the complete dataset, otherwise sample a number of images.
            Default: None
        show_progress : bool
            When set, a progress bar during the calculation is shown. Default: False

        """

        if num_sample is None:
            num_sample = len(self)
        image_shape = self.__getitem__(0)
        all_struct_arr = np.zeros(
            (num_sample, image_shape[0], image_shape[1], image_shape[2])
        )
        sampled_filenames = np.random.choice(
            self.filenames, num_sample, replace=False
        )

        if show_progress:
            sampled_filenames = tqdm(sampled_filenames)

        for i, filename in enumerate(sampled_filenames):
            all_struct_arr[i] = self.__getitem__(i)

        self.mean = all_struct_arr.mean(0)
        self.std = all_struct_arr.std(0)


class MRIDataset(Dataset):
    """Dataset that consists of MRI images and labels.

    Abstract class that accesses the data during runtime. Data consists of image and label. Optionally applies
    transformation, zooming, masking, cropping to the data.

    Parameters
    ----------
    filenames : list
        All filenames (full paths) of the dataset.
    labels : list
        All labels of the corresponding filename.
    mask : np.ndarray
        Mask applicable to the image. Default: None
    force_to_shape: list
        Shape the data is supposed to cropped to. List containing for each data-dimension the corresponding value.
        Default: None
    transform
        Transformation applicable to the image. See nitorch.transforms for more information. Default: None
    z_factor : float
        Zooming factor applicable to the image. Default: None

    Attributes
    ----------
    filenames : list
        All filenames (full paths) of the dataset.
    labels : list
        All labels of the corresponding filename.
    mask : np.ndarray
        Mask applicable to the image.
    force_to_shape: list
        Shape the data is supposed to cropped to. List containing for each data-dimension the corresponding value.
    transform
        Transformation applicable to the image. See nitorch.transforms for more information.
    z_factor: float
        Zooming factor applicable to the image.
    num_inputs : int
        Value always set to '1'. Necessary for sampling.
    num_targets : int
        Value always set to '1'. Necessary for sampling.
    mean : float
        Default 0, should be set via  'fit_normalization' method.
    std : float
        Default 1, should be set via  'fit_normalization' method.
    shape
        Shape of the data.
    label_counts: dict
        Dictionary containing the counts of each label
    class_weights: np.array
        Weights for each class. On default uniform.

    Methods
    ----------
    get_image_shape()
        Returns the shape of the image(s).
    fit_normalization(num_sample=None, show_progress=False)
        Sets normalization parameters 'mean' and 'std'.
    get_raw_image(idx: int)
        Gets the unrpocessed image at certain index.

    """

    # Todo: develop mutitask data (storing more than one label)
    def __init__(
            self,
            filenames: list,
            labels: list,
            z_factor: float = None,
            mask: np.ndarray = None,
            transform=None,
            force_to_shape=None,
    ):
        """Specifies the dataset."""
        self.filenames = filenames
        self.labels = torch.FloatTensor(labels)
        self.label_counts = dict(zip(*np.unique(labels, return_counts=True)))
        self.class_weights = np.array(list(self.label_counts.values())) / len(labels)
        self.mask = mask
        self.force_to_shape = force_to_shape
        self.transform = transform
        self.z_factor = z_factor

        # Required by torchsample.
        self.num_inputs = 1
        self.num_targets = 1

        # Default values. Should be set via fit_normalization.
        self.mean = 0
        self.std = 1

        self.shape = self.get_image_shape()

    def __len__(self):
        """The length of the dataset.

        Returns
        -------
        int
            Length of the dataset

        """
        return len(self.filenames)

    def __repr__(self):
        """The representation of the dataset.

        Returns
        -------
        str
            String representation of the dataset. Explaining length, data shape, and labels (classes)
        """

        return "MRIDataset - no. samples: {}; shape: {}; no. classes: {}".format(len(self), self.shape,
                                                                                 len(self.labels.unique()))

    def __getitem__(self, idx: int):
        """Returns the image (FloatTensor) and its corresponding label.

        Parameters
        ----------
        idx: int
            index to access data

        Returns
        -------
        struct_arr : torch.FloatTensor
            The image as tensor.
        label : torch.FloatTensor
            The label as tensor with an additional added first emtpy dimension.
        """

        label = self.labels[idx].unsqueeze_(-1)  # add the missing dimension

        struct_arr = load_nifti(
            self.filenames[idx],
            mask=self.mask,
            z_factor=self.z_factor,
            dtype=np.float32,
            force_to_shape=self.force_to_shape,
        )
        # TDOO: Try normalizing each image to mean 0 and std 1 here.
        # struct_arr = (struct_arr - struct_arr.mean()) / (struct_arr.std() + 1e-10)
        # prevent 0 division by adding small factor

        # todo: check why normalizing only when transform is not None. why can't happen both?
        if self.transform is not None:
            struct_arr = self.transform(struct_arr)
        else:
            struct_arr = (struct_arr - self.mean) / (self.std + 1e-10)
            struct_arr = torch.FloatTensor(
                struct_arr[None]
            )  # add (empty) channel dimension

        return struct_arr, label

    def get_image_shape(self):
        """Determines the shape of the MRI images.

        Loads the first image in the dataset and determine its shape. Each image in the dataset is expected to have
        equal size.

        Returns
        -------
        shape
            The shape of the first image.
        """

        img = load_nifti(
            self.filenames[0],
            mask=None,
            z_factor=self.z_factor,
            force_to_shape=self.force_to_shape,
        )
        return img.shape

    def fit_normalization(self, num_sample: int = None, show_progress: bool = False):
        """Calculates the voxel-wise mean and std across the dataset for normalization.

        Iterates through the object and determines the mean and std of each voxel across the dataset. Performs
        downsampling if specified. Additionally shows a progress bar when specified.

        Parameters
        ----------
        num_sample : int
            Number of samples used for the normalization calculation. Can reduce calculation time, especially
            when the dataset is large. Default: None
        show_progress : bool
            Indicates whether to show progress bar or not. Default: False

        """
        if num_sample is None:
            num_sample = len(self)
        image_shape = self.get_image_shape()
        all_struct_arr = np.zeros(
            (num_sample, image_shape[0], image_shape[1], image_shape[2])
        )
        sampled_filenames = np.random.choice(
            self.filenames, num_sample, replace=False
        )

        if show_progress:
            sampled_filenames = tqdm(sampled_filenames)

        for i, filename in enumerate(sampled_filenames):
            all_struct_arr[i] = load_nifti(
                filename,
                mask=self.mask,
                z_factor=self.z_factor,
                force_to_shape=self.force_to_shape,
            )

        self.mean = all_struct_arr.mean(0)
        self.std = all_struct_arr.std(0)

    def get_raw_image(self, idx: int):
        """Return the raw image at a specific index (i.e. no normalization, no transformation)

        Parameters
        ----------
        idx : int
            Index of the image to access.

        Returns
        -------
        np.array
            The raw masked, zoomed and cropped image.

        """
        return load_nifti(
            self.filenames[idx],
            mask=self.mask,
            z_factor=self.z_factor,
            force_to_shape=self.force_to_shape,
        )


def get_image_filepath(df_row, source_dir=""):
    """Determines the filepath of the image that is described in the row of the data table.

    Parameters
    ----------
    df_row
        A row of a pandas data table.
    source_dir
        The source directory the image can be found. Default: ""

    Returns
    -------
    str
        the filepath of the image of the row of the data table.
    """

    # Current format for the image filepath is:
    # <PTID>/<Visit (spaces removed)>/<PTID>_<Scan.Date (/ replaced by -)>_<Visit (spaces removed)>_<Image.ID>_<DX>_Warped.nii.gz

    filedir = os.path.join(df_row["PTID"], df_row["Visit"].replace(" ", ""))
    filename = "{}_{}_{}_{}_{}_Warped.nii.gz".format(
        df_row["PTID"],
        df_row["Scan.Date"].replace("/", "-"),
        df_row["Visit"].replace(" ", ""),
        df_row["Image.ID"],
        df_row["DX"],
    )
    return os.path.join(source_dir, filedir, filename)


class DataBunch:
    """Organizes a collection of MRI images in a handy way.

    Without knowledge on data storage, this class organizes MRI images by only needing a single csv file.
    It builds training, validation and hold out datasets upon initialization which can be used for training
    models of any kind. Assigning data to the different dataset groups can be done based on arguments in many ways,
    respecting classes, grouping several scans at different time points according to their subject ID or balancing data.
    Downsampling, transformation, normalization is additionally supported. Moreover, the object configuration
    can be stored and loaded onto the disk and data can be saved (and loaded) as h5py-file for quicker access.
    Object can be shared to easily reproduce results.

    Parameters
    ----------
    source_dir : str
        Path to source_dir folder, where table and image_dir can be found.
    path : str
        Path where intermediary data will be stored (eg. cache).
    table : str
        CSV file path *relative* to source_dir containing samples.
        The tables *must* contain 'file_col', 'label_col' and 'ptid_col' columns.
    image_dir : str
        Image directory *relative* to source_dir, where the .nii files are.
        Set empty string ("") if 'image dir' equals 'source_dir'. Default: None
    mask : str
        Path to binary brain mask in .nii format. This will be resized with z_factor. Default: None
    transforms
        A PyTorch Compose container object, with the transformations to apply to samples.
        Default: Compose([ToTensor(), IntensityRescale(masked=False, on_gpu=True)])
    labels_to_keep : list
        List of labels to keep in the datasets. Defaults to None (all labels). Default: None
    prediction_type : str
        Either "c" or "classification" for classification problems, or "r" or "regression" for
        regression problems. If regression is specified, labels are taken as they are and not translated to internal
        class IDs. Default: "c"
    get_file_path : callable
        A function mapping the rows of table to the respective file paths of the samples. Default: None
    balance : bool
        Boolean switch for enforcing balanced classes. If True dataset will be balanced before sampling (e.g. splitting)
        Default: False
    num_samples : int
        Total number of samples available for sampling groups (e.g. test, validation, hold-out). Large datasets can
        therefore be reduced in size. Defaults value (None) uses all available images. (e.g. if dataset has
        14.000 datapoints, and 'num_samples'=1000, dataset will be randomly reduced to 1000 samples.)
        Default: None
    num_training_samples : int
        Reduces the number of training samples, but do not reduce size of the whole dataset
        (e.g. validation and hold-out remain large). Usually used for debugging or intensive grid searching.
        Default: None
    z_factor : float
        Zoom factor applied to each image. Default: 0.5
    hold_out_size : float
        Sets the relative size of the hold_out set. Use 'num_samples' to reduce dataset, otherwise 'hold_out_size'
        is relative to the total number of available datapoints. Default: None
    val_size : float
        Sets the relative size of the validation set. Use 'num_samples' to reduce dataset, otherwise 'val_size'
        is relative to the total number of available datapoints. Default: 0.1
    grouped : bool
        Boolean switch for enforcing grouping according to subjectID. Necessary if subjects have more than one
        scan available. Default: False
    cache_filename: str
        Filename specifying a prefix of all stored files. File-extensions in 'cache_filename' will be discarded.
        If not specified "databunch" will be used as default value. Default: None
    force_to_shape : list
        List, specifying for each dimension of the data the size. (e.g. 'force_to_shape'=[80,80,80] will force images
        to have the shape of (80,80,80).)
        Default: None
    file_col : str
        Column name in 'table' identifying the path to the .nii file of the sample. Default: "file_path"
    label_col : str
        Column name in 'table' identifying the path to the label of the sample. Default: "DX"
    ptid_col : str
        Column name in 'table' identifying the path to the patient ID of the sample. Default: "PTID"
    random_state : int
        Random state to enforce reproducibility for train/test splitting. Default: 42
    separator : str
        Separator used to load 'table' and to eventually store csv splits. Default: ","
    kwargs
        Arbitrarily many Pairs of (key, value) which will stored in the object.

    Attributes
    ----------
    source_dir : str
        Path to source_dir folder, where table and image_dir can be found.
    path : str
        Path where intermediary data will be stored (eg. cache).
    table : str
        CSV file path *relative* to source_dir containing samples.
        The tables *must* contain 'file_col', 'label_col' and 'ptid_col' columns.
    image_dir : str
        Image directory *relative* to source_dir, where the .nii files are.
        Set empty string ("") if 'image dir' equals 'source_dir'.
    _mask : str
        Path to binary brain mask in .nii format. This will be resized with z_factor.
    _transforms
        A PyTorch Compose container object, with the transformations to apply to samples.
        Defaults to using ToTensor() and IntensityRescaling() into [0,1].
    _labels_to_keep : list
        List of labels to keep in the datasets. Defaults to None (all labels).
    _prediction_type : str
        Either "c" or "classification" for classification problems, or "r" or "regression" for
        regression problems. If regression is specified, labels are taken as they are and not translated to internal
        class IDs.
    _get_file_path : callable
        A function mapping the rows of table to the respective file paths of the samples.
    _balance : bool
        Boolean switch for enforcing balanced classes. If True dataset will be balanced before sampling (e.g. splitting)
    _num_samples : int
        Total number of samples available for sampling groups (e.g. test, validation, hold-out). Large datasets can
        therefore be reduced in size. Defaults value (None) uses all available images. (e.g. if dataset has
        14.000 datapoints, and 'num_samples'=1000, dataset will be randomly reduced to 1000 samples.)
    _num_training_samples : int
        Reduces the number of training samples, but do not reduce size of the whole dataset
        (e.g. validation and hold-out remain large). Usually used for debugging or intensive grid searching.
    _z_factor : float
        Zoom factor applied to each image.
    _hold_out_size : float
        Sets the relative size of the hold_out set. Use 'num_samples' to reduce dataset, otherwise 'hold_out_size'
        is relative to the total number of available datapoints.
    _val_size : float
        Sets the relative size of the validation set. Use 'num_samples' to reduce dataset, otherwise 'val_size'
        is relative to the total number of available datapoints.
    _grouped : bool
        Boolean switch for enforcing grouping according to subjectID. Necessary if subjects have more than one
        scan available.
    _cache_filename: str
        Filename specifying a prefix of all stored files. File-extensions in 'cache_filename' will be discarded.
        If not specified "databunch" will be used as default value
    _force_to_shape : list
        List, specifying for each dimension of the data the size. (e.g. 'force_to_shape'=[80,80,80] will force images
        to have the shape of (80,80,80).)
    _file_col : str
        Column name in 'table' identifying the path to the .nii file of the sample.
    _label_col : str
        Column name in 'table' identifying the path to the label of the sample.
    _ptid_col : str
        Column name in 'table' identifying the path to the patient ID of the sample.
    random_state : int
        Random state to enforce reproducibility for train/test splitting.
    separator : str
        Separator used to load 'table' and to eventually store csv splits.
    kwargs
        Arbitrarily many Pairs of (key, value) which will stored in the object.
    DEFAULT_FILE : str
        Default column in csv where filenames are stored: "file_path".
    DEFAULT_LABEL : str
        Default column in csv where labels are stored: "DX".
    DEFAULT_PTID : str
        Default column in csv where patient IDs are stored: "PTID".
    CACHE_NAME : str
        Default cache name: "databunch".
    show_stats : bool
        If disabled, stats are not shown
    loaded_cache : bool
        Should not be changed - Indicates an initialization via cached object.
    load_h5py : bool
        Should not be changed - When enabled, data is loaded in h5py format. Automatically set.
    mean: float
        Mean value which will be applied for all datasets.
    std : float
        Standard deviation which will be used for all datasets.
    use_sample : int
        Number of samples to use for normalization process. Default: None (all samples).
    shape
        Shape of the data.
    df_orig : pd.DataFrame
        DataFrame resulting in preprocessing of the csv from the source folder.
    df : pd.DataFrame
        Task specific dataframe. This dataframe is used for group assignemnt. Depending on attributes from the
        initialization process this is a subsample from 'df_orig' or balanced etc.
    df_trn : pd.DataFrame
        Task specific dataframe containing all training samples.
    df_val : pd.DataFrame
        Task specific dataframe containing all validation samples.
    df_ho : pd.DataFrame
        Task specific dataframe containing all hold-out samples.
    train_ds : MRIDataset/H5pyDataset
        Dataset in either MRIDataset or H5pyDataset format. Contains all the training samples.
        Data accassable via indexing.
    val_ds : MRIDataset/H5pyDataset
        Dataset in either MRIDataset or H5pyDataset format. Contains all the validation samples.
        Data accassable via indexing.
    ho_ds : MRIDataset/H5pyDataset
        Dataset in either MRIDataset or H5pyDataset format. Contains all the hold-out samples.
        Data accassable via indexing.
    classes : list
        List of available classes in Task specific dataframe ('df'). Only set in a classification task.
    label2id : dict
        Dictionary translating between class label and internal integer ID. Only set in a classification task.
    id2label : dict
        Dictionary translating between internal integer ID and class label. Only set in a classification task.
    train_dl : DataLoader
        DataLoader of the training dataset. Organizes data in batches.
    val_dl : DataLoader
        DataLoader of the validation dataset. Organizes data in batches.
    ho_dl : DataLoader
        DataLoader of the hold-out dataset. Organizes data in batches.

    Methods
    ----------
    from_csv(folder: str, source_dir: str, load_filename: str = None, kwargs)
        Initialization via csv. The CSVs with 'load_filename' as prefix available in 'folder' are taken.
        CSV containing train subject *MUST* be available under 'folder' (e.g. if  'load_filename'="test",
        csv called "test_train.csv", "test_val.csv", "test_hold.csv" will be used for initialization.)
        All other arguments are similar to usual initialization process.
    from_disk(load_path: str, save_path: str, cache_filename: str, load_h5py: bool = False)
        Initialization via cached object. When specified, data is additionally loaded from disk in H5pyDataset format.
    apply_changes()
        Applies all changes made to that object after initialization or the last call of 'apply_changes()'.
    reset_changes()
        Resets all changes made to that object after the last call of 'apply_changes()' or initialization.
    drop_h5py()
        Switches back to MRIDataset after loading an H5pyDataset.
    reshuffle_datasets()
        Reassignes subjects to train and validation set, leaving hold-out set untouched.
    build_dataloaders(bs: int = 8, normalize: bool = False, use_samples: int = None, num_workers: int = None)
        Organises Datasets (either MRIDataset or H5pyDataset) in batch sizes, applying normalization first.
    save_df(filename: str = None, separator: str = ",")
        Saves dataframes (train, val, hold) on the disk using the prefix provided under 'filename'.
    save_h5py(filename: str = None)
        Saves all MRIDatasets to disk in h5py format, using the prefix provided under 'filename'.
    load_h5py_ds(filename: str = None)
        Loads all available H5pyDataset from the disk, using the prefix provided under 'filename'.
    print_stats()
        Prints the basic statistics of the groups.
    show_sample(cmap="gray")
        Shows a random sample after zooming.
    load(filename: str = None)
        Updates the objects dict using cached file. Using the prefix provided under 'filename'.
    save(filename: str = None)
        Saves the object to disk using the prefix provided under 'filename'.

    """
    DEFAULT_FILE = "file_path"
    DEFAULT_LABEL = "DX"
    DEFAULT_PTID = "PTID"
    CACHE_NAME = "databunch"

    def __init__(
            self,
            source_dir: str,
            path: str,
            table: str,
            image_dir: str = None,
            mask: str = None,
            transforms: Compose = Compose(
                [ToTensor(), IntensityRescale(masked=False, on_gpu=True)]
            ),
            labels_to_keep: list = None,
            prediction_type: str = "classification",
            get_file_path: callable = None,
            balance: bool = False,
            num_samples: int = None,
            num_training_samples: int = None,
            z_factor: float = 0.5,
            hold_out_size: float = None,
            val_size: float = 0.1,
            grouped: bool = False,
            cache_filename: str = None,
            force_to_shape: list = None,
            file_col="file_path",
            label_col="DX",
            ptid_col="PTID",
            random_state: int = 42,
            separator: str = ",",
            **kwargs
    ):
        """Performs initialization routine.

        Returns
        -------
        DataBunch
            The initialized object.

        """
        # do not change the next line or its position!
        self.init = False

        # initialize... self.init *MUST* be set to FALSE!
        self.source_dir = source_dir
        self.path = path
        self.table = table
        self.image_dir = image_dir
        self.file_col = file_col
        self.label_col = label_col
        self.ptid_col = ptid_col
        self.transforms = transforms
        self.labels_to_keep = labels_to_keep
        self.prediction_type = prediction_type
        self.get_file_path = get_file_path
        self.balance = balance
        self.num_samples = num_samples
        self.num_training_samples = num_training_samples
        self.z_factor = z_factor
        self.hold_out_size = hold_out_size
        self.val_size = val_size
        self.grouped = grouped
        self.cache_filename = cache_filename
        self.force_to_shape = force_to_shape
        self.random_state = random_state
        self.separator = separator
        self.mask = mask

        # initialize with default values
        self.show_stats = True
        self.loaded_cache = False
        self.load_h5py = False
        self.mean = None
        self.std = None
        self.use_sample = None
        self.shape = None
        self.df_orig = pd.DataFrame()
        self.df = pd.DataFrame()
        self.df_trn = pd.DataFrame()
        self.df_val = None
        self.df_ho = None
        self.train_ds = None
        self.val_ds = None
        self.ho_ds = None
        self.classes = None
        self.label2id = None
        self.id2label = None
        self.mask_path = mask
        self.train_dl = None
        self.val_dl = None
        self.ho_dl = None

        # initial switch variables to apply changes
        self.set_df = False
        self.set_task_df = False
        self.set_split = False
        self.set_dataset = False
        self.reshuffle = True
        self._changes = {}

        logger.info(
            "Using file column {}; label column {} and patient_id column {}".format(self.file_col, self.label_col,
                                                                                    self.ptid_col)
        )

        # update according to kwargs
        for k, v in kwargs.items():
            self.__setattr__(k, v)

        if not self.loaded_cache:
            # init wrapper
            self._init_wrapper()

        # information
        if self.show_stats:
            self.print_stats()

            # done
            print("DataBunch initialized at {}".format(self.path))

        # allow changes to the object
        self.init = True

    @property
    def file_col(self):
        """Get file_col

        Returns
        -------
        str
            the file column of...

        """
        return self._file_col

    @file_col.setter
    def file_col(self, value):
        if self.init:
            self.reshuffle = False
            self.set_df = True
            p = self._file_col
            if "file_col" not in self._changes:
                self._changes["file_col"] = p
        self._file_col = value

    @property
    def label_col(self):
        """Gets the label column.

        Returns
        -------
        str
            The label column.

        """
        return self._label_col

    @label_col.setter
    def label_col(self, value):
        if self.init:
            self.reshuffle = False
            self.set_df = True
            p = self._label_col
            if "label_col" not in self._changes:
                self._changes["label_col"] = p
        self._label_col = value

    @property
    def ptid_col(self):
        """Gets the patient ID column.

        Returns
        -------
        str
            The patient ID column.
        """
        return self._ptid_col

    @ptid_col.setter
    def ptid_col(self, value):
        if self.init:
            self.reshuffle = False
            self.set_df = True
            p = self._ptid_col
            if "ptid_col" not in self._changes:
                self._changes["ptid_col"] = p
        self._ptid_col = value

    @property
    def mask(self):
        """Gets the mask.

        Notes
        -----
            Changing the mask also sets the mask_path.

        Returns
        -------
        np.ndarray/None
            The mask, either an numpy array or None.

        """
        return self._mask

    @mask.setter
    def mask(self, value):
        if self.init:
            # perform initialization process
            self.reshuffle = False
            self.set_dataset = True
            p = self._mask
            p2 = self.mask_path
            if "mask" not in self._changes:
                self._changes["mask"] = p
                self._changes["mask_path"] = p2
        if isinstance(value, str):
            self._mask = load_nifti(str(value), z_factor=self.z_factor, force_to_shape=self.force_to_shape) \
                if value is not None else None
            self.mask_path = value
        elif isinstance(value, np.ndarray):
            self._mask = load_nifti(str(self.mask_path), z_factor=self.z_factor, force_to_shape=self.force_to_shape) \
                if value is not None else None
        elif value is None:
            self._mask = None
        else:
            raise ValueError("Cannot set mask. Make sure mask is a String to a valid .nii file.")

    @property
    def z_factor(self):
        """Gets the z_factor.

        Notes
        -----
            Changing the z_factor also changes the mask.

        Returns
        -------
        float/None
            The z_factor if set or None

        """
        return self._z_factor

    @z_factor.setter
    def z_factor(self, value):
        if value is not None:
            if value < 0 or value > 1:
                raise ValueError("z_factor must be between 0 and 1.")
        if self.init:
            self.reshuffle = False
            self.set_dataset = True
            p = self._z_factor
            if "z_factor" not in self._changes:
                self._changes["z_factor"] = p
        self._z_factor = None if value == 0 else value
        # apply changes to other attributes
        if self.init:
            self.mask = self.mask_path

    @property
    def val_size(self):
        """Gets the val_size.

        Returns
        -------
        float/None
            The validation size.

        """
        return self._val_size

    @val_size.setter
    def val_size(self, value):
        if value is not None:
            if value < 0 or value > 1:
                raise ValueError("val_size must be between 0 and 1.")
        if self.init:
            self.set_split = True
            p = self._val_size
            if "val_size" not in self._changes:
                self._changes["val_size"] = p
        self._val_size = None if value == 0 else value

    @property
    def hold_out_size(self):
        """Gets the hold_out_size.

        Returns
        -------
        float/None
            The hold out size.

        """
        return self._hold_out_size

    @hold_out_size.setter
    def hold_out_size(self, value):
        if value is not None:
            if value < 0 or value > 1:
                raise ValueError("hold_out_size must be between 0 and 1.")
        if self.init:
            self.set_split = True
            self.reshuffle = False
            p = self._hold_out_size
            if "hold_out_size" not in self._changes:
                self._changes["hold_out_size"] = p
        self._hold_out_size = None if value == 0 else value

    @property
    def source_dir(self):
        """Gets the source_dir.

        Returns
        -------
        str
            The source_dir.

        """
        return self._source_dir

    @source_dir.setter
    def source_dir(self, value):
        if not os.path.isdir(value):
            raise RuntimeError("{} not existing!".format(value))
        if self.init:
            raise AttributeError("Cannot change source_dir once it is set!")
        self._source_dir = Path(value)

    @property
    def path(self):
        """Gets the path.

        Returns
        -------
        str
            The path the object is initialized to.
        """
        return self._path

    @path.setter
    def path(self, value):
        os.makedirs(value, exist_ok=True)
        self._path = Path(value)

    @property
    def cache_filename(self):
        """Gets the cache_filename

        Returns
        -------
        str
            The chache_filename.
        """
        return self._cache_filename

    @cache_filename.setter
    def cache_filename(self, value):
        self._cache_filename = self.CACHE_NAME if value is None or value == "" else os.path.splitext(value)[0]

    @property
    def image_dir(self):
        """Gets the image_dir.

        Returns
        -------
        str
            The image_dir.

        """
        return self._image_dir

    @image_dir.setter
    def image_dir(self, value):
        if self.init:
            raise AttributeError("Cannot change image_dir once it is set!")
        self._image_dir = (os.path.join(self.source_dir, value) if value is not None else None)

    @property
    def get_file_path(self):
        """Gets the get_file_path function.

        Returns
        -------
        callable/None
            The get_file_path function.

        """
        return self._get_file_path

    @get_file_path.setter
    def get_file_path(self, fkt):
        if self.init:
            self.reshuffle = False
            self.set_df = True
            p = self._get_file_path
            if "get_file_path" not in self._changes:
                self._changes["get_file_path"] = p
        self._get_file_path = (get_image_filepath if fkt is None else fkt)

    @property
    def prediction_type(self):
        """Gets the prediction type

        Returns
        -------
        str
            The prediction type.
        """
        return self._prediction_type

    @prediction_type.setter
    def prediction_type(self, value):
        if self.init:
            self.reshuffle = False
            self.set_df = True
            p = self._prediction_type
            if "prediction_type" not in self._changes:
                self._changes["prediction_type"] = p
        # set value
        if value in ["classification", "c"]:
            self._prediction_type = "c"
        elif value in ["regression", "r"]:
            self._prediction_type = "r"
        else:
            raise RuntimeError(
                "Unknown prediction type \"prediction_tpye\""
                "please specify \"c\" or \"classification\" for classification problems or"
                "\"r\" or \"regression\" for regression problems."
            )

    @property
    def force_to_shape(self):
        """Gets the force_to_shape

        Returns
        -------
        list
            The force_to_shape. List containing for each dimension a value to which the images are cropped.

        """
        return self._force_to_shape

    @force_to_shape.setter
    def force_to_shape(self, value):
        if self.init:
            self.reshuffle = False
            self.set_dataset = True
            p = self._force_to_shape
            if "force_to_shape" not in self._changes:
                self._changes["force_to_shape"] = p
        self._force_to_shape = value
        if self.init:
            self.mask = self.mask_path

    @property
    def transforms(self):
        """Gets the transformation

        Returns
        -------
        Compose
            The transformation(s) which shell be applied to the datasets.

        """
        return self._transforms

    @transforms.setter
    def transforms(self, value):
        if self.init:
            self.reshuffle = False
            self.set_dataset = True
            p = self._transforms
            if "transforms" not in self._changes:
                self._changes["transforms"] = p
        self._transforms = value

    @property
    def labels_to_keep(self):
        """Gets the labels to keep

        Returns
        -------
        list/None
            The labels to keep.

        """
        return self._labels_to_keep

    @labels_to_keep.setter
    def labels_to_keep(self, value):
        if self.init:
            self.reshuffle = False
            self.set_task_df = True
            p = self._labels_to_keep
            if "labels_to_keep" not in self._changes:
                self._changes["labels_to_keep"] = p
        self._labels_to_keep = value

    @property
    def balance(self):
        """Gets the balance option.

        Returns
        -------
        bool
            The balance option of the object.
        """
        return self._balance

    @balance.setter
    def balance(self, value):
        if self.init:
            self.reshuffle = False
            self.set_task_df = True
            p = self._balance
            if "balance" not in self._changes:
                self._changes["balance"] = p
        self._balance = value

    @property
    def num_samples(self):
        """Gets the num_samples used to reduce the dataset-size.

        Returns
        -------
        int
            Number of samples used to reduce dataset-size.
        """
        return self._num_samples

    @num_samples.setter
    def num_samples(self, value):
        if self.init:
            self.reshuffle = False
            self.set_split = True
            p = self._num_samples
            if "num_samples" not in self._changes:
                self._changes["num_samples"] = p
        self._num_samples = value

    @property
    def num_training_samples(self):
        """The number of training samples.

        Returns
        -------
        int
            The number of training samples.

        """
        return self._num_training_samples

    @num_training_samples.setter
    def num_training_samples(self, value):
        if self.init:
            self.reshuffle = False
            self.set_split = True
            p = self._num_training_samples
            if "num_training_samples" not in self._changes:
                self._changes["num_training_samples"] = p
        self._num_training_samples = value

    @property
    def grouped(self):
        """Returns the grouped option.

        Returns
        -------
        bool
            The grouped option.

        """
        return self._grouped

    @grouped.setter
    def grouped(self, value):
        if self.init:
            self.reshuffle = False
            self.set_split = True
            p = self._grouped
            if "grouped" not in self._changes:
                self._changes["grouped"] = p
        self._grouped = value

    def _init_wrapper(self, reshuffle=False):
        """ Routine to prepare the dataframe, set the dask dataframe and prepare the splits of the DataBunch object.

        This routine:
        - prepares the dataframe (e.g. reads the original df, checks and extracts labels,
            alters column linking to the images files),
        - sets the task dataframe (e.g. balances labels, removes unwanted
            labels)
        - sets the datasplits (e.g. splits scans patient wise, image wise, stratified, according to arguments),
        - sets the datasets

        """
        # set the dataframe
        self._set_dataframe(source_dir=self.source_dir,
                            table=self.table,
                            prediction_type=self.prediction_type,
                            label_col=self.label_col,
                            file_col=self.file_col,
                            separator=self.separator,
                            image_dir=self.image_dir,
                            get_file_path=self.get_file_path)

        # prepare the dataframe
        self._set_task_dataframe(df_orig=self.df_orig,
                                 labels_to_keep=self.labels_to_keep,
                                 balance=self.balance,
                                 prediction_type=self.prediction_type)

        # build splits
        self._set_datasplits(df=self.df,
                             val_size=self.val_size,
                             hold_out_size=self.hold_out_size,
                             num_samples=self.num_samples,
                             prediction_type=self.prediction_type,
                             num_training_samples=self.num_training_samples,
                             random_state=self.random_state,
                             grouped=self.grouped,
                             reshuffle=reshuffle)

        # build datasets
        self._set_datasets(
            self.df_trn,
            df_val=self.df_val,
            df_ho=self.df_ho,
            label2id=self.label2id,
            # id2label=self.id2label,
            z_factor=self.z_factor,
            force_to_shape=self.force_to_shape,
            mask=self.mask,
            build_hold_out_ds=True if self.hold_out_size is not None else False,
            transforms=self.transforms,
            prediction_type=self.prediction_type,
            reshuffle=reshuffle
        )

    def _set_dataframe(self,
                       source_dir: str,
                       table: str,
                       file_col,
                       label_col,
                       prediction_type: str,
                       separator: str,
                       image_dir: str = None,
                       get_file_path: callable = None):
        """ Prepares the dataframe. Checks for valid image dirs and labels."""
        # read dataset and extract label information
        df = pd.read_csv(os.path.join(source_dir, table), index_col=None, sep=separator)
        print("Found {} images in {}".format(len(df), table))
        if prediction_type == "c":
            lbl = df[label_col].unique().tolist()
            print(
                "Found {} labels: {}".format(len(lbl), lbl)
            )

        # check for existence of FILE in columns - try to correct if not
        if file_col not in df.columns:
            if (
                    get_file_path is not None
                    and image_dir is not None
                    and callable(get_file_path)
            ):
                df[file_col] = df.apply(
                    lambda r: get_file_path(r, image_dir), axis=1
                )
            else:
                raise RuntimeError(
                    "If {} column is not in {},"
                    "please pass a valid 'get_file_path' function and an 'image_dir'.".format(file_col, table)
                )

        # relative paths in column self.FILE! Extend column in csv
        elif image_dir is not None:
            df[file_col] = df.apply(
                lambda r: os.path.join(image_dir, r[file_col]), axis=1
            )

        # store original dataset
        self.df_orig = df

    def _set_task_dataframe(self,
                            df_orig: pd.DataFrame,
                            labels_to_keep: list,
                            balance: bool = False,
                            prediction_type: str = None):
        """Create and sets the task dataframe.

        Downsampling, discarding unwanted labels, balancing, translating class to class IDs based on arguments.

        """
        # prepare dataset
        df = df_orig

        # discard unwanted labels
        if not prediction_type == "r":
            labels_to_keep = (
                df[self.label_col].unique().tolist()
                if labels_to_keep is None
                else labels_to_keep
            )
            df = df[df[self.label_col].isin(labels_to_keep)]
            print(
                "Dropped {} samples that were not in {}".format(len(df_orig) - len(df), labels_to_keep[:20])
            )

        # keep necessary columns only
        df = df[[self.file_col, self.label_col, self.ptid_col]].dropna()

        # balance the data for given label
        if balance:
            if prediction_type == "r":
                logger.warning("Cannot balance dataset for a regression problem. Skipping...")
            else:
                subsample_idx = balanced_subsample(df[self.label_col])
                df = df[df.index.isin(subsample_idx)]
                print("Balanced dataset according to labels: {}".format(labels_to_keep))

        # print information
        print(
            "Final dataframe contains {} samples from {} patients".format(len(df),
                                                                          len(df[self.ptid_col].unique()))
        )

        # get classes
        if prediction_type == "c":
            classes = df[self.label_col].unique().tolist()[::-1]
            label2id = {
                k: v for k, v in zip(classes, np.arange(len(classes)))
            }
            id2label = dict(zip(label2id.values(), label2id.keys()))
        else:
            classes = None
            label2id = None
            id2label = None

        # set results
        self.df = df
        self.classes = classes
        self.label2id = label2id
        self.id2label = id2label

    @classmethod
    def from_csv(cls,
                 folder: str,
                 source_dir: str,
                 path: str,
                 table: str,
                 load_filename: str = None,
                 image_dir: str = None,
                 mask: str = None,
                 transforms: Compose = Compose(
                     [ToTensor(), IntensityRescale(masked=False, on_gpu=True)]
                 ),
                 labels_to_keep: list = None,
                 prediction_type: str = "classification",
                 get_file_path: callable = None,
                 balance: bool = False,
                 num_samples: int = None,
                 num_training_samples: int = None,
                 z_factor: float = 0.5,
                 hold_out_size: float = None,
                 val_size: float = 0.1,
                 grouped: bool = False,
                 cache_filename: str = None,
                 force_to_shape: list = None,
                 file_col="file_path",
                 label_col="DX",
                 ptid_col="PTID",
                 random_state: int = 42,
                 separator: str = ",",
                 **kwargs):
        """" Initializes the object using splits in csv-files.

        Uses CSVs available in 'folder' to initialize databunch object. Uses "X_train.csv", "X_val.csv", "X_hold.csv"
        for initialization.  At least "X_train.csv" *MUST* be available under 'folder'. Prefix "X" can be set
        with parameter 'load_filename'. A source_dir containing the images and specified in 'image_dir' and a csv table
        describing the whole dataset *MUST* be specified too. Moreover, a path *MUST* be set where to store intermediate
        files and results.

        Notes
        -----
            CSVs available in 'folder' are *NOT* checked for correct format. Can lead to unexpected errors!


        Parameters
        ----------
        folder : str
            Path where to find "X_train.csv", "X_val.csv", "X_hold.csv". At least "X_train.csv" necessary for successful
            initialization process.  Prefix "X" can be set with parameter 'load_filename'.
        load_filename : str
            Sets the name - prefix for the CSV files. (e.g. if 'load_filename' = "test", initialization routine will
            look for "test_train.csv", "test_val.csv", "test_hold.csv" files).
            Default: None, corresponding to "databunch"
        source_dir : str
            Path to source_dir folder, where table and image_dir can be found.
        path : str
            Path where intermediary data will be stored (eg. cache).
        table : str
            CSV file path *relative* to source_dir containing samples.
            The tables *must* contain 'file_col', 'label_col' and 'ptid_col' columns.
        image_dir : str
            Image directory *relative* to source_dir, where the .nii files are.
            Set empty string ("") if 'image dir' equals 'source_dir'. Default: None
        mask : str
            Path to binary brain mask in .nii format. This will be resized with z_factor. Default: None
        transforms
            A PyTorch Compose container object, with the transformations to apply to samples.
            Defaults to using ToTensor() and IntensityRescaling() into [0,1].
        labels_to_keep : list
            List of labels to keep in the datasets. Defaults to None (all labels). Default: None
        prediction_type : str
            Either "c" or "classification" for classification problems, or "r" or "regression" for
            regression problems. If regression is specified, labels are taken as they are and not translated to internal
            class IDs. Default: "c"
        get_file_path : callable
            A function mapping the rows of table to the respective file paths of the samples. Default: None
        balance : bool
            Boolean switch for enforcing balanced classes.
            If True dataset will be balanced before sampling (e.g. splitting). Default: False
        num_samples : int
            Total number of samples available for sampling groups (e.g. test, validation, hold-out). Large datasets can
            therefore be reduced in size. Defaults value (None) uses all available images. (e.g. if dataset has
            14.000 datapoints, and 'num_samples'=1000, dataset will be randomly reduced to 1000 samples.)
            Default: None (all)
        num_training_samples : int
            Reduces the number of training samples, but do not reduce size of the whole dataset
            (e.g. validation and hold-out remain large). Usually used for debugging or intensive grid searching.
            Default: None (all)
        z_factor : float
            Zoom factor applied to each image. Default: None
        hold_out_size : float
            Sets the relative size of the hold_out set. Use 'num_samples' to reduce dataset, otherwise 'hold_out_size'
            is relative to the total number of available datapoints. Default: None
        val_size : float
            Sets the relative size of the validation set. Use 'num_samples' to reduce dataset, otherwise 'val_size'
            is relative to the total number of available datapoints. Default: 0.1
        grouped : bool
            Boolean switch for enforcing grouping according to subjectID. Necessary if subjects have more than one
            scan available. Default: False
        cache_filename: str
            Filename specifying a prefix of all stored files. File-extensions in 'cache_filename' will be discarded.
            If not specified "databunch" will be used as default value.
        force_to_shape : list
            List, specifying for each dimension of the data the size. (e.g. 'force_to_shape'=[80,80,80] will
            force images to have the shape of (80,80,80).) Default: None
        file_col : str
            Column name in 'table' identifying the path to the .nii file of the sample. Default: "file_path"
        label_col : str
            Column name in 'table' identifying the path to the label of the sample. Default: "DX"
        ptid_col : str
            Column name in 'table' identifying the path to the patient ID of the sample. Default: "PTID"
        random_state : int
            Random state to enforce reproducibility for train/test splitting. Default: 42
        separator : str
            Separator used to load 'table' and to eventually store csv splits. Default: ","
        kwargs
            Arbitrarily many pairs of (key, value) which will stored in the object.

        Returns
        -------
        DataBunch
            The initialized Databunch object.

        """

        def _load_df(df_path, separator):
            return pd.read_csv(df_path, sep=separator, index_col=0)

        # init
        load_filename = cls.CACHE_NAME if load_filename is None or load_filename == "" \
            else os.path.splitext(load_filename)[0]

        # load the .csv in path
        if os.path.exists(folder):
            # check for csv
            filename_trn = load_filename + "_train.csv"
            filename_val = load_filename + "_val.csv"
            filename_hold = load_filename + "_hold.csv"

            # must have df
            if os.path.exists(os.path.join(folder, filename_trn)):
                df_trn = _load_df(os.path.join(folder, filename_trn), separator=separator)
            else:
                raise FileNotFoundError

            # optional df
            df_val = _load_df(os.path.join(folder, filename_val), separator=separator) if os.path.exists(
                os.path.join(folder, filename_val)) else None
            df_ho = _load_df(os.path.join(folder, filename_hold), separator=separator) if os.path.exists(
                os.path.join(folder, filename_hold)) else None

            # build with attributes from init
            db = cls(
                source_dir=source_dir,
                path=path,
                table=table,
                image_dir=image_dir,
                mask=mask,
                transforms=transforms,
                labels_to_keep=labels_to_keep,
                prediction_type=prediction_type,
                get_file_path=get_file_path,
                balance=balance,
                num_samples=num_samples,
                num_training_samples=num_training_samples,
                z_factor=z_factor,
                hold_out_size=hold_out_size,
                val_size=val_size,
                grouped=grouped,
                cache_filename=cache_filename,
                force_to_shape=force_to_shape,
                file_col=file_col,
                label_col=label_col,
                ptid_col=ptid_col,
                random_state=random_state,
                separator=separator,
                df_trn=df_trn,
                df_val=df_val,
                df_ho=df_ho,
                loaded_cache=True,
                show_stats=False,
                load_h5py=False,
                kwargs=kwargs)

            # init here, do not create datasplits
            db._set_dataframe(source_dir=db.source_dir,
                              table=db.table,
                              prediction_type=db.prediction_type,
                              label_col=db.label_col,
                              file_col=db.file_col,
                              separator=db.separator,
                              image_dir=db.image_dir,
                              get_file_path=db.get_file_path)

            # prepare the dataframe
            db._set_task_dataframe(df_orig=db.df_orig,
                                   labels_to_keep=db.labels_to_keep,
                                   balance=db.balance,
                                   prediction_type=db.prediction_type)

            # build datasets
            db._set_datasets(
                db.df_trn,
                df_val=db.df_val,
                df_ho=db.df_ho,
                label2id=db.label2id,
                # id2label=db.id2label,
                z_factor=db.z_factor,
                force_to_shape=db.force_to_shape,
                mask=db.mask,
                build_hold_out_ds=True if db.hold_out_size is not None else False,
                transforms=db.transforms,
                prediction_type=db.prediction_type,
                reshuffle=False
            )

            return db

        else:
            raise FileExistsError("Cannot find folder {}. No such folder.".format(folder))

    @classmethod
    def from_disk(cls, load_path: str, save_path: str, load_filename: str = None, load_h5py: bool = False):
        """Initializes new object by loading a cached objects dict from disk.

        Parameters
        ----------
        load_path : str
            The path where the cached databunch object is stored.
        save_path : str
            The path which will be used for the new object to store intermediate files and results to.
        load_filename : str
            The filename of the cached object. If not set, default CACHE_NAME will be used (e.g. "databunch")
        load_h5py : bool
            Option to directly load h5py files. Must be available in 'load_path'. Correspond to calling
            function 'load_h5py_ds()' after initialization.
            Default: False

        Returns
        -------
        DataBunch
            The initialized Databunch object.

        """
        # init
        load_filename = cls.CACHE_NAME if load_filename is None or load_filename == "" \
            else os.path.splitext(load_filename)[0]

        # load
        filename = load_filename + ".pkl"

        if os.path.exists(os.path.join(load_path, filename)):
            try:
                print("Trying to load DataBunch object from {}".format(os.path.join(load_path, filename)))
                # load the dictionary
                attr_dict = cls._load(path=load_path, filename=filename, default_cache_name=DataBunch.CACHE_NAME)

                # build with attributes from dictionary
                db = cls(
                    source_dir=attr_dict["_source_dir"],
                    path=save_path,
                    table=attr_dict["table"],
                    image_dir=attr_dict["_image_dir"],
                    mask=attr_dict["_mask"],
                    mask_path=attr_dict["mask_path"],
                    transforms=attr_dict["_transforms"],
                    labels_to_keep=attr_dict["_labels_to_keep"],
                    prediction_type=attr_dict["_prediction_type"],
                    get_file_path=attr_dict["_get_file_path"],
                    balance=attr_dict["_balance"],
                    num_samples=attr_dict["_num_samples"],
                    num_training_samples=attr_dict["_num_training_samples"],
                    z_factor=attr_dict["_z_factor"],
                    hold_out_size=attr_dict["_hold_out_size"],
                    val_size=attr_dict["_val_size"],
                    grouped=attr_dict["_grouped"],
                    cache_filename=attr_dict["_cache_filename"],
                    force_to_shape=attr_dict["_force_to_shape"],
                    file_col=attr_dict["_file_col"],
                    label_col=attr_dict["_label_col"],
                    ptid_col=attr_dict["_ptid_col"],
                    random_state=attr_dict["random_state"],
                    separator=attr_dict["separator"],
                    mean=attr_dict["mean"],
                    std=attr_dict["std"],
                    use_sample=attr_dict["use_sample"],
                    shape=attr_dict["shape"],
                    df_orig=attr_dict["df_orig"],
                    df=attr_dict["df"],
                    df_trn=attr_dict["df_trn"],
                    df_val=attr_dict["df_val"],
                    df_ho=attr_dict["df_ho"],
                    train_ds=attr_dict["train_ds"],
                    val_ds=attr_dict["val_ds"],
                    ho_ds=attr_dict["ho_ds"],
                    classes=attr_dict["classes"],
                    label2id=attr_dict["label2id"],
                    id2label=attr_dict["id2label"],
                    train_dl=attr_dict["train_dl"],
                    val_dl=attr_dict["val_dl"],
                    ho_dl=attr_dict["ho_dl"],
                    loaded_cache=True,
                    load_h5py=load_h5py)

                if db.load_h5py:
                    db.load_h5py_ds()

                # return build object
                return db

            except EOFError:
                logger.warning(
                    "Pickled DataBunch is corrupted at {}".format(load_path)
                )
                print(
                    "Cannot load {} because DataBunch cache is corrupted."
                    " Building Databunch..\n".format(filename)
                )
        else:
            raise FileExistsError("Cannot load file {}. No such file.".format(os.path.join(load_path, filename)))

    def _safe_changes(self):
        """Promts a save question, forcing the user to interact.

        Returns
        -------
        int
            Either 0, if user answered question with "n", or 1 if answered with "y".

        """
        var = input("This will reshuffle groups. Continue?: [y/n]")
        if var == "n":
            return 0
        elif var != "y":
            self._safe_changes()
        return 1

    def reset_changes(self):
        """Resets all changes made to the object after the last sucessful call of 'apply_changes()'."""
        # reset
        for k, v in self._changes.items():
            self.__setattr__(k, v)

        # reset flags
        self.set_df = False
        self.set_task_df = False
        self.set_split = False
        self.set_dataset = False
        self.reshuffle = True
        self._changes = {}

    def apply_changes(self, safe: bool = True):
        """Applies changes to the object. Depending on the changes groups (e.g. test, val, hold) are re-assigned.

        If attributes of the object are altered after initialization this function applies the changes by
        executing necessary routines. E.g. if for example the labels are changed, dataframes, splits and datasets are
        rebuild, whereas a change in the mask attribute only renews the datasets.

        Parameters
        ----------
        safe
            If set, user must agree before changes are made affecting the groups, else not.

        """
        if self.load_h5py:
            print("Cannot apply changes to h5py file. Changes will be reset! Please use drop_h5py first!")
            self.reset_changes()
            return

        if self.set_df:
            self._init_wrapper()
        elif self.set_task_df:
            if safe:
                if not self._safe_changes():
                    self.reset_changes()
                    return

            self._set_task_dataframe(df_orig=self.df_orig,
                                     labels_to_keep=self.labels_to_keep,
                                     balance=self.balance,
                                     prediction_type=self.prediction_type)

            # build splits
            self._set_datasplits(df=self.df,
                                 val_size=self.val_size,
                                 hold_out_size=self.hold_out_size,
                                 num_samples=self.num_samples,
                                 prediction_type=self.prediction_type,
                                 num_training_samples=self.num_training_samples,
                                 random_state=self.random_state,
                                 grouped=self.grouped,
                                 reshuffle=self.reshuffle)

            # build datasets
            self._set_datasets(
                self.df_trn,
                df_val=self.df_val,
                df_ho=self.df_ho,
                label2id=self.label2id,
                z_factor=self.z_factor,
                force_to_shape=self.force_to_shape,
                mask=self.mask,
                build_hold_out_ds=True if self.hold_out_size is not None else False,
                transforms=self.transforms,
                prediction_type=self.prediction_type,
                reshuffle=self.reshuffle)

        elif self.set_split:
            if safe:
                if not self._safe_changes():
                    self.reset_changes()
                    return
            # build splits
            self._set_datasplits(df=self.df,
                                 val_size=self.val_size,
                                 hold_out_size=self.hold_out_size,
                                 num_samples=self.num_samples,
                                 prediction_type=self.prediction_type,
                                 num_training_samples=self.num_training_samples,
                                 random_state=self.random_state,
                                 grouped=self.grouped,
                                 reshuffle=self.reshuffle)

            # build datasets
            self._set_datasets(
                self.df_trn,
                df_val=self.df_val,
                df_ho=self.df_ho,
                label2id=self.label2id,
                z_factor=self.z_factor,
                force_to_shape=self.force_to_shape,
                mask=self.mask,
                build_hold_out_ds=True if self.hold_out_size is not None else False,
                transforms=self.transforms,
                prediction_type=self.prediction_type,
                reshuffle=self.reshuffle)
        elif self.set_dataset:
            self._set_datasets(
                self.df_trn,
                df_val=self.df_val,
                df_ho=self.df_ho,
                label2id=self.label2id,
                z_factor=self.z_factor,
                force_to_shape=self.force_to_shape,
                mask=self.mask,
                build_hold_out_ds=True if self.hold_out_size is not None else False,
                transforms=self.transforms,
                prediction_type=self.prediction_type,
                reshuffle=self.reshuffle)
        else:
            print("No changes to apply for!")
            return

        # reset dataloaders
        self.train_dl = None
        self.val_dl = None
        self.ho_dl = None

        # reset flags
        self.set_df = False
        self.set_task_df = False
        self.set_split = False
        self.set_dataset = False
        self.reshuffle = True
        self._changes = {}

        print("DataBunch successfully changed! Please re-initalize DataLoaders!")

    def _set_datasplits(self,
                        df: pd.DataFrame,
                        val_size: float = 0.1,
                        hold_out_size: float = 0.1,
                        num_samples: int = None,
                        prediction_type: str = "c",
                        num_training_samples: int = None,
                        random_state: int = None,
                        grouped: bool = False,
                        reshuffle: bool = False
                        ):
        """Splits all data based on arguments."""
        # initialize
        df_trn, df_val, df_ho = None, None, None

        if reshuffle:
            print("Reshuffle datasets")
        else:
            print("Building datasets")

        random_state = (
            self.random_state if random_state is None else random_state
        )

        # downsample
        if not reshuffle:
            if num_samples is not None:
                df = df.sample(n=num_samples, random_state=random_state)
                print("Sampling {} samples".format(num_samples))

        # build hold_out set
        if not reshuffle:
            if hold_out_size is not None:
                if grouped:
                    print("Patient-wise creation of hold out set with hold_out_size = {}".format(hold_out_size))
                    gss = GroupShuffleSplit(
                        n_splits=1, test_size=hold_out_size, random_state=random_state
                    )
                    rest, ho = next(
                        iter(gss.split(df, groups=df[self.ptid_col].tolist()))
                    )
                    df_rest, df_ho = df.iloc[rest, :], df.iloc[ho, :]
                    df = df_rest
                else:
                    print("Scan-wise creation of hold out set with hold_out_size = {}".format(hold_out_size))
                    df_ho = df.sample(n=int(len(df) * hold_out_size), random_state=random_state)
                    # drop those sampled from the dataset as they shell not appear in any other set
                    df = df.drop(index=df_ho.index)

        # sample train and test set if specified
        if val_size is not None:
            if grouped:
                print("Patient-wise train/test splitting with val_size = {}".format(val_size))
                gss = GroupShuffleSplit(
                    n_splits=1, test_size=val_size, random_state=random_state
                )
                trn, tst = next(
                    iter(gss.split(df, groups=df[self.ptid_col].tolist()))
                )
                df_trn, df_val = df.iloc[trn, :], df.iloc[tst, :]
            else:
                if prediction_type == "c":
                    print("Scan-wise stratified train/test splitting with val_size = {}".format(val_size))
                    df_trn, df_val = train_test_split(
                        df,
                        test_size=val_size,
                        stratify=df[self.label_col],
                        random_state=random_state,
                        shuffle=True,
                    )
                # in case of regression do not look at the relative sizes of the "classes" e.g. no straitify
                elif prediction_type == "r":
                    print("Scan-wise train/test splitting with val_size = {}".format(val_size))
                    df_trn, df_val = train_test_split(
                        df,
                        test_size=val_size,
                        stratify=None,
                        shuffle=True,
                        random_state=random_state,
                    )
                logger.warning(
                    "A patient having scans at multiple time points might appear in the val and train split!")
        else:
            df_trn = df

        if num_training_samples is not None:
            df_trn = df_trn.sample(n=num_training_samples)
            logger.info("Sampling {} training samples".format(num_training_samples))

        # update
        self.df, self.df_trn, self.df_val = df, df_trn, df_val

        # only update when not in reshuffling modus
        if not reshuffle:
            self.df_ho = df_ho

    def _set_datasets(
            self,
            df_trn: pd.DataFrame,
            df_val: pd.DataFrame = None,
            df_ho: pd.DataFrame = None,
            label2id: dict = None,
            # id2label: dict = None,
            z_factor: float = 0.5,
            force_to_shape: list = None,
            mask: np.ndarray = None,
            build_hold_out_ds: bool = True,
            transforms: list = None,
            prediction_type: str = "c",
            reshuffle: bool = False
    ):
        """Builds the datasets."""

        train_ds, val_ds, ho_ds = None, None, None

        if prediction_type == "c":
            label_trn = [label2id[l] for l in df_trn[self.label_col]]
            label_tst = [label2id[l] for l in df_val[self.label_col]]
            if not reshuffle and self.hold_out_size is not None:
                label_ho = [label2id[l] for l in df_ho[self.label_col]]
        else:  # take the labels as they are
            label_trn = pd.Series.tolist(df_trn[self.label_col])
            label_tst = pd.Series.tolist(df_val[self.label_col])
            if not reshuffle and build_hold_out_ds:
                label_ho = pd.Series.tolist(df_ho[self.label_col])

        train_ds = MRIDataset(
            df_trn[self.file_col].tolist(),
            label_trn,
            # id2label=id2label,
            z_factor=z_factor,
            transform=transforms,
            force_to_shape=force_to_shape,
            mask=mask
        )
        val_ds = MRIDataset(
            df_val[self.file_col].tolist(),
            label_tst,
            # id2label=id2label,
            z_factor=z_factor,
            transform=transforms,
            force_to_shape=force_to_shape,
            mask=mask
        )
        if not reshuffle:
            if build_hold_out_ds:
                ho_ds = MRIDataset(
                    df_ho[self.file_col].tolist(),
                    label_ho,
                    # id2label=id2label,
                    z_factor=z_factor,
                    transform=transforms,
                    force_to_shape=force_to_shape,
                    mask=mask
                )

        # update attributes
        self.shape = train_ds.shape
        self.train_ds, self.val_ds = train_ds, val_ds
        # only update ho_ds if not in reshuffling modus
        if not reshuffle:
            self.ho_ds = ho_ds

    def drop_h5py(self, safe: bool = True):
        """ Switches from H5pyDataset to MRIDataset. This re-assignes the datasets, including hold-out set.

        Parameters
        ----------
        safe : bool
            If set, user must agree before changes are made affecting the groups, else not.

        """
        if safe:
            print("This will re-assignes the datasets, including the hold-out set.")
            if not self._safe_changes():
                return

        # delete attributes
        self.shape = None
        self.train_ds, self.val_ds, self.ho_ds = None, None, None

        # set new datasets
        self._set_datasets(self.df_trn,
                           df_val=self.df_val,
                           df_ho=self.df_ho,
                           label2id=self.label2id,
                           # id2label=self.id2label,
                           z_factor=self.z_factor,
                           force_to_shape=self.force_to_shape,
                           mask=self.mask,
                           build_hold_out_ds=True if self.hold_out_size is not None else False,
                           transforms=self.transforms,
                           prediction_type=self.prediction_type,
                           reshuffle=False)

        # build new dataloaders - if required
        bs = self.train_dl.batch_size if self.train_dl is not None else None
        nw = self.train_dl.num_workers if self.train_dl is not None else None
        norm_flag = True if self.mean is not None else False
        if bs is not None:
            self.build_dataloaders(bs=bs,
                                   normalize=norm_flag,
                                   use_samples=self.use_sample,
                                   num_workers=nw)

    def _set_new_rand_state(self):
        new_rs = np.random.randint(pow(2, 32) - 1)
        if new_rs != 0 and new_rs != self.random_state:
            self.random_state = new_rs

    # reshuffles the datasets
    def reshuffle_datasets(self):
        """Reshuffles the datasets.

        Reshuffles the datasets, thereby reassigning subjects in group train and test, but keeping subjects in
        the hold out dataset (if any). Renews the normalization process for the new training group.

        """
        if self.load_h5py:
            logger.warning("Cannot reshuffle datasets when using h5py files. Use \"drop_h5py\" first.")

        else:
            # get information from old dataloaders:
            bs = self.train_dl.batch_size if self.train_dl is not None else None
            nw = self.train_dl.num_workers if self.train_dl is not None else None
            norm_flag = True if self.mean is not None else False

            # we need to set a new random state - otherwise it wouldn't be reshuffling
            self._set_new_rand_state()

            self._set_datasplits(df=self.df,
                                 val_size=self.val_size,
                                 hold_out_size=self.hold_out_size,
                                 num_samples=self.num_samples,
                                 prediction_type=self.prediction_type,
                                 num_training_samples=self.num_training_samples,
                                 random_state=self.random_state,
                                 grouped=self.grouped,
                                 reshuffle=True)

            # reshuffle
            self._set_datasets(
                self.df_trn,
                df_val=self.df_val,
                df_ho=self.df_ho,
                label2id=self.label2id,
                # id2label=self.id2label,
                z_factor=self.z_factor,
                force_to_shape=self.force_to_shape,
                mask=self.mask,
                build_hold_out_ds=True if self.hold_out_size is not None else False,
                transforms=self.transforms,
                prediction_type=self.prediction_type,
                reshuffle=True
            )
            self.print_stats()
            # build new dataloaders
            if bs is not None:
                self.build_dataloaders(bs=bs,
                                       normalize=norm_flag,
                                       use_samples=self.use_sample,
                                       num_workers=nw)

    def normalize(self, use_samples: int = None, show_progress: bool = False):
        """Normalizes the dataset with mean and std calculated on the training set.

        Parameters
        ----------
        use_samples : int
            Number of samples included in the normalization process. Default: None (all)
        show_progress : int
            If set, shows the progress of the normalization process. Default: False

        """

        if not hasattr(self, "train_ds"):
            raise RuntimeError("Attribute 'train_ds' not found.")
        print("Normalizing datasets")

        if use_samples is None:
            use_samples = len(self.train_ds)
        else:
            use_samples = (
                len(self.train_ds)
                if use_samples > len(self.train_ds)
                else use_samples
            )
        print(
            "Calculating mean and std for normalization based on {} train samples:".format(use_samples)
        )
        self.train_ds.fit_normalization(num_sample=use_samples, show_progress=show_progress)
        # set calculated mean and std
        self.mean, self.std = self.train_ds.mean, self.train_ds.std
        self.train_ds.transform = None
        if self.val_ds is not None:
            self.val_ds.mean, self.val_ds.std = self.mean, self.std
            self.val_ds.transform = None
        if self.ho_ds is not None:
            self.ho_ds.mean, self.ho_ds.std = self.mean, self.std
            self.ho_ds.transform = None

    def build_dataloaders(
            self,
            bs: int = 8,
            normalize: bool = False,
            use_samples: int = None,
            num_workers: int = None,
            show_progress: bool = False
    ):
        """Build DataLoaders.

         Builds DataLoaders with batch-size (bs), optionally normalizing the datasets.
         Optionally performs downsampling to reduce normalization processing time.

        Parameters
        ----------
        bs : int
            Number of images to build a batch. Default: 8
        normalize : bool
            If enabled, datasets will be normalized. Default: False
        use_samples : int
            Number of samples included in the normalization process. Default: None (all)
        num_workers
            Number of workers. Default: None (1)
        show_progress : bool
            If set, shows the progress of the normalization process. Default: False

        """
        print("Building dataloaders")

        self.use_sample = use_samples

        if normalize:
            if self.mean is not None:
                print(
                    "Already normalized -- using attributes 'mean' and 'std'."
                )
            else:
                self.normalize(use_samples=self.use_sample, show_progress=show_progress)

        else:
            logger.warning(
                "Dataset not normalized, performance might be significantly hurt!"
            )
        print(
            "No. training/test samples: {}".format(len(self.train_ds) / len(self.val_ds))
        )

        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
        pin_memory = torch.cuda.is_available()
        self.train_dl = DataLoader(
            self.train_ds,
            batch_size=bs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.val_dl = DataLoader(
            self.val_ds,
            batch_size=bs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if self.hold_out_size is not None:
            self.ho_dl = DataLoader(
                self.ho_ds,
                batch_size=1,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

    def save_df(self, filename: str = None, separator: str = ","):
        """Saves dataframes used to define datasets as CSV files to disk.

        Saves dataframes used to define datasets as CSV files to disk. Filename is set according to function
        arguments or default object attributes.


        Parameters
        ----------
        filename : str
            Prefix for the filename. (e.g. if 'filename'="test", output files will be named "test_train.csv"
            "test_val.csv", "test_hold.csv")
            Default: 'filename'= None = "databunch".
        separator : str
            Separator used to create CSV files. Default: ","

        """

        filename = self._get_filename(filename, "")
        filename_ending = ["_train.csv", "_val.csv", "_hold.csv"]
        for idx, df in enumerate([self.df_trn, self.df_val, self.df_ho]):
            if df is not None:
                df.to_csv(os.path.join(self.path, filename + filename_ending[idx]), sep=separator)

    def save_h5py(self, filename: str = None):
        """Saves image and label data-matrix to disk using h5py file format.

        Saves image and label data-matrix to disk using h5py file format. Filename is set according to function
        arguments or default object attributes.

        Notes
        -----
            This might take a lot time!

        Parameters
        ----------
        filename : str
            Prefix for the filename. (e.g. if 'filename'="test", output files will be named "test_train.h5py"
            "test_val.h5py", "test_hold.h5py")
            Default: 'filename'= None = "databunch".

        """

        # save DataBunch
        self.save(filename=filename)

        filename = self._get_filename(filename, "")
        filename_ending = ["_train.h5py", "_val.h5py", "_hold.h5py"]

        print("Saving datasets as h5py...")

        for idx, ds in enumerate([self.train_ds, self.val_ds, self.ho_ds]):
            if ds is not None:

                # initialize empty numpy array
                mat = np.empty(np.append(np.array([len(ds)]), self.shape))
                lab = np.empty(np.array([len(ds)]))

                # load each scan in matrix
                scan_idx = 0
                for scan, lbl in ds:
                    mat[scan_idx] = scan
                    lab[scan_idx] = lbl
                    scan_idx += 1

                # save matrix on the disk
                h5 = h5py.File(os.path.join(self.path, filename + filename_ending[idx]), 'w')
                h5.create_dataset('X', data=mat, compression='gzip', compression_opts=9)
                h5.create_dataset('y', data=lab, compression='gzip', compression_opts=9)
                h5.close()

                print("File \"{}\" successfully saved".format(os.path.join(self.path, filename + filename_ending[idx])))

        print("Done saving datasets.")

    def load_h5py_ds(self, filename: str = None):
        """Loads a cached H5pyDataset.

        Loads a cached H5pyDataset. Filename is set according to function arguments or default object attributes.

        Parameters
        ----------
        filename : str
            Prefix for the filename. (e.g. if 'filename'="test", function will look for files which are named
            "test_train.h5py","test_val.h5py", "test_hold.h5py")
            Default 'filename'= None = "databunch".

        """

        # initialize
        filename = self._get_filename(filename, "")
        train_h5ds, test_h5ds, hold_h5ds = None, None, None

        train_h5ds = self._load_h5py_ds(os.path.join(self.path, filename + "_train.h5py"), self.transforms, self.mask)
        val_h5ds = self._load_h5py_ds(os.path.join(self.path, filename + "_val.h5py"), self.transforms, self.mask)
        if self.hold_out_size is not None:
            hold_h5ds = self._load_h5py_ds(os.path.join(self.path, filename + "_hold.h5py"), self.transforms, self.mask)

        # update dataset
        self.train_ds, self.val_ds, self.ho_ds = train_h5ds, val_h5ds, hold_h5ds

        # set h5py_flag
        self.load_h5py = True

        # update dataloaders - eventually
        if self.train_dl is not None:
            norm_flag = True if self.mean is not None else False

            self.build_dataloaders(bs=self.bs, normalize=norm_flag, use_samples=self.use_sample)

    @staticmethod
    def _load_h5py_ds(file, transform=None, mask=None, dtype=None) -> H5pyDataset:
        """Loads h5py file from disk.

        Returns
        -------
        H5pyDataset
            The loaded H5pyDataset.

        """
        if os.path.exists(file):
            try:
                ds_h5 = h5py.File(file, 'r')
                X_ds, y_ds = ds_h5['X'], ds_h5['y']
                h5py_ds = H5pyDataset(X_ds, y_ds, transform=transform, mask=mask, dtype=dtype)
                return h5py_ds
            except EOFError:
                logger.warning("H5py file is corrupted at {}"
                               .format(os.path.splitext(file)[0]))
                print("Cannot load {} because H5py cache is corrupted."
                      .format(os.path.splitext(file)[0]))

    def print_stats(self):
        """Print statistics about the patients and images."""
        headers = []
        headers.append("IMAGES")
        if self.prediction_type == "r":
            lbl_tmp = self.df[self.label_col].unique()
            hist, bin_edges = np.histogram(lbl_tmp, bins='auto')
            p = bin_edges[0]
            for s in bin_edges[1:]:
                headers.append(str(np.round(p, 2)) + " - " + str(np.round(s, 2)))
                p = s
        else:
            headers += [cls for cls in self.classes]
        headers.append("PATIENTS")
        if self.prediction_type == "r":
            hist, bin_edges = np.histogram(self.df[self.label_col].unique(), bins='auto')
            p = bin_edges[0]
            for s in bin_edges[1:]:
                headers.append(str(np.round(p, 2)) + " - " + str(np.round(s, 2)))
                p = s
        else:
            headers += [cls for cls in self.classes]

        if self.hold_out_size is not None:
            stats = [
                ["Train"] + self._get_stats(self.df_trn, self.prediction_type, self.label_col, self.ptid_col,
                                            self.classes),
                ["Val"] + self._get_stats(self.df_val, self.prediction_type, self.label_col, self.ptid_col,
                                          self.classes),
                ["Hold"] + self._get_stats(self.df_ho, self.prediction_type, self.label_col, self.ptid_col,
                                           self.classes),
                ["Total"] + self._get_stats(self.df, self.prediction_type, self.label_col, self.ptid_col, self.classes),
            ]
        else:
            stats = [
                ["Train"] + self._get_stats(self.df_trn, self.prediction_type, self.label_col, self.ptid_col,
                                            self.classes),
                ["Val"] + self._get_stats(self.df_val, self.prediction_type, self.label_col, self.ptid_col,
                                          self.classes),
                ["Total"] + self._get_stats(self.df, self.prediction_type, self.label_col, self.ptid_col, self.classes),
            ]
        print(tabulate(stats, headers=headers))
        print()
        print("Data shape: {}".format(self.train_ds.shape))
        if self.z_factor is not None:
            print(
                "NOTE: data have been downsized by a factor of {}".format(self.z_factor)
            )

    @staticmethod
    def _get_stats(df: pd.DataFrame, prediction_type: str, label_col, ptid_col, classes):
        """Returns basic statistic (number of images and number of patients) from a pandas dataset.

        Returns
        -------
        list
            Basic statistics about the dataframe.

        """
        image_count, patient_count = (
            [len(df)],
            [len(df[ptid_col].unique())],
        )
        if prediction_type == "r":
            # get histogram bin_size and determine statistics
            _, bin_edges = np.histogram(df[label_col].unique(), bins='auto')
            p = bin_edges[0]
            for s in bin_edges[1:]:
                image_count.append(len(df[(df[label_col] > p) & (df[label_col] <= s)]))
                patient_count.append(len(df[(df[label_col] > p) & (df[label_col] <= s)][ptid_col].unique()))
                p = s
        else:
            image_count += [
                len(df[df[label_col] == cls]) for cls in classes
            ]
            patient_count += [
                len(df[df[label_col] == cls][ptid_col].unique()) for cls in classes
            ]

        return image_count + patient_count

    def show_sample(self, cmap="gray"):
        """Shows a random training sample after zooming, masking and tranformations.

        Parameters
        ----------
        cmap : str
            Color map which should be applied to the plot. Default: "gray"

        """

        if self.train_ds is None:
            raise RuntimeError(
                "'train_ds' not found, please call 'build' method first."
            )
        img, lbl = self.train_ds[np.random.randint(0, len(self.train_ds))]
        if self.prediction_type == "c":
            label = self.id2label[lbl.item()]
        else:
            label = lbl
        print("label={}".format(label))
        _ = show_brain(img[0].numpy(), cmap=cmap)
        plt.show()

    def save(self, filename: str = None):
        """Cache the entire DataBunch object.

        Caches the entire DataBunch object to 'path'. Filename is set according to function arguments or default
        object attributes.

        Parameters
        ----------
        filename : str
            Prefix for the filename. (e.g. if 'filename'="test", output file will be named "test.pkl").
            Default is 'filename'= None = "databunch".

        """

        filename = self._get_filename(filename, ".pkl")

        with open(os.path.join(self.path, filename), "wb", closefd=True) as file:
            pickle.dump(self.__dict__, file, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved DataBunch to {}".format(os.path.join(self.path, filename)))

    def load(self, filename: str = None):
        """Loads a cached DataBunch object from disk.

        Loads cached DataBunch object from 'path'. Filename is set according to function arguments or default
        object attributes.

        Parameters
        ----------
        filename : str
            Prefix for the filename. (e.g. if 'filename'="test", output file will be named "test.pkl").
            Default is 'filename'="databunch".

        """

        filename = self._get_filename(filename, ".pkl")

        with open(os.path.join(self.path, filename), "rb") as file:
            tmp_dict = pickle.load(file)
            self.__dict__.update(tmp_dict)
        print("Cached DataBunch \"{}\" has been successfully loaded.".format(filename))

    def _get_filename(self, filename: str = None, extension: str = ""):
        """Sets the filename according to specified function arguments or default object attributes.

         If specified, adds an extension to the filename, otherwise returns filename without extension.

         Returns
         -------
         str
            The filename.

        """

        if filename is None:
            if self.cache_filename is None:
                filename = self.CACHE_NAME + extension
            else:
                filename = os.path.splitext(self.cache_filename)[0] + extension
        else:
            filename = os.path.splitext(filename)[0] + extension
        return filename

    @staticmethod
    def _load(default_cache_name: str, path: str = None, filename: str = None):
        """Loads and returns the dictionary of the pickle object under path.

        Returns
        -------
        dict
            The dictionary of the loaded object.

        """
        filename = default_cache_name + ".pkl" if filename is None else os.path.splitext(filename)[0] + ".pkl"
        with open(os.path.join(path, filename), "rb") as file:
            tmp_dict = pickle.load(file)
        return tmp_dict


def show_brain(
        img,
        cut_coords=None,
        figsize=(10, 5),
        cmap="nipy_spectral",
        draw_cross=True,
        return_fig=False,
):
    """Displays 2D cross-sections of a 3D image along all 3 axis.

    Parameters
    ----------
    img : numpy.ndarray/nibabel.NiftiImage/str
         either a 3-dimensional numpy.ndarray, a nibabel.Nifti1Image object or a path to the image file
         stored in nifTI format.
    cut_coords
        The voxel coordinates of the axes where the cross-section cuts will be performed.
        Should be a 3-tuple: (x, y, z). Default is the center = img_shape/2
    figsize
        matplotlib figsize. Default: (10,5).
    cmap
        matplotlib colormap to be used. Default: "nipy_spectral"
    draw_cross
        Draws horizontal and vertical lines which show where the cross-sections have been performed. Default: True
    return_fig
        Additionally retunrs the figure when set. Default: False

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.

    """

    if isinstance(img, str) and os.path.isfile(img):
        img_arr = load_nifti(img)
    elif isinstance(img, nibabel.Nifti1Image):
        img_arr = img.get_data()

    elif isinstance(img, np.ndarray):
        assert (
                img.ndim == 3
        ), "The numpy.ndarray must be 3-dimensional with shape (H x W x Z)"
        img_arr = img
    else:
        raise TypeError(
            "Invalid type provided for 'img'- {}. "
            "Either provide a 3-dimensional numpy.ndarray of a MRI image or path to "
            "the image file stored as a nifTI format.".format(
                type(img)
            )
        )

    # print(img_arr.shape)
    # img_arr = np.moveaxis(img_arr, 0, 1)
    # print(img_arr.shape)

    x_len, y_len, z_len = img_arr.shape
    # if cut_coordinates is not specified set it to the center of the image
    if cut_coords == None:
        cut_coords = (x_len // 2, y_len // 2, z_len // 2)

    f, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)

    ax[0].set_title("Saggital cross-section at x={}".format(cut_coords[0]))
    ax[0].imshow(
        np.rot90(img_arr[cut_coords[0], :, :]), cmap=cmap, aspect="equal"
    )
    # draw cross
    if draw_cross:
        ax[0].axvline(x=cut_coords[1], color="k", linewidth=1)
        ax[0].axhline(y=cut_coords[2], color="k", linewidth=1)

    ax[1].set_title("Coronal cross-section at y={}".format(cut_coords[1]))
    ax[1].imshow(
        np.rot90(img_arr[:, cut_coords[1], :]), cmap=cmap, aspect="equal"
    )
    ax[1].text(
        0.05,
        0.95,
        "L",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax[1].transAxes,
        bbox=dict(facecolor="white"),
    )
    ax[1].text(
        0.95,
        0.95,
        "R",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax[1].transAxes,
        bbox=dict(facecolor="white"),
    )
    # draw cross
    if draw_cross:
        ax[1].axvline(x=cut_coords[0], color="k", linewidth=1)
        ax[1].axhline(y=cut_coords[2], color="k", linewidth=1)

    ax[2].set_title("Axial cross-section at z={}".format(cut_coords[2]))
    ax[2].imshow(
        np.rot90(img_arr[:, :, cut_coords[2]]), cmap=cmap, aspect="equal"
    )
    ax[2].text(
        0.05,
        0.95,
        "L",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax[2].transAxes,
        bbox=dict(facecolor="white"),
    )
    ax[2].text(
        0.95,
        0.95,
        "R",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax[2].transAxes,
        bbox=dict(facecolor="white"),
    )
    # draw cross
    if draw_cross:
        ax[2].axvline(x=cut_coords[0], color="k", linewidth=1)
        ax[2].axhline(y=cut_coords[1], color="k", linewidth=1)

    plt.tight_layout()
    if return_fig:
        return f
