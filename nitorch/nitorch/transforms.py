import numpy as np
import numbers
import torch
from scipy.ndimage.interpolation import rotate


def normalize_float(ndarr, min=-1):
    """Performs min-max normalization on a `numpy.ndarray` matrix.

    Parameters
    ----------
    ndarr : numpy.ndarray
        The numpy array to normalize
    min : int/float
        Either `-1` or `0`. Default: -1

    Returns
    -------
    norm : numpy.ndarray
        The min-max-normalization of the input matrix

    """
    norm = ndarr

    if min == -1:
        norm = (2 * (ndarr - np.min(ndarr)) / (np.max(ndarr) - np.min(ndarr))) - 1
    elif min == 0:
        if not (np.max(ndarr) == 0 and np.min(ndarr) == 0):
            norm = (ndarr - np.min(ndarr)) / (np.max(ndarr) - np.min(ndarr))

    return norm


def normalize_float_torch(x_tensor, min=-1):
    """Performs min-max normalization on a Pytorch tensor matrix.

    Notes
    -----
        Can also deal with Pytorch dictionaries where the data matrix key is 'image'.

    Parameters
    ----------
    ndarr : numpy.ndarray
        The numpy array to normalize
    min : int/float
        Either `-1` or `0`. Default: -1

    Returns
    -------
    norm : numpy.ndarray
        The min-max-normalization of the input matrix

    """
    import torch

    if min == -1:
        norm = (2 * (x_tensor - torch.min(x_tensor)) / (torch.max(x_tensor) - torch.min(x_tensor))) - 1
    elif min == 0:
        if torch.max(x_tensor) == 0 and torch.min(x_tensor) == 0:
            norm = x_tensor
        else:
            norm = (x_tensor - torch.min(x_tensor)) / (torch.max(x_tensor) - torch.min(x_tensor))
    return norm


def normalization_factors(data, train_idx, shape, mode="slice"):
    """Computes normalization factors for the data.

    Parameters
    ----------
    data : numpy.ndarray
        The image data
    train_idx : numpy.ndarray/list
        Training indices.
    shape
        Shape of the image data. Expected to be 3 dimensional.
    mode : str
        Either "slice" or "voxel". Defines the granularity of the normalization.
        Voxelwise normalization does not work well with linear registered data only. Default: "slice"

    Raises
    ------
    NotImplementedError
        Unknown mode selected.

    """
    print("Computing the normalization factors of the training data..")
    if mode == "slice":
        axis = (0, 1, 2, 3)
    elif mode == "voxel":
        axis = 0
    else:
        raise NotImplementedError("Normalization mode unknown.")
    samples = np.zeros(
        [len(train_idx), 1, shape[0], shape[1], shape[2]], dtype=np.float32
    )
    for c, value in enumerate(train_idx):
        samples[c] = data[value]["image"].numpy()
    mean = np.mean(samples, axis=axis)
    std = np.std(samples, axis=axis)
    return np.squeeze(mean), np.squeeze(std)


class CenterCrop(object):
    """Crops the given 3D numpy.ndarray Image at the center.

    Parameters
    ----------
    size : sequence/int
        Desired output size of the crop. If size is an int instead of sequence like (h, w, d),
        a cube crop (size, size, size) is made.

    Attributes
    ----------
    size  : sequence/int
        Desired output size of the crop. If size is an int instead of sequence like (h, w, d),
        a cube crop (size, size, size) is made.


    """

    def __init__(self, size):
        """Initialization routine.

        Raises
        ------
        AssertionError
            If size is not a tuple of length 3.

        """

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size), int(size))
        else:
            self.size = np.asarray(size)
        assert (len(self.size) == 3), "The `size` must be a tuple of length 3 but is length {}".format(
            len(self.size)
        )

    def __call__(self, img):
        """Calling routine.

        Parameters
        ----------
        img : numpy.ndarray
            Image to be cropped.

        Returns
        -------
        numpy.ndarray
            Cropped image.

        Raises
        ------
        ValueError
            Shape of the image is not 4d or 3d.

        """
        # if the 4th dimension of the image is the batch then ignore that dim
        if len(img.shape) == 4:
            img_size = img.shape[1:]
        elif len(img.shape) == 3:
            img_size = img.shape
        else:
            raise ValueError(
                "The size of the image can be either 3 dimension or 4\
                dimension with one dimension as the batch size"
            )

        # crop only if the size of the image is bigger than the size to be cropped to.
        if all(img_size >= self.size):
            slice_start = (img_size - self.size) // 2
            slice_end = self.size + slice_start
            cropped = img[
                slice_start[0]: slice_end[0],
                slice_start[1]: slice_end[1],
                slice_start[2]: slice_end[2],
            ]
            if len(img.shape) == 4:
                cropped = np.expand_dims(cropped, 0)
        else:
            cropped = img

        return cropped

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


class Normalize(object):
    """Normalize tensor with first and second moments.

    Notes
    -----
        By default will only normalize on non-zero voxels. Set
        masked = False if this is undesired.

    Parameters
    ----------
    mean : float
        Mean of the training data.
    std : float
        Standard deviation of the training data. Default: 1
    masked : bool
        Boolean switch. If True, non-zero voxels will not be normalized. Enable with value False. Default: True
    eps : float
        Only set to scale std. Otherwise leave untouched. Default:  1e-10

    Attributes
    ----------
    mean : float
        Mean of the training data.
    std : float
        Standard deviation of the training data.
    masked : bool
        Boolean switch. If True, non-zero voxels will not be normalized. Enable with value False.
    eps : float
        Only set to scale std. Otherwise leave untouched.

    """
    def __init__(self, mean, std=1, masked=True, eps=1e-10):
        """Initialization routine



        """
        self.mean = mean
        self.std = std
        self.masked = masked
        # set epsilon only if using std scaling
        self.eps = eps if np.all(std) != 1 else 0

    def __call__(self, image):
        """Calling procedure.

        Parameters
        ----------
        image : torch.tensor/numpy.ndarray
            The image which shall be normalized.

        Returns
        -------
        image : torch.tensor/numpy.ndarray
            The normalized image.

        """
        if self.masked:
            image = self.zero_masked_transform(image)
        else:
            image = self.apply_transform(image)
        return image

    def denormalize(self, image):
        """Undo normalization procedure.

        Parameters
        ----------
        image : torch.tensor/numpy.ndarray
            The image to reverse normalization for.

        Returns
        -------
        image : torch.tensor/numpy.ndarray
            De-normalized image

        """
        image = image * (self.std + self.eps) + self.mean
        return image

    def apply_transform(self, image):
        """Applies normalization to the image by using object attributes.

        Parameters
        ----------
        image : torch.tensor/numpy.ndarray
            The image to normalize.

        Returns
        -------
        image : torch.tensor/numpy.ndarray
            Normalized image.

        """
        return (image - self.mean) / (self.std + self.eps)

    def zero_masked_transform(self, image):
        """Apply normalization transformation for non-zero voxels only.

        Parameters
        ----------
        image : torch.tensor/numpy.ndarray
            The image to normalize.

        Returns
        -------
        image : torch.tensor/numpy.ndarray
            Normalized image.

        """
        img_mask = image == 0
        # do transform
        image = self.apply_transform(image)
        image[img_mask] = 0.0
        return image


class IntensityRescale:
    """Rescale image intensities between 0 and 1 for a single image.

    Parameters
    ----------
    masked : bool
        applies normalization only on non-zero voxels. Default: True.
    on_gpu : bool
        speed up computation by using GPU. Requires torch.Tensor instead of np.array. Default: False.

    Attributes
    ----------
    masked : bool
        applies normalization only on non-zero voxels.
    on_gpu : bool
        speed up computation by using GPU. Requires torch.Tensor instead of np.array.

    """

    def __init__(self, masked=True, on_gpu=False):
        """Initialization process."""

        self.masked = masked
        self.on_gpu = on_gpu

    def __call__(self, image):
        """Calling procedure

        Parameters
        ----------
        image  : torch.tensor/numpy.ndarray
            Image to transform.

        Returns
        -------
         image : torch.tensor/numpy.ndarray
            Transformed image.

        """
        if self.masked:
            image = self.zero_masked_transform(image)
        else:
            image = self.apply_transform(image)

        return image

    def apply_transform(self, image):
        """Applys tranformation to input.

        Parameters
        ----------
        image : torch.tensor/numpy.ndarray
            The image to transform.

        Returns
        -------
        torch.tensor/numpy.ndarray
            Transformed image.

        """
        if self.on_gpu:
            return normalize_float_torch(image, min=0)
        else:
            return normalize_float(image, min=0)

    def zero_masked_transform(self, image):
        """ Only apply transform where input is not zero.

        Parameters
        ----------
        image
            The image to transform.

        Returns
        -------
        image
            Transformed image.

        """
        img_mask = image == 0
        # do transform
        image = self.apply_transform(image)
        image[img_mask] = 0.0
        return image


########################################################################
# Data augmentations
########################################################################


class ToTensor(object):
    """Convert numpy.ndarrays to Tensors.

    Notes
    -----
        Expands channel axis.

    Parameters
    ----------
    image : numpy.ndarray
        numpy.ndarray of input with dimensions H x W x Z will be transformed
        to torch.tensor of dimensions  C x H x W x Z

    Attributes
    ----------
    image : numpy.ndarray
        numpy.ndarray of input with dimensions H x W x Z will be transformed
        to torch.tensor of dimensions  C x H x W x Z

    """

    def __call__(self, image):
        """Calling routine.

        Returns
        -------
        torch.tensor
            The image as torch.tensor

        """
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.float()
        return image


class Flip:
    """Flip the input along a given axis.

    Parameters
    ----------
    axis
        axis to flip over. Default: 0.
    prob
        probability to flip the image. Executes always when set to 1. Default: 0.5

    Attributes
    ----------
    axis
        axis to flip over. Default is 0.
    prob
         probability to flip the image. Executes always when set to 1. Default: 0.5

    """

    def __init__(self, axis=0, prob=0.5):
        """Initialization routine."""
        self.axis = axis
        self.prob = prob

    def __call__(self, image):
        """Calling routine.

        Parameters
        ----------
        image : numpy.ndarray
            The image to flip.

        Returns
        -------
        numpy.ndarray
            The flipped image.

        """
        rand = np.random.uniform()
        if rand <= self.prob:
            augmented = np.flip(image, axis=self.axis).copy()
        else:
            augmented = image
        return augmented


class SagittalFlip(Flip):
    """Flip image along the sagittal axis (x-axis).

    Notes
    -----
        Expects input shape (X, Y, Z).

    Parameters
    ----------
    prob : float
        The probability the flip happens. Default: 0.5

    Attributes
    ----------
    prob : float
        The probability the flip happens.

    """

    def __init__(self, prob=0.5):
        """Initialization routine."""
        super().__init__(axis=0, prob=prob)

    def __call__(self, image):
        """Calling routine

        Parameters
        ----------
        image : numpy.ndarray
            The image to flip.

        Returns
        -------
        numpy.ndarray
            The flipped image.

        """
        assert len(image.shape) == 3
        return super().__call__(image)


class CoronalFlip(Flip):
    """Flip image along the coronal axis (y-axis).

    Notes
    -----
        Expects input shape (X, Y, Z).

    Parameters
    ----------
    prob : float
        The probability the flip happens. Default: 0.5


    Attributes
    ----------
    prob : float
        The probability the flip happens.

    """

    def __init__(self, prob=0.5):
        """Initialization routine."""
        super().__init__(axis=1, prob=prob)

    def __call__(self, image):
        """Calling routine

        Parameters
        ----------
        image : numpy.ndarray
            The image to flip.

        Returns
        -------
        numpy.ndarray
            The flipped image.

        """
        assert len(image.shape) == 3
        return super().__call__(image)


class AxialFlip(Flip):
    """Flip image along the axial axis (z-axis).

    Notes
    -----
        Expects input shape (X, Y, Z).

    Parameters
    ----------
    prob : float
        The probability the flip happens. Default: 0.5

    Attributes
    ----------
    prob : float
        The probability the flip happens.

    """

    def __init__(self, prob=0.5):
        """Initialization routine."""
        super().__init__(axis=2, prob=prob)

    def __call__(self, image):
        """Calling routine

       Parameters
       ----------
       image : numpy.ndarray
           The image to flip.

       Returns
       -------
       numpy.ndarray
           The flipped image.

       """
        assert len(image.shape) == 3
        return super().__call__(image)


class Rotate:
    """Rotate the input along a given axis.

    Parameters
    ----------
    axis : int
        axis to rotate. Default is 0.
    deg : tuple
        min and max rotation angles in degrees. Randomly rotates
        within that range. Can be scalar, list or tuple. In case of
        scalar it rotates between -abs(deg) and abs(deg). Default: (-3, 3).

    Attributes
    ----------
    axis : int
        axis to rotate. Default: 0.
    deg : tuple
        min and max rotation angles in degrees. Randomly rotates
        within that range. Can be scalar, list or tuple. In case of
        scalar it rotates between -abs(deg) and abs(deg). Default: (-3, 3).

    """

    def __init__(self, axis=0, deg=(-3, 3)):
        """Initialization routine.

        Raises
        ------
        AssertionError
            if `deg` has not length of three.

        """
        if axis == 0:
            self.axes = (1, 0)
        elif axis == 1:
            self.axes = (2, 1)
        elif axis == 2:
            self.axes = (0, 2)

        if isinstance(deg, tuple) or isinstance(deg, list):
            assert len(deg) == 2
            self.min_rot = np.min(deg)
            self.max_rot = np.max(deg)
        else:
            self.min_rot = -int(abs(deg))
            self.max_rot = int(abs(deg))

    def __call__(self, image):
        """Calling procedure.

        Parameters
        ----------
        image : numpy.ndarray
            The image to rotate.

        Returns
        -------
        numpy.ndarray
            Rotated image.

        """
        rand = np.random.randint(self.min_rot, self.max_rot + 1)
        augmented = rotate(
            image, angle=rand, axes=self.axes, reshape=False
        ).copy()
        return augmented


class SagittalRotate(Rotate):
    """Rotate image's sagittal axis (x-axis).

    Notes
    -----
        Expects input shape (X, Y, Z).

    Attributes
    ----------
    deg : tuple
        min and max rotation angles in degrees. Randomly rotates
        within that range. Can be scalar, list or tuple. In case of
        scalar it rotates between -abs(deg) and abs(deg). Default: (-3, 3).

    """

    def __init__(self, deg=(-3, 3)):
        """Initialization routine."""
        super().__init__(axis=0, deg=deg)


class CoronalRotate(Rotate):
    """Rotate image's coronal axis (y-axis).

    Notes
    -----
        Expects input shape (X, Y, Z).

    Attributes
    ----------
    deg : tuple
        min and max rotation angles in degrees. Randomly rotates
        within that range. Can be scalar, list or tuple. In case of
        scalar it rotates between -abs(deg) and abs(deg). Default is (-3, 3).

    """

    def __init__(self, deg=(-3, 3)):
        """Initialization routine."""
        super().__init__(axis=1, deg=deg)


class AxialRotate(Rotate):
    """Rotate image's axial axis (z-axis).

    Notes
    -----
        Expects input shape (X, Y, Z).

    Attributes
    ----------
    deg : tuple
        min and max rotation angles in degrees. Randomly rotates
        within that range. Can be scalar, list or tuple. In case of
        scalar it rotates between -abs(deg) and abs(deg). Default: (-3, 3).

    """

    def __init__(self, deg=(-3, 3)):
        """Initialization routine."""
        super().__init__(axis=2, deg=deg)


class Translate:
    """Translate the input along a given axis.

    Parameters
    ----------
    axis
        axis to rotate. Default is 0
    dist
        min and max translation distance in pixels. Randomly
        translates within that range. Can be scalar, list or tuple.
        In case of scalar it translates between -abs(dist) and
        abs(dist). Default: (-3, 3).
    """

    def __init__(self, axis=0, dist=(-3, 3)):
        """Initialization routine.

        Raises
        ------
        AssertionError
            if `deg` has not length of three.

        """
        self.axis = axis

        if isinstance(dist, tuple) or isinstance(dist, list):
            assert len(dist) == 2
            self.min_trans = np.min(dist)
            self.max_trans = np.max(dist)
        else:
            self.min_trans = -int(abs(dist))
            self.max_trans = int(abs(dist))

    def __call__(self, image):
        """Calling routine

        Parameters
        ----------
        image : numpy.ndarray
            The image to translate

        Returns
        -------
        numpy.ndarray
            The translated image

        """
        rand = np.random.randint(self.min_trans, self.max_trans + 1)
        augmented = np.zeros_like(image)
        if self.axis == 0:
            if rand < 0:
                augmented[-rand:, :] = image[:rand, :]
            elif rand > 0:
                augmented[:-rand, :] = image[rand:, :]
            else:
                augmented = image
        elif self.axis == 1:
            if rand < 0:
                augmented[:, -rand:, :] = image[:, :rand, :]
            elif rand > 0:
                augmented[:, :-rand, :] = image[:, rand:, :]
            else:
                augmented = image
        elif self.axis == 2:
            if rand < 0:
                augmented[:, :, -rand:] = image[:, :, :rand]
            elif rand > 0:
                augmented[:, :, :-rand] = image[:, :, rand:]
            else:
                augmented = image
        return augmented


class SagittalTranslate(Translate):
    """Translate image along the sagittal axis (x-axis).

    Parameters
    ----------
    dist : tuple
       The distance in each direction. x-axis fixed. Default: (-3,3)

    Notes
    -----
        Expects input shape (X, Y, Z).

    """

    def __init__(self, dist=(-3, 3)):
        """Initialization routine."""
        super().__init__(axis=0, dist=dist)


class CoronalTranslate(Translate):
    """Translate image along the coronal axis (y-axis).

    Parameters
    ----------
    dist : tuple
        The distance in each direction. y-axis fixed.  Default: (-3,3)

    Notes
    -----
        Expects input shape (X, Y, Z).

    """

    def __init__(self, dist=(-3, 3)):
        """Initialization routine."""
        super().__init__(axis=1, dist=dist)


class AxialTranslate(Translate):
    """Translate image along the axial axis (z-axis).

    Parameters
    ----------
    dist : tuple
        The distance in each direction. z-axis fixed. Default: (-3,3)


    Notes
    -----
        Expects input shape (X, Y, Z).

    """

    def __init__(self, dist=(-3, 3)):
        """Initialization routine."""
        super().__init__(axis=2, dist=dist)
