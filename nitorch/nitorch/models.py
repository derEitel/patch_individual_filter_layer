import numpy as np
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# nitorch
from nitorch.data import show_brain


# TODO: CHECK IF THIS IS STILL UP TO DATE. ENHANCE DOCUMENTATION!!!

class _CAE_3D(nn.Module):
    """Parent Convolutional Autoencoder class for 3D images.

    Parameters
    ----------
    conv_channels : list
        list of convolution channels or list of convolution channels

    Attributes
    ----------
    layers : int
        Number of layers
    conv_channels : list
        A list of convolution channels or list of convolution channels
    is_nested_conv : list
        A boolean list listing if entries of conv_channels are nested.
    valid_activations : dict
        Lists the possible activations:
        "ELU" -> nn.ELU
        "HARDSHRINK" -> nn.Hardshrink
        "HARDTANH" -> nn.Hardtanh
        "LEAKYRELU" -> nn.LeakyReLU
        "LOGSIGMOID" -> nn.LogSigmoid
        "PRELU" -> nn.PReLU
        "RELU" -> nn.ReLU
        "RELU6" -> nn.ReLU6
        "RRELU" -> nn.RReLU
        "SELU" -> nn.SELU
        "SIGMOID" -> nn.Sigmoid
        "SOFTPLUS" -> nn.Softplus
        "SOFTSHRINK" -> nn.Softshrink
        "TANH" -> nn.Tanh
        "TANHSHRINK" -> nn.Tanhshrink
        "THRESHOLD" -> nn.Threshold

    Notes
    -----
        All other Convolutional Autoencoder classes must inherit from this class.

    """

    def __init__(self, conv_channels):
        """Initialization routine."""
        super().__init__()
        # check if there are multiple convolution layers within a layer of the network or not
        self.is_nested_conv = [
            isinstance(each_c, (list, tuple)) for each_c in conv_channels
        ]
        if any(self.is_nested_conv) and not all(self.is_nested_conv):
            raise TypeError(
                " 'conv_channels' can't be a mixture of both lists and ints."
            )
        self.is_nested_conv = any(self.is_nested_conv)

        self.layers = len(conv_channels)
        self.conv_channels = self._format_channels(
            conv_channels, self.is_nested_conv
        )
        self.valid_activations = {
            "ELU": nn.ELU,
            "HARDSHRINK": nn.Hardshrink,
            "HARDTANH": nn.Hardtanh,
            "LEAKYRELU": nn.LeakyReLU,
            "LOGSIGMOID": nn.LogSigmoid,
            "PRELU": nn.PReLU,
            "RELU": nn.ReLU,
            "RELU6": nn.ReLU6,
            "RRELU": nn.RReLU,
            "SELU": nn.SELU,
            "SIGMOID": nn.Sigmoid,
            "SOFTPLUS": nn.Softplus,
            "SOFTSHRINK": nn.Softshrink,
            "TANH": nn.Tanh,
            "TANHSHRINK": nn.Tanhshrink,
            "THRESHOLD": nn.Threshold,
        }

    @staticmethod
    def _format_channels(conv_channels, is_nested_conv=False):
        channels = []
        if is_nested_conv:
            for i in range(len(conv_channels)):
                inner_channels = []
                for j in range(len(conv_channels[i])):
                    if (i == 0) and (j == 0):
                        inner_channels.append([1, conv_channels[i][j]])
                    elif j == 0:
                        inner_channels.append(
                            [conv_channels[i - 1][-1], conv_channels[i][j]]
                        )
                    else:
                        inner_channels.append(
                            [conv_channels[i][j - 1], conv_channels[i][j]]
                        )
                channels.append(inner_channels)
        else:
            for i in range(len(conv_channels)):
                if i == 0:
                    channels.append([1, conv_channels[i]])
                else:
                    channels.append([conv_channels[i - 1], conv_channels[i]])

        return channels

    def assign_parameter(self, parameter, param_name, enable_nested=True):
        """Wrapper for parameters of the Autoencoder.

        Checks if the len and type of the parameter is acceptable.
        If the parameter is just an single value,
        makes its length equal to the number of layers defined in conv_channels

        Parameters
        ----------
        parameter : int/str/list/tuple
            Parameter to check.
        param_name
            Name of the parameter.
        enable_nested
            Enables nested checking of parameters. Default: True.

        Returns
        -------
        parameter
            The input parameter

        Raises
        ------
        ValueError
            Length of parameter != length of layers
        TypeError
            'parameter' not of type int/str/list/tuple

        """
        if isinstance(parameter, (int, str)):
            if self.is_nested_conv and enable_nested:
                return_parameter = [
                    len(inner_list) * [parameter]
                    for inner_list in self.conv_channels
                ]
            else:
                return_parameter = self.layers * [parameter]
        # Perform sanity checks if a list is already provided
        elif isinstance(parameter, (list, tuple)):
            if len(parameter) != self.layers:
                raise ValueError(
                    "The parameter '{}' can either be a single int \
                    or must be a list of the same length as 'conv_channels'.".format(
                        param_name
                    )
                )

            if self.is_nested_conv and enable_nested:
                if any(
                        [
                            len(c) != len(p)
                            for c, p in zip(self.conv_channels, parameter)
                        ]
                ):
                    raise ValueError(
                        "The lengths of the inner lists of the parameter {} \
                        have to be same as the 'conv_channels'".format(
                            param_name
                        )
                    )
            # if all length checks pass just return the parameter todo: do we need an extra variable???
            return_parameter = parameter

        else:
            raise TypeError(
                "Parameter {} is neither an int/ valid str nor a list/tuple but is of type {}".format(
                    param_name, parameter
                )
            )

        return return_parameter

    def add_conv_with_activation(
            self,
            inp_channels,
            out_channels,
            kernel_size,
            padding,
            stride,
            activation_fn,
    ):
        """Creates a torch.nn.Sequential node of a convolution.

        Parameters
        ----------
        inp_channels
            Number of input channels.
        out_channels
            Number of output channels.
        kernel_size:
            Kernel size for the convolution.
        padding
            padding for the convolution
        stride
            Stride for the convolution.
        activation_fn
            A valid activation function. See class description for more detail.

        Returns
        -------
        node : torch.nn.Sequential
            An torch.nn.Sequential node.

        """
        node = nn.Sequential(
            nn.Conv3d(
                inp_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
            ),
            self.valid_activations[activation_fn](inplace=True),
        )
        return node

    def add_deconv_with_activation(
            self,
            inp_channels,
            out_channels,
            kernel_size,
            padding,
            stride,
            out_padding,
            activation_fn,
    ):
        """Creates a torch.nn.Sequential node of a deconvolution.

        Parameters
        ----------
        inp_channels
            Number of input channels.
        out_channels
            Number of output channels.
        kernel_size:
            Kernel size for the deconvolution.
        padding
            padding for the deconvolution.
        out_padding
            padding after the deconvolution.
        stride
            Stride for the deconvolution.
        activation_fn
            A valid activation function. See class description for more detail.

        Returns
        -------
        node : torch.nn.Sequential
            An torch.nn.Sequential node.

        """
        node = nn.Sequential(
            nn.ConvTranspose3d(
                inp_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
                output_padding=out_padding,
            ),
            self.valid_activations[activation_fn](inplace=True),
        )
        return node

    @staticmethod
    def add_pool(pool_type, kernel_size, padding, stride):
        """Adds pooling.

        Parameters
        ----------
        pool_type
            Either "max" (max-pooling) or "avg" (average-pooling) allowed.
        kernel_size:
            Kernel size for pooling.
        padding
            padding for the pooling.
        stride
            Stride for the pooling.

        Returns
        -------
        node : torch.nn.AvgPool3d/torch.nn.MaxPool3d
            The desired pooling node.

        Raises
        ------
        TypeError
            Invalid 'pool_type' value.
        """
        if pool_type == "max":
            node = nn.MaxPool3d(
                kernel_size, padding=padding, stride=stride, return_indices=True
            )
        elif pool_type == "avg":
            node = nn.AvgPool3d(kernel_size, padding=padding, stride=stride)
        else:
            raise TypeError(
                "Invalid value provided for 'pool_type'.\
                Allowed values are 'max', 'avg'."
            )

        return node

    @staticmethod
    def add_unpool(pool_type, kernel_size, padding, stride):
        """Adds unpooling.

        Parameters
        ----------
        pool_type
            Either "max" (max-unpooling) or "avg" (average-unpooling) allowed.
        kernel_size:
            Kernel size for the unpooling.
        padding
            padding for the unpooling.
        stride
            Stride for the unpooling.

        Returns
        -------
        node : torch.nn.AvgPool3d/torch.nn.MaxPool3d
            The desired unpooling node.

        Raises
        ------
        TypeError
            Invalid 'pool_type' value.

        """
        if pool_type == "max":
            node = nn.MaxUnpool3d(kernel_size, padding=padding, stride=stride)
        elif pool_type == "avg":
            node = nn.MaxPool3d(kernel_size, padding=padding, stride=stride)  # todo: that seems to be wrong!
        else:
            raise TypeError(
                "Invalid value provided for 'pool_type'.\
                Allowed values are 'max', 'avg'."
            )

        return node

    def nested_reverse(self, mylist):
        """Reverses a nested list.

        Parameters
        ----------
        mylist
            List to apply nested_reverse

        Returns
        -------

        """
        result = []
        for e in mylist:
            if isinstance(e, (list, tuple)):
                result.append(self.nested_reverse(e))
            else:
                result.append(e)
        result.reverse()
        return result


class CAE_3D(_CAE_3D):
    """3D Convolutional Autoencoder model with only convolution layers.

    Notes
    -----
        Strided convolution can be used for undersampling.

    Parameters
    ----------
    conv_channels
        A list that defines the number of channels of each convolution layer.
        The length of the list defines the number of layers in the encoder.
        The decoder is automatically constructed as an exact reversal of the encoder architecture.
    activation_fn (optional)
        The non-linear activation function that will be appied after every layer
        of convolution / deconvolution.
        Supported values :'ELU', 'HARDSHRINK', 'HARDTANH', 'LEAKYRELU', 'LOGSIGMOID', 'PRELU', 'RELU',
        'RELU6', 'RRELU', 'SELU', 'SIGMOID', 'SOFTPLUS', 'SOFTSHRINK', 'TANH', 'TANHSHRINK', 'THRESHOLD'
        By default nn.ReLu() is applied.
        Can either be a a single int (in which case the same activation is applied to all layers) or
        a list of same length and shape as 'conv_channels'.
    conv_kernel (optional)
        The size of the 3D convolutional kernels to be used.
        Can either be a list of same length as 'conv_channels' or a single int.
        In the former case each value in the list represents the kernel size of that particular
        layer and in the latter case all the layers are built with the same kernel size as
        specified.
    conv_padding (optional)
        The amount of zero-paddings to be done along each dimension.
        Format same as conv_kernel.
    conv_stride (optional)
        The stride of the 3D convolutions.
        Format same as conv_kernel.
    deconv_out_padding (optional)
        The additional zero-paddings to be done to the output
        of ConvTranspose / Deconvolutions in the decoder network.
        By default does (stride-1) number of padding.
        Format same as conv_kernel.
    second_fc_decoder (optional)
        By default this is disabled.
        If a non-empty list of ints is provided then a secondary fully-connected decoder
        network is constructed as per the list.
        Each value represents the number of cells in each layer. Just like 'conv_channels'
        the length of the list defines the number of layers.
        If enabled, the forward() method returns a list of 2 outputs, one from the Autoencoder's
        decoder and the other from this fully-connected decoder network.

    Attributes
    ----------
    conv_kernel
        Todo: explanation mission
    conv_padding
        Todo: explanation mission
    conv_stride
        Todo: explanation mission
    deconv_out_padding
        Todo: explanation mission
    activation_fn
        Todo: explanation mission
    valid_activations
        Todo: explanation mission
    debug
        Todo: explanation mission
    return_encoder_out
        Todo: explanation mission
    second_fc_decoder
        Todo: explanation mission
    convs : nn.ModuleList
        Todo: explanation mission
    deconvs : nn.ModuleList
        Todo: explanation mission
    fcs : nn.ModuleList
        Todo: explanation mission

    """

    def __init__(
            self,
            conv_channels,
            activation_fn="RELU",
            conv_kernel=3,
            conv_padding=1,
            conv_stride=1,
            deconv_out_padding=None,
            second_fc_decoder=[],
    ):
        """Itialization routine."""
        super().__init__(conv_channels)

        assert not (self.is_nested_conv), "The conv_channels must be a list of ints " \
                                          "(i.e. number of channels). It cannot be a list of lists."

        self.conv_kernel = self.assign_parameter(conv_kernel, "conv_kernel")
        self.conv_padding = self.assign_parameter(conv_padding, "conv_kernel")
        self.conv_stride = self.assign_parameter(conv_stride, "conv_stride")
        if deconv_out_padding is None:
            deconv_out_padding = [s - 1 for s in self.conv_stride]
        self.deconv_out_padding = self.assign_parameter(
            deconv_out_padding, "deconv_out_padding"
        )

        self.activation_fn = self.assign_parameter(
            activation_fn, "activation_function"
        )

        for activation in self.activation_fn:
            assert (
                    activation.upper() in self.valid_activations.keys()
            ), "activation functions can only be one of the following str :\n {}".format(
                self.valid_activations.keys()
            )

        # set the switches used in forward() as false  by default
        self.debug = False
        self.return_encoder_out = False

        if second_fc_decoder:
            self.second_fc_decoder = self._format_channels(second_fc_decoder)[
                                     1:
                                     ]
        else:
            self.second_fc_decoder = []

        self.convs = nn.ModuleList()
        self.deconvs = nn.ModuleList()

        for i in range(self.layers):
            # build the encoder
            self.convs.append(
                self.add_conv_with_activation(
                    self.conv_channels[i][0],
                    self.conv_channels[i][1],
                    self.conv_kernel[i],
                    self.conv_padding[i],
                    self.conv_stride[i],
                    self.activation_fn[i],
                )
            )
            # build the decoder
            self.deconvs.append(
                self.add_deconv_with_activation(
                    self.conv_channels[-i - 1][1],
                    self.conv_channels[-i - 1][0],
                    self.conv_kernel[-i - 1],
                    self.conv_padding[-i - 1],
                    self.conv_stride[-i - 1],
                    self.deconv_out_padding[-i - 1],
                    self.activation_fn[-i - 1],
                )
            )
        if self.second_fc_decoder:
            # build the second fc decoder
            self.fcs = nn.ModuleList()
            for layer in self.second_fc_decoder:
                self.fcs.append(nn.Linear(layer[0], layer[1]))

    def set_debug(self, bool_val):
        """sets the debug flag

        Parameters
        ----------
        bool_val : bool
            Either False or True.

        """
        self.debug = bool_val

    def set_return_encoder_out(self, bool_val):
        """Sets the return_encoder_out parameter.

        Parameters
        ----------
        bool_val : bool
            Either False or True.

        """
        self.return_encoder_out = bool_val

    def forward(self, x):
        """Forward function of the model.

        Parameters
        ----------
        x
            The data to model.

        Returns
        -------
        x
            The output of the modeling process.

        """
        if self.debug:
            print("\nImage dims =" + str(x.size()))

        # encoder
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if self.debug:
                print("conv{} output dim = {}".format(i + 1, x.size()))

        encoder_out = x

        if self.debug:
            print("\nEncoder output dims =" + str(encoder_out.size()) + "\n")

        # decoder
        for i, deconv in enumerate(self.deconvs):
            x = deconv(x)
            if self.debug:
                print("deconv{} output dim = {}".format(i + 1, x.size()))

        if self.debug:
            print("\nDecoder output dims =" + str(x.size()) + "\n")

        if self.return_encoder_out:

            return [x, encoder_out]
        else:

            return x


class CAE_3D_with_pooling(_CAE_3D):
    """3D Convolutional Autoencoder model with alternating Pooling layers.

    Parameters
    ----------
    conv_channels
        A nested list whose length defines the number of layers. Each layer
        can intern have multiple convolutions followed by a layer of Pooling. The lengths of the
        inner list defines the number of convolutions per such layer and the value defines the number of
        channels for each of these convolutions.
        The decoder is constructed to be simply an exact reversal of the encoder architecture.
    activation_fn (optional)
        The non-linear activation function that will be appied after every layer
        of convolution / deconvolution. By default nn.ReLu() is applied.
        Supported values: 'ELU', 'HARDSHRINK', 'HARDTANH', 'LEAKYRELU', 'LOGSIGMOID', 'PRELU', 'RELU',
        'RELU6', 'RRELU', 'SELU', 'SIGMOID', 'SOFTPLUS', 'SOFTSHRINK', 'TANH', 'TANHSHRINK', 'THRESHOLD'
        Can either be a a single int (in which case the same activation is applied to all layers) or
        a list of same length and shape as 'conv_channels'.
    conv_kernel (optional)
        The size of the 3D convolutional kernels to be used.
        Can either be a list of lists of same lengths as 'conv_channels' or a single int.
        In the former case each value in the list represents the kernel size of that particular
        layer and in the latter case all the layers are built with the same kernel size as
        specified.
    conv_padding (optional)
        The amount of zero-paddings to be done along each dimension.
        Format same as conv_kernel.
    conv_stride (optional)
        The stride of the 3D convolutions.
        Format same as conv_kernel.
    deconv_out_padding (optional)
        The additional zero-paddings to be done to the output
        of ConvTranspose / Deconvolutions in the decoder network.
        By default does (stride-1) number of padding.
        Format same as conv_kernel.
    pool_type (optional)
        The type of pooling to be used. Options are (1)"max"  (2)"avg"
    pool_kernel, pool_padding, pool_stride (optional)
        Can either be a single int or a list
        of respective pooling parameter values.
        The length of these list must be same as length of conv_channels i.e. the number of layers.
    second_fc_decoder (optional)
        By default this is disabled.
        If a non-empty list of ints is provided then a secondary decoder of a fully-connected network
        is constructed as per the list.

    Attributes
    ----------
    is_nested_conv
        Todo: explanation mission
    conv_kernel
        Todo: explanation mission
    conv_padding
        Todo: explanation mission
    conv_stride
        Todo: explanation mission
    pool_kernel
        Todo: explanation mission
    pool_padding
        Todo: explanation mission
    pool_stride
        Todo: explanation mission
    activation_fn
        Todo: explanation mission
    valid_activations
        Todo: explanation mission
    deconv_channels
        Todo: explanation mission
    deconv_kernel
        Todo: explanation mission
    deconv_padding
        Todo: explanation mission
    deconv_stride
        Todo: explanation mission
    debug
        Todo: explanation mission
    return_encoder_out
        Todo: explanation mission
    deconv_out_padding
        Todo: explanation mission
    deconv_out_padding
        Todo: explanation mission
    convs : nn.ModuleList
        Todo: explanation mission
    pools : nn.ModuleList
        Todo: explanation mission
    deconvs : nn.ModuleList
        Todo: explanation mission
    unpools : nn.ModuleList
        Todo: explanation mission
    pools
        Todo: explanation mission
    unpools
        Todo: explanation mission

    """

    def __init__(
            self,
            conv_channels,
            activation_fn=nn.ReLU,
            conv_kernel=3,
            conv_padding=1,
            conv_stride=1,
            pool_type="max",
            pool_kernel=2,
            pool_padding=0,
            pool_stride=2,
            deconv_out_padding=None):
        super().__init__(conv_channels)

        assert (
            self.is_nested_conv
        ), "The conv_channels must be a list of list of ints Ex. [[16],[32 64],[64],...] (i.e. number of channels)." \
           "It cannot be a list."

        self.conv_kernel = self.assign_parameter(conv_kernel, "conv_kernel")
        self.conv_padding = self.assign_parameter(conv_padding, "conv_padding")
        self.conv_stride = self.assign_parameter(conv_stride, "conv_stride")
        self.pool_kernel = self.assign_parameter(
            pool_kernel, "pool_kernel", enable_nested=False
        )
        self.pool_padding = self.assign_parameter(
            pool_padding, "pool_padding", enable_nested=False
        )
        self.pool_stride = self.assign_parameter(
            pool_stride, "pool_stride", enable_nested=False
        )

        self.activation_fn = self.assign_parameter(
            activation_fn, "activation_function"
        )

        for activations in self.activation_fn:
            for activation in activations:
                assert (
                        activation.upper() in self.valid_activations.keys()
                ), "activation functions can only be one of the following str :\n {}".format(
                    self.valid_activations.keys()
                )

        self.deconv_channels = self.nested_reverse(self.conv_channels)
        self.deconv_kernel = self.nested_reverse(self.conv_kernel)
        self.deconv_padding = self.nested_reverse(self.conv_padding)
        self.deconv_stride = self.nested_reverse(self.conv_stride)

        # set the switches used by forward() as false by default
        self.debug = False
        self.return_encoder_out = False

        if deconv_out_padding is not None:
            self.deconv_out_padding = self.nested_reverse(
                self.assign_parameter(deconv_out_padding, "deconv_out_padding")
            )
        else:
            self.deconv_out_padding = [
                [s - 1 for s in layer] for layer in self.deconv_stride
            ]

        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.deconvs = nn.ModuleList()
        self.unpools = nn.ModuleList()

        for i in range(self.layers):
            self.convs.append(
                nn.ModuleList(
                    [
                        self.add_conv_with_activation(
                            inner_conv_channels[0],
                            inner_conv_channels[1],
                            self.conv_kernel[i][j],
                            self.conv_padding[i][j],
                            self.conv_stride[i][j],
                            self.activation_fn[i][j],
                        )
                        for j, inner_conv_channels in enumerate(
                        self.conv_channels[i]
                    )
                    ]
                )
            )

            self.deconvs.append(
                nn.ModuleList(
                    [
                        self.add_deconv_with_activation(
                            inner_deconv_channels[0],
                            inner_deconv_channels[1],
                            self.deconv_kernel[i][j],
                            self.deconv_padding[i][j],
                            self.deconv_stride[i][j],
                            self.deconv_out_padding[i][j],
                            self.activation_fn[i][j],
                        )
                        for j, inner_deconv_channels in enumerate(
                        self.deconv_channels[i]
                    )
                    ]
                )
            )

            self.pools.append(
                self.add_pool(
                    pool_type,
                    self.pool_kernel[i],
                    stride=self.pool_stride[i],
                    padding=self.pool_padding[i],
                )
            )
            self.unpools.append(
                self.add_unpool(
                    pool_type,
                    self.pool_kernel[-i - 1],
                    stride=self.pool_stride[-i - 1],
                    padding=self.pool_padding[-i - 1],
                )
            )

    def set_debug(self, bool_val):
        """Sets debug flag.

        Parameters
        ----------
        bool_val : bool
            Either True or False.

        """
        self.debug = bool_val

    def set_return_encoder_out(self, bool_val):

        """Sets the return_encoder_out parameter.

        Parameters
        ----------
        bool_val : bool
            Either False or True.

        """
        self.return_encoder_out = bool_val

    def forward(self, x):
        """Forward function of the model.

        Notes
        -----
            return_encoder_out : If enabled returns a list with 2 values,
            first one is the Autoencoder's output and the other the intermediary output of the encoder.

        Parameters
        ----------
        x
            The data to model.

        Returns
        -------
        x
            The output of the modeling process.

        """

        pool_idxs = []
        pool_sizes = [x.size()]  # https://github.com/pytorch/pytorch/issues/580

        if self.debug:
            print("\nImage dims =" + str(x.size()))

        # encoder
        for i, (convs, pool) in enumerate(zip(self.convs, self.pools)):
            for j, conv in enumerate(convs):
                x = conv(x)
                if self.debug:
                    print(
                        "conv{}{} output dim = {}".format(
                            i + 1, j + 1, x.size()
                        )
                    )

            x, idx = pool(x)
            pool_sizes.append(x.size())
            pool_idxs.append(idx)
            if self.debug:
                print("pool{} output dim = {}".format(i + 1, x.size()))

        encoder_out = x

        if self.debug:
            print("\nEncoder output dims =" + str(encoder_out.size()) + "\n")

        # decoder
        pool_sizes.pop()  # pop out the last size as it is not necessary

        for i, (deconvs, unpool) in enumerate(zip(self.deconvs, self.unpools)):

            x = unpool(x, pool_idxs.pop(), output_size=pool_sizes.pop())
            if self.debug:
                print("unpool{} output dim = {}".format(i + 1, x.size()))

            for j, deconv in enumerate(deconvs):
                x = deconv(x)
                if self.debug:
                    print(
                        "deconv{}{} output dim = {}".format(
                            i + 1, j + 1, x.size()
                        )
                    )

        if self.debug:
            print("\nDecoder output dims =" + str(x.size()) + "\n")

        if self.return_encoder_out:
            return [x, encoder_out]
        else:
            return x


class MLP(nn.Module):
    """Constructs fully-connected deep neural networks.

    Parameters
    ----------
    layers
        Each value represents the number of neurons in each layer. The length of the list defines the number of layers.
    output_activation
        Default: nn.LogSoftmax

    Attributes
    ----------
    layers
        Todo: explanation mission
    debug
        Todo: explanation mission
    fcs
        Todo: explanation mission

    """

    def __init__(self, layers=[], output_activation=nn.LogSoftmax):
        """Initialization routine."""
        super().__init__()
        self.layers = self._format_channels(layers)
        #         self.output_activation = output_activation
        self.debug = False

        # build the fully-connected layers
        self.fcs = nn.ModuleList()

        for layer in self.layers:
            if (layer) is not self.layers[-1]:
                self.fcs.append(self.add_linear_with_Relu(layer))
            elif output_activation is not None:
                self.fcs.append(
                    nn.Sequential(
                        nn.Linear(layer[0], layer[1]), output_activation()
                    )
                )
            else:
                self.fcs.append(nn.Linear(layer[0], layer[1]))

    def set_debug(self, bool_val):
        """Sets debug flag.

        Parameters
        ----------
        bool_val : bool
            Either True or False.

        """
        self.debug = bool_val

    # todo: static function
    def _format_channels(self, layers):
        layer_inout = []
        for i in range(len(layers) - 1):
            layer_inout.append([layers[i], layers[i + 1]])
        return layer_inout

    # todo: static function!!!
    def add_linear_with_Relu(self, layer):
        """Adds a linear layer with Relu to the model.

        Parameters
        ----------
        layer
            A new layer.

        Returns
        -------
        nn.Sequential
            The new layer.

        """
        node = nn.Sequential(nn.Linear(layer[0], layer[1]), nn.ReLU(True))
        return node

    def forward(self, x):
        """Forward function of the model.

        Parameters
        ----------
        x
            The data to model.

        Returns
        -------
        x
            The output of the modeling process.

                """

        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if self.debug:
                print("FC {} output dims ={}".format(i, x.size()))

        return x
