import torch
from torch.nn.modules.module import Module


class PatchIndividualFilters3D(Module):
    """Layer that splits incoming tensor in sub-parts (patches) applying convolutions patch-wise. Reassembles output.

    A 3D input tensor will be split in user defined patches each of equal size. A convolution operation will
    be applied for each patch separately. Hence, a kernel learnt during training does not slide over the whole
    input tensor, but is restricted to its patch only.

    Parameters
    ----------
    input_dim : list
        Input dimension of the incoming tensor. Initialization of other layer parameter depend on this information.
    filter_shape : list
        Shape of the filter (or kernel) which will be used for EACH convolution operation.
    patch_shape
        Shape of the patch. Incoming 3D tensor will be split in as many patches of that `patch_shape` as
        fit in `input_dim`. Padding will be done for the last patch if patch_shape does not directly
        fit 'x'-times in `input_dim`.
    num_local_filter_in
        Number of channels of the input tensor.
    num_local_filter_out
        Number of channels of the output tensor (e.g. how many filter will be learnt).
    conv_padding
        Applies padding for each convolution operation.
    conv_stride
        Applies a certain stride for each convolution operation.
    reassemble
        Flag indicating the reassembly of the patches. During forward process, incoming tensor will be split in patches.
        If `reassemble` is false output will be a 6D tensor.
    overlap
        Flag indicating whether to create an additional set of patches 
        that are shifted by half the patch size per dimension. This way 
        border regions of the original patches become central in the new
        set. Increases the amount of filters more than two-fold.
        Resulting feature maps are added along the channel dimension.
    debug
        Prints additional information, including tensor dimensions after each tensor operation during forwarding.

    """
    def __init__(self,
                 input_dim,
                 filter_shape,
                 patch_shape,
                 num_local_filter_in,
                 num_local_filter_out,
                 conv_padding=0,
                 conv_stride=1,
                 reassemble=True,
                 overlap=0,
                 debug=0):
        super(PatchIndividualFilters3D, self).__init__()
        self.input_dim = input_dim  # expects it to be a 3D tensor
        self.filter_shape = filter_shape
        self.patch_shape = patch_shape
        self.num_local_filter_out = num_local_filter_out
        self.num_local_filter_in = num_local_filter_in
        self.conv_padding = conv_padding
        self.conv_stride = conv_stride
        self.overlap = overlap

        # calc padding and num_patches
        self.padding_dim = [0] * len(self.input_dim) * 2
        # create 2nd padding dims when using overlapped patches
        if self.overlap > 0:
            self.padding_dim *= 2

        # initialize all local convolution operations
        self.num_patches, self.num_patches_per_dim = self.calc_pad_dim_num_patches(overlap)

        # initialize for each patch a convolution operation
        for patch in range(self.num_patches):
            self.add_module("conv_{}".format(patch),
                            torch.nn.Conv3d(self.num_local_filter_in,
                                      self.num_local_filter_out,
                                      self.filter_shape,
                                      padding=self.conv_padding,
                                      stride=self.conv_stride))
        # further options
        self.grads = {}
        self.reassemble_flag = reassemble
        self.debug = debug

    def save_grad(self, name):
        """Adds a gradient or parameter to store during forwarding of a input tensor.

        Parameters
        ----------
        name : str
            Name of the parameter which shell be stored.

        Returns
        -------
        function
            a function which when called stores the object `grad` in
            class-object dictionary `grads` under the entry `name`.
        """

        def hook(grad):
            """Saves a gradient or parameter in the class-object dictionary.

            Parameters
            ----------
            grad : any
                Whatever shell be stored during forward operation.
            """
            self.grads[name] = grad

        return hook

    def calc_pad_dim_num_patches(self):
        """Calculates the number of patches according to `patch_shape` and `input_shape`.
        
        Returns
        -------
        int
            Number of patches in total.
        list
            Number of splits necessary per dimension.
        """

        # Initalization
        num_patches = 1
        num_patches_per_dim = [0] * len(self.input_dim)
        if self.overlap == 1:
            # call this with size reduced by patchsize/2
        elif self.overlap == 2:
            # call this with size increased by patchsize/2

        # check how often patch_shape fits input_dim for each dimension
        for idx, dim in enumerate(self.patch_shape):
            tmp_remain = self.input_dim[idx] % dim
            tmp_division = self.input_dim[idx] // dim
            if tmp_remain != 0:
                # how much we need to add such that patch_shape fits perfectly
                self.padding_dim[idx * 2] = dim - tmp_remain
                tmp_division += 1
            num_patches_per_dim[idx] = tmp_division
            num_patches = num_patches * tmp_division

        print(num_patches)
        print(num_patches_per_dim)



        return num_patches, num_patches_per_dim

    def pad_to_patch_size(self, input5d):
        """Pads 5D input tensor for perfect fit of `patch_shape`.

        Parameters
        ----------
        input5d : torch.Tensor
            5D input tensor.

        Returns
        -------
        torch.Tensor
            Padded tensor.
        """

        # pad input
        if sum(self.padding_dim):
            # add batch and channel dimension, each twice, to padding dim
            # reverse because padding has last dimension first
            pad_dim = self.padding_dim[::-1] + [0, 0, 0, 0]

            # use constant pad (caution: last dimension first!)
            input5d = torch.nn.functional.pad(input5d, pad_dim, mode='constant', value=0)

        if self.debug:
            input5d.register_hook(self.save_grad("post_padding"))

        return input5d

    def split_5d_channel_last(self, input_tensor):
        """Splits a 5D input tensor to patches of equal size.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The 6D tensor. Patch dimension right after batch dimension (2. Position).
        int
            The batch size.
        """

        # get batch size of input. Always first dimension
        bs = input_tensor.shape[0]

        # initialize
        splits = []

        # loop over batch and split each "input" into patches
        for batch in range(bs):
            feature_map_in = input_tensor[batch]
            # get chunks back from feature map ("input")
            t = self.get_equal_chunks_channel_last(feature_map_in)
            # stack along new "chunk" dimension
            splits_filter = torch.stack(t, 0)
            # append to batch-list
            splits.append(splits_filter)
        # stack upon "batch" dimension
        splits = torch.stack(splits, 0)

        if self.debug:
            input_tensor.register_hook(self.save_grad("post_reshape"))

        return splits, bs  # a 6D tensor

    def get_equal_chunks_channel_last(self, dddc_input):
        """Splits 4D tensor in chunks (patches) of equal size using torch.split.

        Parameters
        ----------
        dddc_input : torch.Tensor
            4D input tensor. First dimension is NOT the batch size. Last dimension MUST be channel (filter) dimension.

        Returns
        -------
        list
            list containing all chunks (patches). list of 4D tensors.
        """
        dddc_input_shape = dddc_input.shape
        tns = [dddc_input]
        # for dirst 3 dimension in 4d input we split according to dimension and number of desired patches
        for dim_idx, dim_size in enumerate(dddc_input_shape[0:3]):
            tns_tmp = []
            for tnsr in tns:
                # determine how to split the tensor.
                # Example:
                # dimension is 4: 2 batches desired in that dimension -> split dimension in [2,2]
                # dimension is 4: 4 batches desired in that dimension -> split dimension in [1,1,1,1]
                dim_split_list = [int(dim_size / self.num_patches_per_dim[dim_idx])] * self.num_patches_per_dim[dim_idx]
                tmp = torch.split(tnsr, dim_split_list, dim_idx)
                tns_tmp += list(tmp)
            tns = tns_tmp
        return tns

    def call_conv(self, idx, patch5d):
        """Calls the convolution for a singe patch.

        Parameters
        ----------
        idx : int
            The patch index.
        patch5d : torch.Tensor
            The patch tensor.

        Returns
        -------
        torch.Tensor
            The result of the convolution for the particular patch. 6D tensor due to call of unsqueeze(2) for
            later cat operation.
        """

        if self.debug:
            patch5d.register_hook(self.save_grad("pre_convol_{}".format(idx)))

        # apply the convolution for the patch
        convolution_res = getattr(self, "conv_{}".format(idx))(patch5d)

        if self.debug:
            convolution_res.register_hook(self.save_grad("post_conv_{}".format(idx)))

        # add 6th dimension at 2. place for later cat
        convolution_res = convolution_res.unsqueeze(2)

        if self.debug:
            convolution_res.register_hook(self.save_grad("post_usqz_c_{}".format(idx)))

        # 6D tensor
        return convolution_res

    def forward(self, input_tensor):
        """Forward function of the layer.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The 5D input tensor.

        Returns
        -------
        torch.Tensor
            The tensor after the forward call. Either 5D, or 6D if `reassemble` is set.
        """

        # set hook to save gradient in debug mode
        if self.debug:
            input_tensor.register_hook(self.save_grad("backward_out"))

        # pad
        input_tensor = self.pad_to_patch_size(input_tensor)

        # switch dimensions. Channel dimension now last dimension.
        input_tensor = torch.einsum("bcxyz->bxyzc", input_tensor)

        # split, channel dimension most inner dimension
        input_tensor, bs = self.split_5d_channel_last(input_tensor)

        # convolution patch wise
        patch_out = []

        # loop over the number of patches
        for idx in range(self.num_patches):
            # get patch
            patch = input_tensor[:, idx]  # get each patch (2nd Dimension of 6D tensor)
            # switch channel back to 2nd dimension for convolution opeartion
            patch = torch.einsum("bxyzc->bcxyz", patch)
            # do separate convolutions with each 5D tensor!
            patch_out += [self.call_conv(idx, patch)]
        # concat to get 6D tensor back
        disassembled_out = torch.cat(patch_out, dim=2)

        if self.reassemble_flag:
            # return 5D tensor
            return self.reassemble(disassembled_out, bs)

        # return 6D tensor
        return disassembled_out

    def reassemble(self, input6d, bs):
        """Reassembles the 6D tensor back to 5D.

        Parameters
        ----------
        input6d : torch.Tensor
            The 6D tensor.
        bs : int
            The batch size.

        Returns
        -------
        torch.Tensor
            The reassembled 5D tensor.
        """
        # switch channel again to last dimension
        input6d = torch.einsum("bcpxyz->bpxyzc", input6d)

        # initialize output
        f_out = []

        # concat the feature maps of each patch channel back to one single feature map
        for batch in range(bs):
            # select patches
            patches = input6d[batch]
            # loop over dimensions in reversed order
            for dim_idx, dim_val in reversed(list(enumerate(self.num_patches_per_dim))):
                patches = self.reassemble_along_axis(patches, dim_idx)
            # delete now unnecessary patch dimension
            f_out.append(patches.squeeze(0))
        # cat together along batch dimension
        f_out = torch.stack(f_out, 0)

        # switch back dimensions
        f_out = torch.einsum("bxyzc->bcxyz", f_out)

        if self.debug:
            f_out.register_hook(self.save_grad("backward_in"))

        return f_out

    def reassemble_along_axis(self, patch, dim):
        """Reassembles the 5D tensor back to 4D.

        Parameters
        ----------
        patch
            The 5D patch (e.g. no batch dimension).
        dim
            The dimension to reassemble.

        Returns
        -------
        torch.Tensor
            The reassembled 4D tensor. No batch dimension yet!
        """

        # Information: patch is 5D tensor but dim refers to original input dimension

        # initialize
        return_tensors = []

        # first dimension-value in patch.shape determines the number of "patches"
        num_of_patches_to_put_together = patch.shape[0] // self.num_patches_per_dim[dim]

        # go through all "cubes" which belong together
        # if nothing was split in that dimension (==1) we don't need to concatenate anything
        if self.num_patches_per_dim[dim] != 1:
            start_idx = 0
            for idx in range(num_of_patches_to_put_together):
                # initialize
                tensors_to_cat_in_dim = []

                # always select from 1. dimension in patch
                # create indices, which we want to select to cut together, assuming they are in a "row" in the array
                # these are as many as the number of tensors created during the split in that dimension
                select_idx = [start_idx] * self.num_patches_per_dim[dim]
                # add "offset"
                select_idx = [m1 + m2 for m1, m2 in zip(select_idx, list(range(0, self.num_patches_per_dim[dim])))]

                # create tensor
                cuda_check = patch.is_cuda
                if cuda_check:
                    cuda_device = patch.get_device()
                    # select tensors being cat together
                    t = torch.index_select(patch, 0, torch.tensor(select_idx).cuda(cuda_device))
                else:
                    t = torch.index_select(patch, 0, torch.tensor(select_idx))

                # select each one and append to list
                for tns in t:
                    tensors_to_cat_in_dim.append(tns)

                # cut them along desired axis
                x = torch.cat(tensors_to_cat_in_dim, dim=dim)
                return_tensors.append(x)

                start_idx = start_idx + self.num_patches_per_dim[dim]
            # stack to 5D again
            return_tensors = torch.stack(return_tensors, dim=0)
            return return_tensors
        return patch
