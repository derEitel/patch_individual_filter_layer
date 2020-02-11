import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
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
        Defines whether patches should overlap. 0 will create disjoint patches, 1 will create a minimal number of
        maximally overlapping patches. 2 is  not yet implemented.
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
                 overlap=1,
                 debug=0):
        super().__init__()
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

        # initialize all local convolution operations
        self.num_patches, self.num_patches_per_dim, self.padded_input_dim = self.calc_pad_dim_num_patches()

        # assert whether overlap is possible
        if self.overlap:
            assert self.num_patches != 1,\
                "Can not build overlap patches for input_shape: {} and patch_shape {}".format(self.input_dim,
                                                                                              self.patch_shape)

            self.num_overlap_patches, self.num_overlap_patches_per_dim = self.calc_overlap_num_patches()

        # initialize for each patch a convolution operation
        for patch in range(self.num_patches):
            self.add_module("conv_{}".format(patch),
                            nn.Conv3d(self.num_local_filter_in,
                                      self.num_local_filter_out,
                                      self.filter_shape,
                                      padding=self.conv_padding,
                                      stride=self.conv_stride))

        # initialize convolutions for overlap patches
        if self.overlap:
            for patch in range(self.num_overlap_patches):
                # initialize convolution object -
                self.add_module("conv_ov_{}".format(patch),
                                nn.Conv3d(self.num_local_filter_in,
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
        list
            Length per dimension after padding
        """

        num_patches = 1
        num_patches_per_dim = [0] * len(self.input_dim)
        padded_input_dim = [0] * len(self.input_dim)

        # check if patch_shape fits input dimension
        for idx, dim in enumerate(self.patch_shape):
            tmp_remain = self.input_dim[idx] % dim
            tmp_division = self.input_dim[idx] // dim
            if tmp_remain != 0:
                # how much we need to add such that patch_shape fits perfectly
                pad_in_dim = dim - tmp_remain
                self.padding_dim[idx * 2] = pad_in_dim
                padded_input_dim[idx] = self.input_dim[idx] + pad_in_dim
                tmp_division += 1
            else:
                padded_input_dim[idx] = self.input_dim[idx]
            # set how many patches fit in dim & update total number if patches in all dimensions
            num_patches_per_dim[idx] = tmp_division
            num_patches = num_patches * tmp_division
        return num_patches, num_patches_per_dim, padded_input_dim

    def calc_overlap_num_patches(self):
        """Calculates the number of overlapping patches according.

        Returns
        -------
        int
            Number of patches in total.
        list
            Number of splits necessary per dimension.
        """
        # uses one patch less per dimension 
        num_ov_patches_per_dim = [np_pd - 1 if np_pd > 1 else 1 for np_pd in self.num_patches_per_dim]
        num_ov_patches = np.prod(num_ov_patches_per_dim)

        return num_ov_patches, num_ov_patches_per_dim

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

    def split_5d_channel_last(self, input_tensor, tmp_num_patches_per_dim):
        """Splits a 5D input tensor to patches of equal size.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor.
        tmp_num_patches_per_dim : list
            Number of patches per dimension.
            
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
            t = self.get_equal_chunks_channel_last(feature_map_in, tmp_num_patches_per_dim)
            # stack along new "chunk" dimension
            splits_filter = torch.stack(t, 0)
            # append to batch-list
            splits.append(splits_filter)
        # stack upon "batch" dimension
        splits = torch.stack(splits, 0)

        if self.debug:
            input_tensor.register_hook(self.save_grad("post_reshape"))

        return splits, bs  # a 6D tensor

    def get_equal_chunks_channel_last(self, dddc_input ,tmp_num_patches_per_dim):
        """Splits 4D tensor in chunks (patches) of equal size using torch.split.

        Parameters
        ----------
        dddc_input : torch.Tensor
            4D input tensor. First dimension is NOT the batch size. Last dimension MUST be channel (filter) dimension.
        tmp_num_patches_per_dim : list
            Number of patches per dimension.
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
                dim_split_list = [int(dim_size / tmp_num_patches_per_dim[dim_idx])] * tmp_num_patches_per_dim[dim_idx]
                tmp = torch.split(tnsr, dim_split_list, dim_idx)
                tns_tmp += list(tmp)
            tns = tns_tmp
        return tns

    def call_conv(self, idx, patch5d, mode=""):
        """Calls the convolution for a single patch.

        Parameters
        ----------
        idx : int
            The patch index.
        patch5d : torch.Tensor
            The patch tensor.
        mode : String 
            Set to "ov" for overlapping patches.

        Returns
        -------
        torch.Tensor
            The result of the convolution for the particular patch. 6D tensor due to call of unsqueeze(2) for
            later cat operation.
        """

        if self.debug:
            patch5d.register_hook(self.save_grad("pre_convol_{}".format(idx)))

        # two mode: overlap or normal
        if mode == "ov":
            convolution_res = getattr(self, "conv_ov_{}".format(idx))(patch5d)
            # debug
            if self.debug:
                convolution_res.register_hook(self.save_grad("post_conv_ov_{}".format(idx)))
        else:
            convolution_res = getattr(self, "conv_{}".format(idx))(patch5d)
            # debug
            if self.debug:
                convolution_res.register_hook(self.save_grad("post_conv_{}".format(idx)))

        # add 6th dimension at 2. place for later cat
        convolution_res = convolution_res.unsqueeze(2)

        if self.debug:
            convolution_res.register_hook(self.save_grad("post_usqz_c_{}".format(idx)))

        # 6D tensor
        return convolution_res

    def prepare_overlap_input(self, input_tensor):
        """Creates a tensor with dimensions adjusted for the overlapping patches.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The 5D input tensor.

        Returns
        -------
        torch.Tensor
            Adjusted input tensor matching the overlapping patches' dimensions.
        """
        for dim_idx, dim_val in enumerate(self.padded_input_dim):
            # check if overlapping patches possible in that dimension
            if dim_val > self.patch_shape[dim_idx]:
                # delete a patch from input for each dimension -> maximize overlap
                tmp_remain = self.patch_shape[dim_idx] % 2
                tmp_division = self.patch_shape[dim_idx] # use full patch size as input is padded already

                if tmp_remain:  # patch of odd size, no perfect overlap possible
                    offset = [tmp_division, input_tensor.shape[dim_idx + 1] - tmp_division - 1]  # add batch dim
                else:
                    offset = [tmp_division, input_tensor.shape[dim_idx + 1] - tmp_division]  # add batch dim

                # narrow input in "dim_idx + 1" dimension (added batch dimension)
                input_tensor = torch.narrow(input_tensor, dim_idx + 1, offset[0], offset[1] + tmp_remain) # add remain to ensure it matches the patch size again
        return input_tensor

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
        patch : torch.Tensor
            The 5D patch (e.g. no batch dimension).
        dim : int
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

    def patch_wise_conv(self, input_tensor, tmp_num_patches, mode=""):
        """Apply the patch individual convolutions by looping over the patches.

        Parameters
        ----------
        input_tensor : torch.Tensor
            6D input tensor split into patches.
        tmp_num_patches : int
            number of patches.
        mode :  String
            Set to "ov" for overlapping patches.

        Returns
        -------
        torch.Tensor
            The concatenated 6D tensor of all feature maps.
        """

        # initialize
        patch_out = []

        for idx in range(tmp_num_patches):
            # get patch
            patch = input_tensor[:, idx]  # get each patch (2nd Dimension of 6D tensor)
            # switch channel back to 2nd dimension
            patch = torch.einsum("bxyzc->bcxyz", patch)
            # do separate convolutions with each 5D tensor!
            patch_out += [self.call_conv(idx, patch, mode)]
        # concat to get 6D tensor back
        return torch.cat(patch_out, dim=2)

    def merge_overlapped_output(self, og, ov):
        """Merge the original feature maps with those of the overlapped patches.
        Pads with zeros where the overlapped feature map is too small, resulting
        from the reduced input to the overlapped convolutions.

        Parameters
        ----------
        og : torch.Tensor
            Original feature maps.
        ov : torch.Tensor
            Feature maps from the overlapped patches.

        Returns
        -------
        torch.Tensor
            The concatenated 6D tensor of all feature maps.
        """
        # merge the overlapped output back in with the original
        if ov.shape[-3:] != og.shape[-3:]:
            # a smaller overlapped output needs to be padded
            pad_len = np.flip(np.array(og.shape) - np.array(ov.shape))[:3] # pad only the spatial dimensions
            pad_len_idv = []
            # pad left and right the same amount unless its uneven
            for idx, pad in enumerate(pad_len):
                remain = pad % 2
                pad = pad // 2
                pad_len_idv.append(pad)
                pad_len_idv.append(pad + remain)
            ov = torch.nn.functional.pad(ov, tuple(pad_len_idv), mode='constant', value=0)
        return torch.cat((og, ov), dim=1)

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

        if self.overlap:
            # clone for building  overlap patches. NEW TENSOR CLONE! Backward flow is conserved!
            input_ov = input_tensor.clone()
            # reshape for overlap, channel dimension most inner dimension, result is 6D
            # NEW TENSOR CLONE! Backward flow is conserved!
            input_ov = self.prepare_overlap_input(input_ov)
            input_ov, _ = self.split_5d_channel_last(input_ov, self.num_overlap_patches_per_dim)

        # split, channel dimension most inner dimension
        input_tensor, bs = self.split_5d_channel_last(input_tensor, self.num_patches_per_dim)

        # convolution patch wise
        patch_out = []

        # convolution patch wise
        disassembled_out = self.patch_wise_conv(input_tensor, self.num_patches)
        if self.overlap:
            disassembled_ov_out = self.patch_wise_conv(input_ov, self.num_overlap_patches, mode="ov")

        if self.reassemble_flag:
            # reassemeble and merge to 5D tensor
            r = self.reassemble(disassembled_out, bs, self.num_patches_per_dim)
            if self.overlap:
                r_ov = self.reassemble(disassembled_ov_out, bs, self.num_overlap_patches_per_dim)
                # pad ov and merge with non-overlapped
                output = self.merge_overlapped_output(r, r_ov)
            else:
                output = r
        else:
            # merge to 6D tensor
            if self.overlap:
                # individual feature maps have the same size for both og and ov 
                output = torch.cat((disassembled_out, disassembled_ov_out), dim=2)
            else:
                output = disassembled_out

        return output