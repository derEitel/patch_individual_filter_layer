# Patch individual filter layer module
Implementation of a PIF layer in PyTorch accompanying the submission of the manuscript.

**Authors**:
Fabian Eitel, Jan Philipp Albrecht, Martin Weygandt, Friedemann Paul, Kerstin Ritter

**Abstract**:
Neuroimaging data, e.g. obtained from magnetic resonance imaging (MRI), is comparably homogeneous due to (1) the uniform structure of the brain and (2) additional efforts to spatially normalize the data to a standard template using linear and non-linear transformations. 
Convolutional neural networks (CNNs), in contrast, have been specifically designed for highly heterogeneous data, such as natural images, by sliding convolutional filters over different positions in an image. Here, we suggest a new CNN architecture that combines the idea of hierarchical abstraction in neural networks with a prior on the spatial homogeneity of neuroimaging data: Whereas early layers are trained globally using standard convolutional layers, we introduce for higher, more abstract layers patch individual filters (PIF). By learning filters in individual image regions (patches) without sharing weights, PIF layers can learn abstract features faster and with fewer samples. We thoroughly evaluated PIF layers for three different tasks and data sets, namely sex classification on UK Biobank data, Alzheimer's disease detection on ADNI data and multiple sclerosis detection on  private hospital data. We demonstrate that CNNs using PIF layers result in higher accuracies, especially in low sample size settings, and need fewer training epochs for convergence. To the best of our knowledge, this is the first study which introduces a prior on brain MRI for CNN learning.


The code for the experiments carried out in the study can be found in the *experiments* directory and give further examples on how to employ the PIF architecture.

**Usage**:

The *pif.py* file contains the Patch Individual Filter layer class which can be incorporated as any other layer type.

```python

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv_1 = nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=0)
        self.pool_1 = nn.MaxPool3d(kernel_size=3, stride=1, padding=0)
        self.conv_2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=0)
        self.pool_2 = nn.MaxPool3d(kernel_size=3, stride=1, padding=0)
        
        self.pif = PatchIndividualFilters3D([15,20,15],
                                            filter_shape=(3,3,3),
                                            patch_shape=(5,5,5),
                                            num_local_filter_in=16,
                                            num_local_filter_out=6,
                                            overlap=1,
                                            reassemble=False,
                                            debug=False)
                                            
    def forward(self, x):
    ....
```

The default version of the PIF layer contains only a single Convolution. One can increase the depth of the PIF layer by adding further operations in:
https://github.com/derEitel/patch_individual_filter_layer/blob/31f5f3736aa99e1dcea1310b8c34f1fced225ea3/pif.py#L79

