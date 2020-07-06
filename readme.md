# Patch individual filter layer module
Implementation of a PIF layer in PyTorch accompanying the submission of the manuscript.

**Authors**:
Fabian Eitel, Jan Philipp Albrecht, Martin Weygandt, Friedemann Paul, Kerstin Ritter

**Abstract**:
Neuroimaging data, e.g. obtained from magnetic resonance imaging (MRI), is comparably homogeneous due to (1) the uniform structure of the brain and (2) additional efforts to spatially normalize the data to a standard template using linear and non-linear transformations. 
Convolutional neural networks (CNNs), in contrast, have been specifically designed for highly heterogeneous data, such as natural images, by sliding convolutional filters over different positions in an image. Here, we suggest a new CNN architecture that combines the idea of hierarchical abstraction in neural networks with a prior on the spatial homogeneity of neuroimaging data: Whereas early layers are trained globally using standard convolutional layers, we introduce for higher, more abstract layers patch individual filters (PIF). By learning filters in individual image regions (patches) without sharing weights, PIF layers can learn abstract features faster and with fewer samples. We thoroughly evaluated PIF layers for three different tasks and data sets, namely sex classification on UK Biobank data, Alzheimer's disease detection on ADNI data and multiple sclerosis detection on  private hospital data. We demonstrate that CNNs using PIF layers result in higher accuracies, especially in low sample size settings, and need fewer training epochs for convergence. To the best of our knowledge, this is the first study which introduces a prior on brain MRI for CNN learning.


Full scripts for experiments and link to the publication will be added soon.
