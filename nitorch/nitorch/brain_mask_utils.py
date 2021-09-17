import torch
import nibabel
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from nilearn.image import resample_img


all_areas = {
    "Accumbens": (23, 30),
    "Amygdala": (31, 32),
    "Brain Stem": (35, 35),
    "Caudate": (36, 37),
    "Cerebellum": (38, 41),
    "Hippocampus": (47, 48),
    "Parahippocampal gyrus": (170, 171),
    "Pallidum": (55, 56),
    "Putamen": (57, 58),
    "Thalamus": (59, 60),
    "CWM": (44, 45),
    "ACG": (100, 101),
    "Ant. Insula": (102, 103),
    "Post. Insula": (172, 173),
    "AOG": (104, 105),
    "AG": (106, 107),
    "Cuneus": (114, 115),
    "Central operculum": (112, 113),
    "Frontal operculum": (118, 119),
    "Frontal pole": (120, 121),
    "Fusiform gyrus": (122, 123),
    "Temporal pole": (202, 203),
    "TrIFG": (204, 205),
    "TTG": (206, 207),
    "Entorh. cortex": (116, 117),
    "Parietal operculum": (174, 175),
    "SPL": (198, 199),
    "CSF": (46, 46),
    "3rd Ventricle": (4, 4),
    "4th Ventricle": (11, 11),
    "Lateral Ventricles": (49, 52),
    "Diencephalon": (61, 62),
    "Vessels": (63, 64),
    "Optic Chiasm": (69, 69),
    "Vermal Lobules": (71, 73),
    "Basal Forebrain": (75, 76),
    "Calc": (108, 109),
    "GRe": (124, 125),
    "IOG": (128, 129),
    "ITG": (132, 133),
    "LiG": (134, 135),
    "LOrG": (136, 137),
    "MCgG": (138, 139),
    "MFC": (140, 141),
    "MFG": (142, 143),
    "MOG": (144, 145),
    "MOrG": (146, 147),
    "MPoG": (148, 149),
    "MPrG": (150, 151),
    "MSFG": (152, 153),
    "MTG": (154, 155),
    "OCP": (156, 157),
    "OFuG": (160, 161),
    "OpIFG": (162, 163),
    "OrIFG": (164, 165),
    "PCgG": (166, 167),
    "PCu": (168, 169),
    "PoG": (176, 177),
    "POrG": (178, 179),
    "PP": (180, 181),
    "PrG": (182, 183),
    "PT": (184, 185),
    "SCA": (186, 187),
    "SFG": (190, 191),
    "SMC": (192, 193),
    "SMG": (194, 195),
    "SOG": (196, 197),
    "STG": (200, 201),
}


def rescale_mask(target_image_path, neuromorph_map_path):
    """Rescales a given neuromorphological mask to a new affine.

    Parameters
    ----------
    target_image_path : str
        path to the target image. Affine and shape used as template to rescale mask.
    neuromorph_map_path : str

    Returns
    -------
    nmm_mask : nibabel.Nifti1Image

    """
    neuromorph_map = nibabel.load(neuromorph_map_path)
    target_image = nibabel.load(target_image_path)
    
    nmm_mask = resample_img(neuromorph_map, target_affine=target_image.affine,
                            target_shape=target_image.get_data().shape, interpolation="nearest")
    return nmm_mask


def get_mask(mask_path):
    """Loads the data from a given mask.

    Parameters
    ----------
    mask_path : str
        Path to the mask.

    Returns
    -------
    numpy.ndarray
        Data of the given mask.

    """
    return nibabel.load(mask_path).get_data()


def extract_region_mask(mask, region_name):
    """Extracts a given region from the mask.

    Parameters
    ----------
    mask : numpy.ndarray
        The mask.
    region_name : str
        A region name. For available regions please read the documentation.

    Returns
    -------
    region_mask : numpy.ndarray
        Region specific mask. Same shape as input mask.

    """
    region_mask = (mask == all_areas[region_name][0]).astype(int) | (mask == all_areas[region_name][1]).astype(int)
    return region_mask


def extract_multiple_regions_mask(mask, regions):
    """Extracts multiple regions from a mask.

    Parameters
    ----------
    mask : numpy.ndarray
        The mask.
    regions : list
        Regions to be extracted.

    Returns
    -------
    region_mask : numpy.ndarray
        Region specific mask. Same shape as input mask.

    """
    region_mask = np.zeros(mask.shape)    
    for region_name in regions:
        region_mask = region_mask.astype(int) | (mask==all_areas[region_name][0]).astype(int) | (mask==all_areas[region_name][1]).astype(int)
    return region_mask


def extract_region(img, region_mask, gpu):
    """Extracts region(s) from a given images using a mask.

    Parameters
    ----------
    img : numpy.ndarray/torch.tensor
        5D data of a batch of nifti image(s). Dimension expected to be  B, C, H, W, D
    region_mask : numpy.ndarray
        The region mask.
    gpu : int/str
        GPU number.

    Returns
    -------
    patch : numpy.ndarray

    See Also
    --------
    get_mask : Loads the data from a given mask.

    """
    region_mask = torch.from_numpy(region_mask).to("cuda:" + str(gpu))
        
    B, C, H, W, D = img.shape
        
    patch = []
    for i in range(B):
        im = img[i].unsqueeze(dim=0)
         #T = im.shape[-1]

        im = im*region_mask.float()
        # and finally extract
        patch.append(im)
    patch = torch.cat(patch)
    return patch

