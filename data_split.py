import os
import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import zoom
from tabulate import tabulate
import nibabel as nib
import h5py


def print_df_stats(df, df_train, df_val, df_test):
    """Print some statistics about the patients and images in a dataset."""
    headers = ['Images', '-> AD', '-> CN', 'Patients', '-> AD', '-> CN']

    def get_stats(df):
        df_ad = df[df['DX'] == 'Dementia']
        df_cn = df[df['DX'] == 'CN']
        return [len(df), len(df_ad), len(df_cn), len(df['PTID'].unique()), len(df_ad['PTID'].unique()), len(df_cn['PTID'].unique())]

    stats = []
    stats.append(['All'] + get_stats(df))
    stats.append(['Train'] + get_stats(df_train))
    stats.append(['Val'] + get_stats(df_val))
    stats.append(['Test'] + get_stats(df_test))

    print(tabulate(stats, headers=headers))
    print()
    
# load images in matrix
def create_dataset(dataset, z_factor, settings, mask=None):
    data_matrix = [] 
    labels = [] 
    for idx, row in dataset.iterrows():
        path = os.path.join(settings["ADNI_DIR"],
                            str(row["PTID"]),
                            row["Visit"].replace(" ", ""),
                            str(row["PTID"]) + "_" + str(row["Scan.Date"]).replace("/", "-") + "_" + row["Visit"].replace(" ", "") + "_" + str(row["Image.ID"]) + "_" + row["DX"] + "_Warped.nii.gz")
        struct_arr = np.NAN
        scan = nib.load(path)
        struct_arr = scan.get_data().astype(np.float32)
        if mask is not None:
            struct_arr *= mask
        if z_factor is not None:
            struct_arr = zoom(struct_arr, z_factor)
        data_matrix.append(struct_arr)
        labels.append((row["DX"] == "Dementia") *1)      
    return np.array(data_matrix), np.array(labels)