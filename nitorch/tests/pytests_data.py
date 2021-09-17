import os
import sys
import torch
import shutil

# set nitorch path
NITORCH_DIR = os.getcwd()
# load functions from nitorch
sys.path.insert(0, os.path.join(NITORCH_DIR))
from nitorch.data import *
from nitorch.callbacks import EarlyStopping, ModelCheckpoint
from nitorch.trainer import Trainer
from nitorch.initialization import weights_init
from nitorch.metrics import regression_accuracy
from ritter_utils.data import *

if __name__ == "__main__":
    print("Starting tests ...")
    if torch.cuda.is_available():
        device = torch.device('cpu')
        print("Running on GPU ...")
    else:
        device = torch.device('cpu')
        print("Running on CPU ...")

    source_dir = os.path.join(NITORCH_DIR, "data/OASIS_BIDS_example_data/")

    #############################################
    # Test 1-a : loading example data for classification - no hold out set
    #############################################
    print(
        "-----------------------------------------------\nTesting nitorch.data.DataBunch() :"
        " Binary classification - no hold out set")

    oasis_test_data_c = DataBunch(
        source_dir=source_dir,
        image_dir="",
        table=f"OASIS_example.csv",
        separator="\t",
        path=os.path.join(NITORCH_DIR, source_dir, "results"),
        mask=os.path.join(NITORCH_DIR, source_dir, "masks/mask_121_145_121.nii"),
        file_col="path",
        label_col="labels_simplified",
        labels_to_keep=["clinical dementia", "healthy"],
        ptid_col="subjectID",
        random_state=69,
        balance=True,
        val_size=0.1,
        num_samples=16,
        z_factor=0.5,
        transforms=None,
        grouped=True
    )
    oasis_test_data_c.build_dataloaders(bs=2)
    trn_subj = oasis_test_data_c.df_trn.subjectID.tolist()
    tst_subj = oasis_test_data_c.df_val.subjectID.tolist()
    assert len(set(trn_subj) & set(tst_subj)) == 0, "Test 1-a 'data loading - classification - no hold out set':" \
                                                    " At least one subject appears in test and " \
                                                    "train split - TEST FAILED"
    print("Test 1-a 'data loading - classification -no hold out set': "
          "Patient wise splitting successful - TEST PASSED")

    #############################################
    # Test 1-b : loading example data for classification - hold out set - normalization
    #############################################
    print(
        "-----------------------------------------------\nTesting nitorch.data.DataBunch() : "
        "Binary classification - hold out set - normalization")

    oasis_test_data_c_h = DataBunch(
        source_dir=source_dir,
        image_dir="",
        table=f"OASIS_example.csv",
        separator="\t",
        path=os.path.join(NITORCH_DIR, source_dir, "results"),
        mask=None,
        file_col="path",
        label_col="labels_simplified",
        labels_to_keep=["clinical dementia", "healthy"],
        ptid_col="subjectID",
        random_state=69,
        balance=True,
        val_size=0.1,
        hold_out_size=0.1,
        num_samples=14,
        z_factor=0.5,
        transforms=None,
        grouped=True,
        force_to_shape=[80, 80, 80]
    )
    oasis_test_data_c_h.build_dataloaders(bs=2, normalize=True)
    trn_subj = oasis_test_data_c_h.df_trn.subjectID.tolist()
    tst_subj = oasis_test_data_c_h.df_val.subjectID.tolist()
    ho_subj = oasis_test_data_c_h.df_ho.subjectID.tolist()
    assert len(set(trn_subj) & set(tst_subj) & set(ho_subj)) == 0, "Test 1-b 'data loading - classification - " \
                                                                   "hold out set - normalization': " \
                                                                   "At least one subject appears " \
                                                                   "in test and/or train and/or hold out split" \
                                                                   " - TEST FAILED"
    print("Test 1-b 'data loading - classification - hold out set - normalization': "
          "Patient wise splitting successful - TEST PASSED")
    oasis_test_data_c_h.reshuffle_datasets()
    trn_subj_r = oasis_test_data_c_h.df_trn.subjectID.tolist()
    tst_subj_r = oasis_test_data_c_h.df_val.subjectID.tolist()
    assert len(set(trn_subj_r) & set(tst_subj_r)) == len(
        set(trn_subj) & set(tst_subj)), "Test 1-b 'data loading - classification - " \
                                        "hold out set - normalization': " \
                                        "Reshuffling incorrect - TEST FAILED"
    print("Test 1-b 'data loading - classification - hold out set - normalization': "
          "Reshuffling successful - TEST PASSED")
    ho_subj_r = oasis_test_data_c_h.df_ho.subjectID.tolist()
    assert len(set(ho_subj) & set(ho_subj_r)) == len(set(ho_subj)), "Test 1-b 'data loading - classification - " \
                                                                    "hold out set - normalization': " \
                                                                    "Reshuffling mixed hold out set" \
                                                                    " - TEST FAILED"
    print("Test 1-b 'data loading - classification - hold out set - normalization': "
          "Hold out set not touched in reshuffling process- TEST PASSED")

    #############################################
    # Test 2 : loading example data for regression
    #############################################

    print("-----------------------------------------------\nTesting nitorch.data.DataBunch() : Regression")

    oasis_test_data_r = DataBunch(
        source_dir=source_dir,
        image_dir="",
        table=f"OASIS_example.csv",
        separator="\t",
        path=os.path.join(NITORCH_DIR, source_dir, "results"),
        mask=None,
        file_col="path",
        label_col="Age",
        prediction_type="r",
        ptid_col="subjectID",
        random_state=69,
        balance=True,
        val_size=0.1,
        num_samples=16,
        z_factor=0.5,
        transforms=None,
        grouped=True
    )
    oasis_test_data_r.build_dataloaders(bs=2)
    trn_subj = oasis_test_data_r.df_trn.subjectID.tolist()
    tst_subj = oasis_test_data_r.df_val.subjectID.tolist()
    assert len(set(trn_subj) & set(tst_subj)) == 0, "Test 2 'data loading - regression': At least one subject " \
                                                    "appears in test and train split - TEST FAILED"
    print("Test 2 'data loading - regression': "
          "Patient wise splitting successful - TEST PASSED")

    #############################################
    # Test 3 : saving databunch - storing and reloading data
    #############################################

    # delete old results
    if os.path.exists(os.path.join(NITORCH_DIR, source_dir, "results")):
        shutil.rmtree(os.path.join(NITORCH_DIR, source_dir, "results"))

    oasis_test_data_c = DataBunch(
        source_dir=source_dir,
        image_dir="",
        table=f"OASIS_example.csv",
        separator="\t",
        path=os.path.join(NITORCH_DIR, source_dir, "results"),
        mask=None,
        file_col="path",
        label_col="labels_simplified",
        labels_to_keep=["clinical dementia", "healthy"],
        ptid_col="subjectID",
        random_state=69,
        balance=True,
        val_size=0.1,
        hold_out_size=0.1,
        cache_filename="test_3",
        num_samples=14,
        z_factor=0.5,
        grouped=True,
        force_to_shape=[80, 80, 80]
    )
    # store
    oasis_test_data_c.save_h5py()
    # reload
    oasis_test_data_h = DataBunch.from_disk(
        load_path=os.path.join(NITORCH_DIR, source_dir, "results"),
        save_path=os.path.join(NITORCH_DIR, source_dir, "results"),
        load_filename="test_3",
        load_h5py=True)
    trn_subj_c = oasis_test_data_c.df_trn.subjectID.tolist()
    trn_subj_h = oasis_test_data_h.df_trn.subjectID.tolist()
    assert trn_subj_c == trn_subj_h, "Test 3 'saving databunch - storing and reloading data':" \
                                     "Different assignment of subjects after " \
                                     "loading from disk! - TEST FAILED"

    # check data availability and equality
    assert torch.all(torch.eq(oasis_test_data_c.train_ds[0][0], oasis_test_data_h.train_ds[0][0])), \
        "Test 3 'saving databunch - storing and reloading data':" \
        "Different images when loading from h5py and from MRIDataset! - TEST FAILED"

    oasis_test_data_h.drop_h5py(safe=False)
    assert torch.all(torch.eq(oasis_test_data_c.train_ds[0][0], oasis_test_data_h.train_ds[0][0])), \
        "Test 3 'saving databunch - storing and reloading data':" \
        "Different images after dropping h5py! - TEST FAILED"

    print("Test 3 'saving databunch - storing data ': - TEST PASSED")

    #############################################
    # Test 4 : altering databunch
    #############################################

    oasis_test_data_r = DataBunch(
        source_dir=source_dir,
        image_dir="",
        table=f"OASIS_example.csv",
        separator="\t",
        path=os.path.join(NITORCH_DIR, source_dir, "results"),
        mask=None,
        file_col="path",
        label_col="Age",
        prediction_type="r",
        ptid_col="subjectID",
        random_state=69,
        balance=False,
        val_size=0.1,
        num_samples=16,
        z_factor=0.5,
        transforms=None,
        grouped=True
    )
    # a simple save
    oasis_test_data_r.save(filename="test_4")

    # save the dataframe
    oasis_test_data_r_df = oasis_test_data_r.df_trn

    # change from regression task to classification - not changing anything else
    oasis_test_data_r.label_col = "labels_simplified"
    oasis_test_data_r.prediction_type = "c"
    oasis_test_data_r.labels_to_keep = ["clinical dementia", "healthy"]
    oasis_test_data_r.apply_changes(safe=False)

    assert oasis_test_data_r_df.index.to_list() == oasis_test_data_r.df_trn.index.to_list(), \
        "Test 4 'altering the databunch': Not the same patientsIDs in the splits after a label change. - TEST FAILED"

    assert oasis_test_data_r.train_dl is None, "Test 4 'altering the databunch' : " \
                                               "DataLoader not reset after change! - TEST FAILED"

    oasis_test_data_r.save_df(filename="test_4", separator="\t")

    #############################################
    # Test 4-a : loading databunch from csv
    #############################################

    oasis_test_data_r = DataBunch.from_csv(
        folder=os.path.join(NITORCH_DIR, source_dir, "results"),
        load_filename="test_4",
        cache_filename="test_5",
        source_dir=source_dir,
        image_dir="",
        table=f"OASIS_example.csv",
        separator="\t",
        path=os.path.join(NITORCH_DIR, source_dir, "results"),
        mask=os.path.join(NITORCH_DIR, source_dir, "masks/mask_121_145_121.nii"),
        file_col="path",
        label_col="labels_simplified",
        labels_to_keep=["clinical dementia", "healthy"],
        prediction_type="c",
        ptid_col="subjectID",
        random_state=69,
        balance=True,
        val_size=0.1,
        num_samples=16,
        z_factor=0.5,
        transforms=None,
        grouped=True,
        force_to_shape=[80, 80, 80]
    )
    # change to create a HO set
    ho_size = oasis_test_data_r.hold_out_size
    # change
    oasis_test_data_r.hold_out_size = 0.1
    oasis_test_data_r.hold_out_size = 0.3
    oasis_test_data_r.hold_out_size = 0.15
    # reset changes
    oasis_test_data_r.reset_changes()
    # check
    assert ho_size == oasis_test_data_r.hold_out_size, "Test 4 'altering the databunch' : " \
                                                       "reset_changes() did not correctly reset changes!  - TEST FAILED"
    # decide to do a bigger HO set
    oasis_test_data_r.hold_out_size = 0.2
    oasis_test_data_r.apply_changes(safe=False)
    assert oasis_test_data_r.df_ho is not None, "Test 4 'altering the databunch' : " \
                                                "Post hoc creation of hold_out_data failed!  - TEST FAILED"
    # change z_factor - this affects the mask!
    m = oasis_test_data_r.mask
    oasis_test_data_r.z_factor = 0.8
    oasis_test_data_r.apply_changes(safe=False)
    m2 = oasis_test_data_r.mask

    assert m.shape != m2.shape, "Test 4 'altering the databunch' : " \
                                "changing zooming factor did not change mask adequately!"

    print("Test 4 'altering the databunch': - TEST PASSED")
    print("Test 4a 'loading databunch from csv': - TEST PASSED")
