import random
import h5py
import numpy as np
import sys
import os
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms

# load functions from nitorch
sys.path.insert(0,"/analysis/fabiane/phd/nitorch/")
from nitorch.callbacks import EarlyStopping, ModelCheckpoint
from nitorch.transforms import IntensityRescale, ToTensor
from nitorch.trainer import Trainer
from nitorch.utils import count_parameters
from nitorch.metrics import balanced_accuracy, sensitivity, specificity

from dataset_classes import ADNIDataset, UKBDataset, MSDataset
from utils import update_outer_scores
from collections import OrderedDict

mask = None
dtype = np.float64

def pick_model(name, dataset):
    # import correct models for the data set
    if dataset == "ADNI":
        from models_adni import ModelA, ModelB, ModelAPIF, ModelBPIF
    elif dataset == "MS":
        from models_ms import ModelA, ModelB, ModelAPIF, ModelBPIF
    elif dataset == "UKB":
        from models_ukb import ModelA, ModelB, ModelAPIF, ModelBPIF
    
    # choose the specific model
    print(f"Picking model {name}")
    if name == "ModelA":
        net = ModelA()
    elif name == "ModelB":
        net = ModelB()
    elif name == "ModelAPIF":
        net = ModelAPIF()
    elif name == "ModelBPIF":
        net = ModelBPIF()
    return net

def run(
    data,
    shape,
    train_data_path,
    hyperparams,
    dataset,
    experiment,
    output_dir,
    augmentations=[],
    masked=False,
    zero_one_normalize=False,
    metrics=[],
    subset_only=False,
    k_folds=None,
    b=4,
    wd=1e-4,
    num_epochs=35,
    ignore_epochs=5,
    retain_metric=None,
    gpu=0,
    show_train_steps=50
):      
   
    fold_metric = []
    all_outer_bests = []
    best_metric = []
    models = []
    fold = 0
    initial_prepend = None
    
    random_states = np.arange(20, 30) # will be used again to split in testing phase
    
    for outer_fold, r in enumerate(random_states):
        print("###### Starting outer fold {}".format(outer_fold))
        
        ### Load ADNI data set
        if dataset == "ADNI":
            # load H5 files with random state 'r' being in the file name
            train_h5 = h5py.File(train_data_path + str(r) + ".h5", 'r')
            val_h5 = h5py.File(train_data_path.replace("train", "val") + str(r) + ".h5", 'r')

            X_train, y_train = np.array(train_h5['X']), np.array(train_h5['y'])
            X_val, y_val = val_h5['X'], val_h5['y']

            if subset_only:
                # select only 20 % 
                X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=0.2, random_state=42)
                
            if zero_one_normalize:
                intensity = IntensityRescale(data_min=np.min(X_train), data_max=np.max(X_train)) # use the same min and max for all datasets
                data_train = ADNIDataset(X_train, y_train, transform=transforms.Compose(augmentations + [intensity, ToTensor()]), mask=mask, dtype=dtype)
                data_val = ADNIDataset(X_val, y_val, transform=transforms.Compose([intensity, ToTensor()]), mask=mask, dtype=dtype)
            else:
                data_train = ADNIDataset(X_train, y_train, transform=transforms.Compose(augmentations + [ToTensor()]), mask=mask, dtype=dtype)
                data_val = ADNIDataset(X_val, y_val, transform=transforms.Compose([ToTensor()]), mask=mask, dtype=dtype)
            
            # dataset overview
            y = np.concatenate([np.array(y_train), np.array(y_val)])
            print("# of patients {}, # of healthy controls {}".format(np.sum(y), len(y)-np.sum(y)))
            print(len(y_train))
            print(len(y_val))
            # close h5 files
            train_h5.close()
            val_h5.close()
                
        ### Load UKB data set
        elif dataset == "UKB":
            # load new random data split each outer fold
            # load dataset as DataFrame
            assert(data is not None)
            df = pd.read_csv(data)
            # shuffle the dataset and take subset
            if subset_only:
                df = df.sample(frac=0.2, random_state=23)
            else:
                df = df.sample(frac=1, random_state=23)
            
            # separate new test set
            no_test_df = df.sample(frac=0.8, random_state=r) # split according 
             # split train/validation
            train_df = no_test_df.sample(frac=0.8, random_state=32) # can be the same r at each inner fold
            val_df = no_test_df[~no_test_df["PTID"].isin(train_df["PTID"])]

            if zero_one_normalize:
                intensity = IntensityRescale()
                data_train = UKBDataset(train_df, transform=transforms.Compose(augmentations + [intensity, ToTensor()]), mask=mask)
                data_val = UKBDataset(val_df, transform=transforms.Compose([intensity, ToTensor()]), mask=mask)
            else:
                data_train = UKBDataset(train_df, transform=transforms.Compose(augmentations + [ToTensor()]), mask=mask)
                data_val = UKBDataset(val_df, transform=transforms.Compose([ToTensor()]), mask=mask)

            print(len(data_train))
            print(len(data_val))
            
        ### Loas MS data
        elif dataset == "MS":
            # the MS data's original train/holdout split will be ignored
            # in order to create 10 individual splits
            train_h5 = h5py.File(train_data_path, 'r')
            holdout_h5 = h5py.File(train_data_path.replace("train", "holdout"), 'r')
            X_holdout, y_holdout = holdout_h5['X'], holdout_h5['y']
            X_train, y_train = train_h5['X'], train_h5['y']
            # concatenate the old splits
            X = np.concatenate([X_train, X_holdout])
            y = np.concatenate([y_train, y_holdout])
            # separate 85% as the training set
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.15, random_state=r)
            # normalize
            if zero_one_normalize:
                intensity = IntensityRescale()
                data_train = MSDataset(X_train, y_train, transform=transforms.Compose([intensity, ToTensor()]), mask=mask, dtype=dtype)
            else:
                data_train = MSDataset(X_train, y_train, transform=transforms.Compose([ToTensor()]), mask=mask, dtype=dtype)
            # separate 10% for the validation set
            all_indices = list(range(len(X_train)))
            train_idx, val_idx = train_test_split(all_indices, test_size=0.1, random_state=12)
            print(len(train_idx))
            print(len(val_idx))
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

        
        outer_fold_best = {
            "final_acc" : 0,
            "best_acc" : 0,
            "final_iter" : 0,
            "best_iter" : 0,
            "params" : {},
            "inner_fold_idx" : 0
        }
        
        for inner_fold in range(5):
            print("###### Starting inner fold {}".format(inner_fold))
            # sample randomly from the hyperparameter options each inner fold
            b = random.choice(hyperparams["batch_size"])
            patience = random.choice(hyperparams["patience"]["small"])
            lr = random.choice(hyperparams["lr"])
            
            # force each model architecture to occur twice
            if inner_fold < 2:
                model_name = "ModelA"
            elif inner_fold < 4:
                model_name = "ModelB"
            else:
                # only choose model randomly in the last run
                model_name = random.choice(hyperparams["model"])
                
            # extensions for PIF and patch based
            if experiment == "PIF":
                model_name = model_name + "PIF"
            elif experiment == "LIU":
                model_name = model_name + "LIU"

            selected_hyperparams = {
                "model" : model_name,
                "batch_size" : b,
                "patience" : patience,
                "lr" : lr
                }
            print(selected_hyperparams)
            
            # load model
            if isinstance(gpu, list):
                net = torch.nn.DataParallel(pick_model(model_name, dataset), device_ids=gpu).cuda(gpu[0])
            else:
                net = pick_model(model_name, dataset).cuda(gpu)
                
            
            print("Trainable model parameters: {}".format(count_parameters(net)))
        
            callbacks = [EarlyStopping(patience=patience, ignore_before=ignore_epochs, retain_metric="loss", mode='min'),
                         ModelCheckpoint(path=output_dir,
                                         prepend="outer_{}_inner_{}".format(outer_fold, inner_fold),
                                         store_best=True,
                                         ignore_before=ignore_epochs,
                                         retain_metric=retain_metric)]
        
            # add current fold number to model checkpoint path
            if callbacks is not None:
                for idx, callback in enumerate(callbacks):
                    if isinstance(callback, ModelCheckpoint):
                        if initial_prepend is None:
                            initial_prepend = callbacks[idx].prepend
                        callbacks[idx].prepend = initial_prepend + "cv_fold_{}_".format(fold)
                        initial_prepend = None
            fold += 1
            
            # set optimizer
            criterion = nn.BCEWithLogitsLoss().cuda(gpu)
            optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
            
            # prepate data loaders
            if dataset == "MS":
                train_loader = DataLoader(
                    data_train, batch_size=b, num_workers=4, sampler=train_sampler
                )
                val_loader = DataLoader(
                    data_train, batch_size=1, num_workers=1, sampler=val_sampler
                )
            else:
                train_loader = DataLoader(
                    data_train, batch_size=b, num_workers=4, shuffle=True
                )
                val_loader = DataLoader(
                    data_val, batch_size=1, num_workers=1, shuffle=True
                )
        
            # setup nitorch trainer class
            trainer = Trainer(
                net,
                criterion,
                optimizer,
                metrics=metrics,
                callbacks=callbacks,
                device=gpu
            )
            # train model and store results
            net, report = trainer.train_model(
                train_loader,
                val_loader,
                num_epochs=num_epochs,
                show_train_steps=show_train_steps,
                show_validation_epochs=1,
            )
            # append validation score of the retain metric
            if isinstance(retain_metric, str):
                # use the average of the last 10 epochs
                best_val = np.mean(report["val_metrics"][retain_metric][-10:])
                fold_metric.append(report["val_metrics"][retain_metric][-1])
                best_metric.append(best_val)
                # replace best model for this outer_fold in case of improvement
                #if report["val_metrics"][retain_metric][-1] > outer_fold_best["final_acc"]:
                if report["val_metrics"][retain_metric][-1] > outer_fold_best["final_acc"]:
                    update_outer_scores(outer_fold_best, report, retain_metric, selected_hyperparams, inner_fold)
            else:
                # use the average of the last 10 epochs
                best_val = np.mean(report["val_metrics"][retain_metric.__name__][-10:])
                fold_metric.append(report["val_metrics"][retain_metric.__name__][-1])
                best_metric.append(best_val)
                # replace best model for this outer_fold in case of improvement
                #if report["val_metrics"][retain_metric.__name__][-1] > outer_fold_best["final_acc"]:
                if best_val > outer_fold_best["final_acc"]:
                    update_outer_scores(outer_fold_best, report, retain_metric.__name__, selected_hyperparams, inner_fold, ignore_epochs)
            models.append(net)        
            print("Finished inner fold.")

            # visualize result
            trainer.visualize_training(report, metrics)
            trainer.evaluate_model(val_loader, gpu)
        print("Finished outer fold.")
        print(outer_fold_best)
        all_outer_bests.append(outer_fold_best)

    print("################################")
    print("################################")
    print("All accuracies: {}".format(fold_metric))
    print("Best accuracies mean: {} All :{}".format(np.mean(best_metric), best_metric))
    print(all_outer_bests)
    return fold_metric, models, all_outer_bests


def run_inference(training_res,
                  train_data_path,
                  output_dir,
                  dataset,
                  experiment,
                  gpu,
                  data=None,
                  zero_one_normalize=False):
    metrics = []
    random_states = np.arange(20, 30) # important to use the same values as during training

    for outer_fold, r in enumerate(random_states):
        print("Outer fold {}".format(outer_fold))

        all_scores = []
        all_preds = []
        all_labels = []

        train_res = training_res[outer_fold]
        model_dir = os.path.join(output_dir,"outer_{}_inner_{}_cv_fold_{}_BEST_ITERATION.h5".format(outer_fold, train_res["inner_fold_idx"], outer_fold * 5 + train_res["inner_fold_idx"]))
        model_name = train_res["params"]["model"]
        """if experiment == "PIF":
            model_name = model_name + "PIF"
        elif experiment == "LIU":
            model_name = model_name + "LIU"
        """
        net = pick_model(model_name, dataset)
        state_dict = torch.load(model_dir, map_location='cpu')
        new_state_dict = OrderedDict()
        for param in state_dict.items():
            new_state_dict.update([(param[0].replace("module.", ""), param[1])])

        net.load_state_dict(new_state_dict)
        net = net.cuda(gpu)
        net.eval()

        ### Load ADNI data set
        if dataset == "ADNI":
            test_h5 = h5py.File(train_data_path.replace("train", "holdout") + str(r) + ".h5", 'r')

            X_test, y_test = np.array(test_h5['X']), np.array(test_h5['y'])

            # dataset overview
            print("# of patients {}, # of healthy controls {}".format(np.sum(y_test), len(y_test)-np.sum(y_test)))

            print(len(y_test))

            if zero_one_normalize:
                intensity = IntensityRescale(data_min=np.min(X_train), data_max=np.max(X_train)) # use the same min and max for all datasets
                data_test= ADNIDataset(X_test, y_test, transform=transforms.Compose([intensity, ToTensor()]), mask=mask, dtype=dtype)
            else:
                data_test = ADNIDataset(X_test, y_test, transform=transforms.Compose([ToTensor()]), mask=mask, dtype=dtype)

            test_h5.close()
        elif dataset == "UKB":
            ### Load UKB data set
            assert(data is not None)
            df = pd.read_csv(data)
            # separate new test set each run
            no_test_df = df.sample(frac=0.8, random_state=r) # split according 
            test_df = df[~df["PTID"].isin(no_test_df["PTID"])]

            print(len(test_df))

            if zero_one_normalize:
                intensity = IntensityRescale()
                data_test = UKBDataset(test_df, transform=transforms.Compose([intensity, ToTensor()]), mask=mask)
            else:
                data_test = UKBDataset(test_df, transform=transforms.Compose([ToTensor()]), mask=mask)


        test_loader = DataLoader(
                data_test, batch_size=1, num_workers=1, shuffle=False)

        with torch.no_grad():
            for sample in test_loader:
                img = sample["image"]
                label = sample["label"]
                img = img.to(torch.device("cuda:" + str(gpu)))

                output = net.forward(img)
                score = torch.sigmoid(output)
                pred = score >= 0.5

                all_scores.append(score.cpu().numpy().item())
                all_preds.append(pred.cpu().numpy().item())
                all_labels.append(label.numpy().item())

        balanced_acc = balanced_accuracy(all_labels, all_preds)
        roc_score = roc_auc_score(all_labels, all_scores)
        sens = sensitivity(all_labels, all_preds)
        spec = specificity(all_labels, all_preds)
        print("Bal. accuracy {:.3f} %".format(balanced_acc*100))
        print("ROC AUC Score {:.3f} %".format(roc_score*100))
        print("Sensitivity {:.3f} %".format(sens*100))
        print("Specificity {:.3f} %".format(spec*100))
        metrics.append([balanced_acc, roc_score, sens, spec, all_labels, all_scores])
    # reset network to training mode
    net.train()
    print("######## Final results ########")
    metrics_df = pd.DataFrame(metrics, columns=["Bal Acc", "ROC AUC", "Sensitivity", "Specificity", "Labels", "Scores"])
    print(metrics_df)
    print("Balanced accuracy mean {:.3f} %".format(np.mean(metrics_df["Bal Acc"])*100))
    print("ROC AUC mean {:.3f} %".format(np.mean(metrics_df["ROC AUC"])*100))
    print("Sensitivity mean {:.3f} %".format(np.mean(metrics_df["Sensitivity"])*100))
    print("Specificity mean {:.3f} %".format(np.mean(metrics_df["Specificity"])*100))
    return metrics_df
