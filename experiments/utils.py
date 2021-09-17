import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('ggplot')
# uncomment the following lines for latex saving
"""
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size':8,
    'text.usetex': True,
    'pgf.rcfonts': False,
})"""

def plot_roc_curve(metrics_df, title="Receiver operating characteristic example", path=None):
    # Computing and plotting the ROC curve
    
    # preparing the variables
    fpr = dict()
    tpr = dict()
    roc_auc = np.zeros(shape=(len(metrics_df),))
    # preparing the Figure
    plt.figure()
    lw = 2

    # looping through all runs
    for i in range(len(metrics_df)):
        # computing the FPR/TPR and area under the curve
        fpr[i], tpr[i], _ = roc_curve(metrics_df["Labels"].iloc[i], metrics_df["Scores"].iloc[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # plotting each run
        plt.plot(fpr[i], tpr[i],
                 lw=lw, label=f"ROC curve run {i} (area = {roc_auc[i]:0.3f})")

    # finishing the plot
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    #plt.show()
    if path is not None:
        plt.savefig(path, bbox_inches='tight', dpi=250)
        

def update_outer_scores(outer_fold_best, report, retain_metric, selected_hyperparams, inner_fold_idx, ignore_epochs=5):
    outer_fold_best["final_acc"] = report["val_metrics"][retain_metric][-1]
    outer_fold_best["best_acc"] = np.max(report["val_metrics"][retain_metric][ignore_epochs:])
    outer_fold_best["final_iter"] = len(report["val_metrics"][retain_metric])
    outer_fold_best["best_iter"] = outer_fold_best["final_iter"] - np.argmax(np.flip(np.copy(report["val_metrics"][retain_metric])))
    outer_fold_best["params"] = selected_hyperparams
    outer_fold_best["inner_fold_idx"] = inner_fold_idx
    return outer_fold_best