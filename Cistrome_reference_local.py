import numpy as np
import pandas as pd
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import seaborn as sns
import pickle

path = "/Users/chen-haochen/Dropbox/Cistrome_imputation/scATAC/VAE/data"

def plot_embedding(X, labels, classes=None, method='tSNE', cmap='tab20', figsize=(4, 4), markersize=10, marker=None,
                   return_emb=False, save=False, save_emb=False, show_legend=True, show_axis_label=True, **legend_params):
    if marker is not None:
        X = np.concatenate([X, marker], axis=0)
    N = len(labels)
    if X.shape[1] != 2:
        if method == 'tSNE':
            from sklearn.manifold import TSNE
            X = TSNE(n_components=2, random_state=124).fit_transform(X)
        if method == 'UMAP':
            from umap import UMAP
            X = UMAP(n_neighbors=30, min_dist=0.3, metric='correlation').fit_transform(X)
        if method == 'PCA':
            from sklearn.decomposition import PCA
            X = PCA(n_components=2, random_state=124).fit_transform(X)
    #print(X)
    plt.figure(figsize=figsize)
    if classes is None:
        classes = np.unique(labels)
    if cmap is not None:
        cmap = cmap
    elif len(classes) <= 10:
        cmap = 'tab10'
    elif len(classes) <= 20:
        cmap = 'tab20'
    else:
        cmap = 'husl'
    colors = sns.color_palette(cmap, n_colors=len(classes))

    for i, c in enumerate(classes):
        label_index = [i for i,j in enumerate(labels) if j==c]
        if "BMMC" in c:
            plt.scatter(X[label_index, 0], X[label_index, 1], s=30,marker="^", color=colors[i], label=c)
        else:
            plt.scatter(X[label_index, 0], X[label_index, 1], s=markersize, color=colors[i], label=c)

    legend_params_ = {'loc': 'center left',
                     'bbox_to_anchor':(1.0, 0.45),
                     'fontsize': 10,
                     'ncol': 1,
                     'frameon': False,
                     'markerscale': 1.5
                    }
    legend_params_.update(**legend_params)
    if show_legend:
        plt.legend(**legend_params_)
    sns.despine(offset=10, trim=True)
    if show_axis_label:
        plt.xlabel(method+' dim 1', fontsize=12)
        plt.ylabel(method+' dim 2', fontsize=12)

    if save:
        plt.savefig(save, format='pdf', bbox_inches='tight')
    else:
        plt.show()

def VAE_plot():
    samples = pickle.load(open("/%s/DNase_samples.p" %(path),'rb'))
    features = pd.read_csv("/%s/VAE_test_v3_feature.txt" %(path),sep="\t",header=None,index_col=0)
    features.index = samples

    Tissues = []
    celltypes = []
    anno = pd.read_csv("/Users/chen-haochen/Dropbox/Cistrome_imputation/scATAC/giggle_justification/DC_haveProcessed_20190506_filepath_qc.xls",sep="\t",header=0,index_col=0)
    for sample in samples:
        Tissues.append(anno.at[int(sample),"Tissue"])
        celltypes.append(anno.at[int(sample),"CellType"])
    features["Tissue"] = Tissues
    features["celltypes"] = celltypes

    features = features.loc[features["Tissue"].isin(["Blood","Lung","Brain","Breast","Kidney","Skin","Heart","Thymus","Stomach","Prostate"])]
    labels = list(features["Tissue"])
    #features = features.loc[features["Tissue"]=="Blood"]
    #labels = list(features["celltypes"])

    outdir = path
    plot_embedding(features[[1,2,3,4,5,6,7,8,9,10]].values, labels,save=os.path.join(path, 'Cistrome_embedding_UMAP.pdf'), save_emb=os.path.join(outdir, 'tsne.txt'))

def sc_plot():
    samples = pickle.load(open("/%s/DNase_samples.p" %(path),'rb'))
    features = pd.read_csv("/%s/VAE_test_v3_feature.txt" %(path),sep="\t",header=None,index_col=0)
    features.index = samples

    Tissues = []
    celltypes = []
    anno = pd.read_csv("/Users/chen-haochen/Dropbox/Cistrome_imputation/scATAC/giggle_justification/DC_haveProcessed_20190506_filepath_qc.xls",sep="\t",header=0,index_col=0)
    for sample in samples:
        Tissues.append(anno.at[int(sample),"Tissue"])
        celltypes.append(anno.at[int(sample),"CellType"])
    features["Tissue"] = Tissues
    features["celltypes"] = celltypes
    features = features.loc[features["Tissue"]=="Blood"]
    for type in ["BMMC_healthy","BMMC_PBMC","BMMC_CLL"]:
        sc_features = pd.read_csv("/%s/%s_VAE_test_v3_specific_feature.txt" %(path,type),sep="\t",header=None,index_col=0)
        sc_features["Tissue"] = sc_features.index.tolist()
        sc_features["celltypes"] = sc_features.index.tolist()
        new_features = pd.concat([features,sc_features],axis=0)
        labels = list(new_features["celltypes"])
        outdir = path
        plot_embedding(new_features[[1,2,3,4,5,6,7,8,9,10]].values, labels,save=os.path.join(path, 'Cistrome_embedding_UMAP_%s.pdf' %(type)), save_emb=os.path.join(outdir, 'tsne.txt'))

def main():
    VAE_plot()
    #sc_plot()

main()
