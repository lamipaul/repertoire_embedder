import hdbscan
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

species = np.loadtxt('good_species.txt', dtype=str)
frontends = ['16_pcenMel128', '16_logMel128', '16_logSTFT', '16_Mel128', '8_pcen64', '32_pcenMel128']
plt.figure()
for specie in species:
    df = pd.read_csv(f'{specie}/{specie}.csv')
    for i, frontend in enumerate(frontends):
        print(specie, frontend)
        dic = np.load(f'{specie}/encodings_{specie}_{frontend}_sparrow_encoder_decod2_BN_nomaxPool.npy', allow_pickle=True).item()
        idxs, encodings, X = dic['idxs'], dic['encodings'], dic['umap']
        
        clusters = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0, core_dist_n_jobs=-1, cluster_selection_method='best').fit_predict(X)
        df.loc[idxs, 'cluster'] = clusters.astype(int)
        mask = ~df.loc[idxs].label.isna()
        clusters, labels = clusters[mask], df.loc[idxs[mask]].label
        plt.scatter(metrics.normalized_mutual_info_score(labels, clusters), i, label=specie)
        df.drop('cluster', inplace=True)

plt.ytick_labels(range(len(frontends)), frontends)
plt.ylabel('archi')
plt.xlabel('NMI with expert labels')
plt.grid()
plt.legend()
plt.savefig('NMIs_hdbscan.pdf')