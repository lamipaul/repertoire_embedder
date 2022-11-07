import hdbscan
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

species = np.loadtxt('good_species.txt', dtype=str)
frontends = ['16_pcenMel128', '16_logMel128', '16_logSTFT', '16_Mel128', '8_pcenMel64', '32_pcenMel128']
plt.figure()
for specie in species:
    df = pd.read_csv(f'{specie}/{specie}.csv')
    nmis = []
    for i, frontend in enumerate(frontends):
        print(specie, frontend)
        dic = np.load(f'{specie}/encodings_{specie}_{frontend}_sparrow_encoder_decod2_BN_nomaxPool.npy', allow_pickle=True).item()
        idxs, encodings, X = dic['idxs'], dic['encodings'], dic['umap']

        clusters = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=20, cluster_selection_epsilon=0.05, core_dist_n_jobs=-1, cluster_selection_method='leaf').fit_predict(X)
        df.loc[idxs, 'cluster'] = clusters.astype(int)
        mask = ~df.loc[idxs].label.isna()
        clusters, labels = clusters[mask], df.loc[idxs[mask]].label
        nmis.append(metrics.normalized_mutual_info_score(labels, clusters))
        df.drop('cluster', axis=1, inplace=True)
    plt.scatter(nmis, np.arange(len(frontends)), label=specie)

plt.yticks(range(len(frontends)), frontends)
plt.ylabel('archi')
plt.xlabel('NMI with expert labels')
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig('NMIs_hdbscan.pdf')
