import hdbscan
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics, cluster
from scipy.stats import linregress

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

        ks = (5*1.2**np.arange(20)).astype(int)
        distorsions = [cluster.KMeans(n_clusters=k).fit(encodings).inertia_ for k in ks]
        errors = [linregress(ks[:i], distorsions[:i]).stderr + linregress(ks[i+1:], distorsions[i+1:]).stderr for i in range(2, len(ks)-2)]
        k = ks[np.argmin(errors)]
        clusters = cluster.KMeans(n_clusters=k).fit_predict(encodings)
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
plt.legend()
plt.tight_layout()
plt.savefig('NMIs_kmeans.pdf')
