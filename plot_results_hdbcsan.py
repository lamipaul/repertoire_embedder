import hdbscan
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import os

species = np.loadtxt('good_species.txt', dtype=str)
frontends = ['biosound'] #['vggish', '256_logMel128', '256_logSTFT', '256_Mel128', '32_pcenMel128', '64_pcenMel128', '128_pcenMel128', '256_pcenMel128', '512_pcenMel128']

file = open('hdbscan_HP2.txt', 'w')

plt.figure()
for specie in ['humpback', 'dolphin', 'black-headed_grosbeaks', 'california_thrashers']: #species:
    df = pd.read_csv(f'{specie}/{specie}.csv')
    nmis = []
    for i, frontend in tqdm(enumerate(frontends), desc=specie, total=len(frontends)):
        file.write(specie+' '+frontend+'\n')
        fn = f'{specie}/encodings/encodings_' + (f'{specie}_{frontend}_sparrow_encoder_decod2_BN_nomaxPool.npy' if not frontend in ['vggish', 'biosound'] else frontend+'.npy')
        if not os.path.isfile(fn):
            nmis.append(None)
            print('not found')
            continue
        dic = np.load(fn, allow_pickle=True).item()
        idxs,  X = dic['idxs'], dic['umap']

        # clusters = hdbscan.HDBSCAN(min_cluster_size=max(10, len(df)//100), core_dist_n_jobs=-1, cluster_selection_method='leaf').fit_predict(X)
        # df.loc[idxs, 'cluster'] = clusters.astype(int)
        # mask = ~df.loc[idxs].label.isna()
        # clusters, labels = clusters[mask], df.loc[idxs[mask]].label
        # nmis.append(metrics.normalized_mutual_info_score(labels, clusters))
        # df.drop('cluster', axis=1, inplace=True)
        # continue

        nmi = 0
        for mcs in [10, 20, 50, 100, 150, 200]:
            for ms in [None, 3, 5, 10, 20, 30]:
                for eps in [0.0, 0.01, 0.02, 0.05, 0.1]:
                    for al in ['leaf', 'eom']:
                        clusters = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_epsilon=eps, \
                            core_dist_n_jobs=-1, cluster_selection_method=al).fit_predict(X)
                        df.loc[idxs, 'cluster'] = clusters.astype(int)
                        mask = ~df.loc[idxs].label.isna()
                        clusters, labels = clusters[mask], df.loc[idxs[mask]].label
                        tnmi = metrics.normalized_mutual_info_score(labels, clusters)
                        file.write(f'{tnmi} {mcs} {ms} {eps} {al}\n')
                        if tnmi > nmi :
                            nmi
                        df.drop('cluster', axis=1, inplace=True)
        nmis.append(nmi)
    plt.scatter(nmis, np.arange(len(frontends)), label=specie)

file.close()

plt.yticks(range(len(frontends)), frontends)
plt.ylabel('archi')
plt.xlabel('NMI with expert labels')
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig('NMIs_hdbscan.pdf')
plt.close()
