import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import torch, hdbscan
import models, utils as u


species = np.loadtxt('good_species.txt', dtype=str)
fig, ax = plt.subplots(nrows=len(species), figsize=(7, 10))
for i, specie in enumerate(species):
    meta = models.meta[specie]
    frontend = models.frontend['pcenMel'](meta['sr'], meta['nfft'], meta['sampleDur'], 128)
    dic = np.load(f'{specie}/encodings//encodings_{specie}_256_pcenMel128_sparrow_encoder_decod2_BN_nomaxPool.npy', allow_pickle=True).item()
    idxs, X = dic['idxs'], dic['umap']
    df = pd.read_csv(f'{specie}/{specie}.csv')
    clusters = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=3, cluster_selection_epsilon=0.05, core_dist_n_jobs=-1, cluster_selection_method='leaf').fit_predict(X)
    df.loc[idxs, 'cluster'] = clusters.astype(int)
    for j, cluster in enumerate(np.random.choice(np.arange(df.cluster.max()), 5)):
        for k, (x, name) in enumerate(torch.utils.data.DataLoader(u.Dataset(df[df.cluster==cluster].sample(10), f'{specie}/audio/', meta['sr'], meta['sampleDur']), batch_size=1)):
            ax[i].imshow(frontend(x).squeeze().numpy(), extent=[k, k+1, j, j+1], origin='lower', aspect='auto')
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    # ax[i].grid(color='w', xdata=np.arange(1, 10), ydata=np.arange(1, 5))
    ax[i].set_ylabel(specie.replace('_', ' '))
    ax[i].set_xlim(0, 10)
    ax[i].set_ylim(0, 5)

plt.tight_layout()
plt.savefig('clusters.pdf')