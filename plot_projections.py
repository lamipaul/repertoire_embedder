import matplotlib.pyplot as plt
import numpy as np, pandas as pd


species = np.loadtxt('good_species.txt', dtype=str)

fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))

non_zero_min = lambda arr: np.min(arr[arr!=0])

for i, specie in enumerate(species):
    dic = np.load(f'{specie}/encodings//encodings_{specie}_256_pcenMel128_sparrow_encoder_decod2_BN_nomaxPool.npy', allow_pickle=True).item()
    df = pd.read_csv(f'{specie}/{specie}.csv')
    df.loc[dic['idxs'], 'umap_x'] = dic['umap'][:,0]
    df.loc[dic['idxs'], 'umap_y'] = dic['umap'][:,1]
    ax[i//4,i%4].scatter(df[df.label.isna()].umap_x, df[df.label.isna()].umap_y, s=2, color='grey')
    for label, grp in df[~df.label.isna()].groupby('label'):
        grp = grp.sample(min(len(grp), 1000))
        ax[i//4,i%4].scatter(grp.umap_x, grp.umap_y, s=2)
    ax[i//4,i%4].set_title(specie.replace('_', ' '))


    sampSize = 100
    X = df.sample(sampSize)[['umap_x', 'umap_y']].to_numpy()
    Y = np.vstack([np.random.normal(np.mean(X[:,0]), np.std(X[:,0]), sampSize), np.random.normal(np.mean(X[:,1]), np.std(X[:,1]), sampSize)]).T
    U = np.sum([min(np.sqrt(np.sum((y - df[['umap_x', 'umap_y']].to_numpy())**2, axis=1))) for y in Y])
    W = np.sum([non_zero_min(np.sqrt(np.sum((x - df[['umap_x', 'umap_y']].to_numpy())**2, axis=1))) for x in X])
    hopkins = U / (U + W)
    ax[i//4, i%4].text(ax[i//4, i%4].get_xlim()[0] + .5, ax[i//4, i%4].get_ylim()[0] + .5, f'{hopkins:.2f}', fontsize=15)

plt.tight_layout()
plt.savefig('projections.pdf')
plt.savefig('projections.png')