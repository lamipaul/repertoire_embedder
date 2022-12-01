from sklearn import metrics
import matplotlib.pyplot as plt
import umap, hdbscan
from tqdm import tqdm
import argparse, os
import models, utils as u
import pandas as pd, numpy as np, torch
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument("specie", type=str)
args = parser.parse_args()

df = pd.read_csv(f'{args.specie}/{args.specie}.csv')

meta = models.meta[args.specie]

if not os.path.isfile(f'{args.specie}/encodings/encodings_vggish.npy'):
    gpu = torch.device('cuda')
    frontend = models.frontend['logMel_vggish'](meta['sr'], meta['nfft'], meta['sampleDur'], 64)
    vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
    # vggish.preprocess = False
    vggish.postprocess = False
    model = torch.nn.Sequential(frontend, vggish).to(gpu)
    model.eval()
    loader = torch.utils.data.DataLoader(u.Dataset(df, f'{args.specie}/audio/', 16000, 1), batch_size=1, shuffle=True, num_workers=8, collate_fn=u.collate_fn)
    with torch.no_grad():
        encodings, idxs = [], []
        for x, idx in tqdm(loader, desc='test '+args.specie, leave=False):
            # encoding = model(x.to(gpu))
            encoding = vggish(x.numpy().squeeze(0), fs=16000)
            idxs.extend(idx.numpy())
            encodings.extend(encoding.cpu().numpy())

    idxs, encodings = np.array(idxs), np.stack(encodings)
    X = umap.UMAP(n_jobs=-1).fit_transform(encodings)
    np.save(f'{args.specie}/encodings/encodings_vggish.npy', {'idxs':idxs, 'encodings':encodings, 'umap':X})
else:
    dic = np.load(f'{args.specie}/encodings/encodings_vggish.npy', allow_pickle=True).item()
    idxs, encodings, X = dic['idxs'], dic['encodings'], dic['umap']

clusters = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5, cluster_selection_epsilon=0.05, core_dist_n_jobs=-1, cluster_selection_method='leaf').fit_predict(X)
df.loc[idxs, 'cluster'] = clusters.astype(int)
mask = ~df.loc[idxs].label.isna()

#print('Found clusters : \n', pd.Series(clusters).value_counts())

plt.figure(figsize=(20, 10))
plt.scatter(X[clusters==-1,0], X[clusters==-1,1], s=2, alpha=.2, color='Grey')
plt.scatter(X[clusters!=-1,0], X[clusters!=-1,1], s=2, c=clusters[clusters!=-1], cmap='tab20')
plt.tight_layout()
plt.savefig(f'{args.specie}/projections/vggish_projection_clusters.png')

plt.figure(figsize=(20, 10))
plt.scatter(X[~mask,0], X[~mask,1], s=2, alpha=.2, color='Grey')
for l, grp in df.groupby('label'):
    plt.scatter(X[df.loc[idxs].label==l, 0], X[df.loc[idxs].label==l, 1], s=4, label=l)
plt.legend()
plt.tight_layout()
plt.savefig(f'{args.specie}/projections/vggish_projection_labels.png')


clusters, labels = clusters[mask], df.loc[idxs[mask]].label
print('Silhouette', metrics.silhouette_score(encodings[mask], clusters))
print('NMI', metrics.normalized_mutual_info_score(labels, clusters))
print('Homogeneity', metrics.homogeneity_score(labels, clusters))
print('Completeness', metrics.completeness_score(labels, clusters))
print('V-Measure', metrics.v_measure_score(labels, clusters))

labelled = df[~df.label.isna()]
for l, grp in labelled.groupby('label'):
    best = (grp.groupby('cluster').fn.count() / labelled.groupby('cluster').fn.count()).idxmax()
    print(f'Best precision for {l} is for cluster {best} with {(df.cluster==best).sum()} points, \
with precision {((labelled.cluster==best)&(labelled.label==l)).sum()/(labelled.cluster==best).sum():.2f} and recall {((labelled.cluster==best)&(labelled.label==l)).sum()/(labelled.label==l).sum():.2f}')
