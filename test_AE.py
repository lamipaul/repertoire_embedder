import matplotlib.pyplot as plt
import models, utils as u
import pandas as pd, numpy as np, torch
import argparse, os
from tqdm import tqdm
from sklearn import metrics
import umap, hdbscan
torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser()
parser.add_argument("specie", type=str)
parser.add_argument("-bottleneck", type=int, default=16)
parser.add_argument("-nMel", type=int, default=128)
parser.add_argument("-encoder", type=str, default='sparrow_encoder')
parser.add_argument("-frontend", type=str, default='logMel')
args = parser.parse_args()

modelname = f'{args.specie}_{args.bottleneck}_{args.frontend}{args.nMel if "Mel" in args.frontend else ""}_{args.encoder}_decod2_BN_nomaxPool.stdc'
meta = models.meta[args.specie]
df = pd.read_csv(f'{args.specie}/{args.specie}.csv')
print(f'Tests for model {modelname}')
print(f'{len(df)} available vocs')

if os.path.isfile(f'{args.specie}/encodings_{modelname[:-4]}npy'):
    dic = np.load(f'{args.specie}/encodings_{modelname[:-4]}npy', allow_pickle=True).item()
    idxs, encodings, X = dic['idxs'], dic['encodings'], dic['umap']
else:
    gpu = torch.device('cuda')
    frontend = models.frontend[args.frontend](meta['sr'], meta['nfft'], meta['sampleDur'], args.nMel)
    encoder = models.__dict__[args.encoder](*((args.bottleneck // 16, (4, 4)) if args.nMel == 128 else (args.bottleneck // 8, (2, 4))))
    decoder = models.sparrow_decoder(args.bottleneck, (4, 4) if args.nMel == 128 else (2, 4))
    model = torch.nn.Sequential(frontend, encoder, decoder).to(gpu)
    model.load_state_dict(torch.load(f'{args.specie}/{modelname}'))
    model.eval()
    loader = torch.utils.data.DataLoader(u.Dataset(df, f'{args.specie}/audio/', meta['sr'], meta['sampleDur']), batch_size=64, shuffle=True, num_workers=8, collate_fn=u.collate_fn)
    with torch.no_grad():
        encodings, idxs = [], []
        for x, idx in tqdm(loader, desc='test '+args.specie, leave=False):
            encoding = model[:2](x.to(gpu))
            idxs.extend(idx)
            encodings.extend(encoding.cpu().detach())

    idxs, encodings = np.array(idxs), np.stack(encodings)
    X = umap.UMAP(n_jobs=-1).fit_transform(encodings)
    np.save(f'{args.specie}/encodings_{modelname[:-4]}npy', {'idxs':idxs, 'encodings':encodings, 'umap':X})

clusters = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5, cluster_selection_epsilon=0.05, core_dist_n_jobs=-1, cluster_selection_method='leaf').fit_predict(X)
df.loc[idxs, 'cluster'] = clusters.astype(int)
mask = ~df.loc[idxs].label.isna()

#print('Found clusters : \n', pd.Series(clusters).value_counts())

plt.figure(figsize=(20, 10))
plt.scatter(X[clusters==-1,0], X[clusters==-1,1], s=2, alpha=.2, color='Grey')
plt.scatter(X[clusters!=-1,0], X[clusters!=-1,1], s=2, c=clusters[clusters!=-1], cmap='tab20')
plt.tight_layout()
plt.savefig(f'{args.specie}/{modelname[:-5]}_projection_clusters.png')

plt.figure(figsize=(20, 10))
plt.scatter(X[~mask,0], X[~mask,1], s=2, alpha=.2, color='Grey')
for l, grp in df.groupby('label'):
    plt.scatter(X[df.loc[idxs].label==l, 0], X[df.loc[idxs].label==l, 1], s=4, label=l)
plt.legend()
plt.tight_layout()
plt.savefig(f'{args.specie}/{modelname[:-5]}_projection_labels.png')


clusters, labels = clusters[mask], df.loc[idxs[mask]].label
print('Silhouette', metrics.silhouette_score(encodings[mask], clusters))
print('NMI', metrics.normalized_mutual_info_score(labels, clusters))
print('Homogeneity', metrics.homogeneity_score(labels, clusters))
print('Completeness', metrics.completeness_score(labels, clusters))
print('V-Measure', metrics.v_measure_score(labels, clusters))

labelled = df[~df.label.isna()]
goodClusters = []
for l, grp in labelled.groupby('label'):
    precisions = grp.groupby('cluster').fn.count() / labelled.groupby('cluster').fn.count()
    best = precisions.idxmax()
    goodClusters.extend(precisions[precisions > 0.9].index)
    print(f'Best precision for {l} is for cluster {best} with {(df.cluster==best).sum()} points, \
    with precision {((labelled.cluster==best)&(labelled.label==l)).sum()/(labelled.cluster==best).sum():.2f}\
     and recall {((labelled.cluster==best)&(labelled.label==l)).sum()/(labelled.label==l).sum():.2f}')


print(f'{len(goodClusters)} clusters would sort {df.cluster.isin(goodClusters).sum()/len(df)*100:.0f}% of samples')
print(f'{len(goodClusters)/df.label.nunique():.1f} cluster per label in avg)')