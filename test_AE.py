import matplotlib.pyplot as plt
import models, utils as u
import pandas as pd, numpy as np, torch
import argparse
from tqdm import tqdm
from sklearn import metrics
import umap, hdbscan

parser = argparse.ArgumentParser()
parser.add_argument("specie", type=str)
parser.add_argument("-bottleneck", type=int, default=16)
parser.add_argument("-nMel", type=int, default=128)
args = parser.parse_args()

modelname = f'{args.specie}_{args.bottleneck}_Mel{args.nMel}.stdc'
gpu = torch.device('cuda')
meta = models.meta[args.specie]
frontend = models.frontend(meta['sr'], meta['nfft'], meta['sampleDur'], args.nMel)
encoder = models.sparrow_encoder(args.bottleneck)
decoder = models.sparrow_decoder(args.bottleneck, (4, 4) if args.nMel == 128 else (2, 4))
model = torch.nn.Sequential(frontend, encoder, decoder).to(gpu)
model.load_state_dict(torch.load(f'{args.specie}/{modelname}'))

df = pd.read_csv(f'{args.specie}/{args.specie}.csv')
print(f'{len(df)} available vocs')

loader = torch.utils.data.DataLoader(u.Dataset(df, f'{args.specie}/audio/', meta['sr'], meta['sampleDur']), \
                               batch_size=64, shuffle=True, num_workers=8, collate_fn=u.collate_fn)

model.eval()
with torch.no_grad():
    encodings, idxs = [], []
    for x, idx in tqdm(loader, desc='test '+args.specie, leave=False):
        encoding = model[:2](x.to(gpu))
        idxs.extend(idx)
        encodings.extend(encoding.cpu().detach())
idxs = np.array(idxs)
encodings = np.stack(encodings)
X = umap.UMAP(n_jobs=-1).fit_transform(encodings)
clusters = hdbscan.HDBSCAN(min_cluster_size=len(df)//100, min_samples=5, core_dist_n_jobs=-1, cluster_selection_method='leaf').fit_predict(X)

print('NMI', metrics.normalized_mutual_info_score(df.loc[idxs].label, clusters))
print('Homogeneity', metrics.homogeneity_score(df.loc[idxs].label, clusters))
print('Completeness', metrics.completeness_score(df.loc[idxs].label, clusters))
print('V-Measure', metrics.v_measure_score(df.loc[idxs].label, clusters))

plt.figure(figsize=(10, 5))
plt.scatter(X[clusters!=-1,0], X[clusters!=-1,1], s=5, alpha=.2, c=clusters, cmap='tab20')
plt.scatter(X[clusters==-1,0], X[clusters==-1,1], s=5, alpha=.2, color='Grey')
plt.tight_layout()
plt.savefig(f'{args.specie}/{modelname[:-5]}_projection.pdf')