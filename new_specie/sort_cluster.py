import utils as u
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch, numpy as np, pandas as pd
import hdbscan
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, \
    description="""Interface to visualize projected vocalizations (UMAP reduced AE embeddings), and tune HDBSCAN parameters.\n
    This script is to be called after compute_embeddings.py.\n
    If satisfying HDBCSCAN parameters are reached, vocalisations spectrograms can be plotted (sorted by clusters) by typing \'y\' after closing the projection plot.\n
    For insights on how to tune HDBSCAN parameters, read https://hdbscan.readthedocs.io/en/latest/parameter_selection.html""")
parser.add_argument('encodings', type=str, help='.npy file containing umap projections and their associated index in the detection.pkl table (built using compute_embeddings.py)')
parser.add_argument('detections', type=str, help=".csv file with detections to be encoded. Columns filename (path of the soundfile) and pos (center of the detection in seconds) are needed")
parser.add_argument('audio_folder', type=str, help='Path to the folder with complete audio files')
parser.add_argument("-SR", type=int, default=44100, help="Sample rate of the samples before spectrogram computation")
parser.add_argument("-sampleDur", type=float, default=1, help="Size of the signal extracts surrounding detections to be encoded")
parser.add_argument('-min_cluster_size', type=int, default=50, help='Used for HDBSCAN clustering.')
parser.add_argument('-min_sample', type=int, default=5, help='Used for HDBSCAN clustering.')
parser.add_argument('-eps', type=float, default=0.0, help='Used for HDBSCAN clustering.')
args = parser.parse_args()

gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv(args.detections)
encodings = np.load(args.encodings, allow_pickle=True).item()
idxs, umap = encodings['idx'], encodings['umap']
# Use HDBSCAN to cluster the embedings (min_cluster_size and min_samples parameters need to be tuned)
df.at[idxs, 'cluster'] = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size,
                                min_samples=args.min_sample,
                                core_dist_n_jobs=-1,
                                cluster_selection_epsilon=args.eps,
                                cluster_selection_method='leaf').fit_predict(umap)
df.cluster = df.cluster.astype(int)

figscat = plt.figure(figsize=(10, 5))
plt.title(f'{args.encodings} {args.min_cluster_size} {args.min_sample} {args.eps}')
plt.scatter(umap[:,0], umap[:,1], s=3, alpha=.2, c=df.loc[idxs].cluster, cmap='tab20')
plt.tight_layout()
plt.savefig('projection')
plt.show()

if input('\nType y to print cluster pngs.\n/!\ the cluster_pngs folder will be reset, backup if needed /!\ ') != 'y':
    exit()

os.system('rm -R cluster_pngs/*')

for c, grp in df.groupby('cluster'):
    if c == -1 or len(grp) > 10_000:
        continue
    os.system('mkdir -p cluster_pngs/'+str(c))
    loader = torch.utils.data.DataLoader(u.Dataset(grp.sample(min(len(grp), 200)), args.audio_folder, args.SR, args.sampleDur), batch_size=1, num_workers=8)
    with torch.no_grad():
        for x, idx in tqdm(loader, leave=False, desc=str(c)):
            plt.specgram(x.squeeze().numpy())
            plt.tight_layout()
            plt.savefig(f'cluster_pngs/{c}/{idx.squeeze().item()}')
            plt.close()
