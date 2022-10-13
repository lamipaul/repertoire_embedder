import utils as u
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import soundfile as sf
import models
import os
import numpy as np
import pandas as pd
import hdbscan
import argparse
try:
    import sounddevice as sd
    soundAvailable = True
except:
    soundAvailable = False

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='blabla')
parser.add_argument('encodings', type=str)
parser.add_argument('--min_cluster_size', type=int, default=50)
parser.add_argument('--min_sample', type=int, default=10)
parser.add_argument('--eps', type=float, default=1)
args = parser.parse_args()

gpu = torch.device('cuda:0')
frontend = models.get['frontend_logMel'].to(gpu)

a = np.load(args.encodings, allow_pickle=True).item()
df = pd.read_pickle('detections.pkl')
idxs, umap = a['idx'], a['umap']

# cluster the embedings (min_cluster_size and min_samples parameters need to be tuned)
df.loc[idxs, 'cluster'] = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size,
                                          min_samples=args.min_sample,
                                          core_dist_n_jobs=-1,
#                                          cluster_selection_epsilon=args.eps,
                                          cluster_selection_method='leaf').fit_predict(umap)
df.loc[idxs, ['umap_x', 'umap_y']] = umap
df.cluster = df.cluster.astype(int)

figscat = plt.figure(figsize=(20, 10))
plt.title(f'{args.encodings} {args.min_cluster_size} {args.min_sample} {args.eps}')
for c, grp in df.groupby('cluster'):
    plt.scatter(grp.umap_x, grp.umap_y, s=3, alpha=.1, c='grey' if c == -1 else None)
#plt.scatter(df[((df.type!='Vocalization')&(~df.type.isna()))].umap_x, df[((df.type!='Vocalization')&(~df.type.isna()))].umap_y, marker='x')
plt.tight_layout()
axScat = figscat.axes[0]
plt.savefig('projection')
figSpec = plt.figure()
plt.scatter(0, 0)
axSpec = figSpec.axes[0]

print(df.groupby('cluster').count())

class temp():
    def __init__(self):
        self.row = ""
    def onclick(self, event):
        #get row
        left, right, bottom, top = axScat.get_xlim()[0], axScat.get_xlim()[1], axScat.get_ylim()[0], axScat.get_ylim()[1]
        rangex, rangey =  right - left, top - bottom
        closest = (np.sqrt(((df.umap_x - event.xdata)/rangex)**2 + ((df.umap_y  - event.ydata)/rangey)**2)).idxmin()
        sig, fs = sf.read(f'/data_ssd/marmossets/{closest}.wav')
        spec = frontend(torch.Tensor(sig).to(gpu).view(1, -1).float()).detach().cpu().squeeze()
        axSpec.imshow(spec, origin='lower', aspect='auto')
        row = df.loc[closest]
        axSpec.set_title(f'{closest}, cluster {row.cluster} ({(df.cluster==row.cluster).sum()} points)')
        axScat.scatter(row.umap_x, row.umap_y, c='r')
        axScat.set_xlim(left, right)
        axScat.set_ylim(bottom, top)
        figSpec.canvas.draw()
        figscat.canvas.draw()
        if soundAvailable:
            sd.play(sig*10, fs)

mtemp = temp()

cid = figscat.canvas.mpl_connect('button_press_event', mtemp.onclick)

plt.show()

if input('print cluster pngs ??') != 'y':
    exit()

os.system('rm -R cluster_pngs/*')
for c, grp in df.groupby('cluster'):
    if c == -1 or len(grp) > 10_000:
        continue
    os.system('mkdir cluster_pngs/'+str(c))
    with torch.no_grad():
        for x, idx in tqdm(torch.utils.data.DataLoader(u.Dataset(grp.sample(200), sampleDur=.5), batch_size=1, num_workers=8), leave=False, desc=str(c)):
            x = x.to(gpu)
            x = frontend(x).cpu().detach().squeeze()
            plt.imshow(x, origin='lower', aspect='auto')
            plt.savefig(f'cluster_pngs/{c}/{idx.squeeze().item()}')
            plt.close()
