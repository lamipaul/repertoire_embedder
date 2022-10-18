import os, argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import models, utils as u
import torch

parser = argparse.ArgumentParser()
parser.add_argument("specie", type=str)
parser.add_argument("-nMel", type=int, default=128)
args = parser.parse_args()

meta = models.meta[args.specie]
df = pd.read_csv(f'{args.specie}/{args.specie}.csv')
frontend = models.frontend(meta['sr'], meta['nfft'], meta['sampleDur'], args.nMel)
for label, grp in df.groupby('label'):
    # if os.path.isdir(f'{args.specie}/annot_pngs/{label}'):
        # continue
    os.system(f'mkdir -p {args.specie}/annot_pngs/{label}')
    loader = torch.utils.data.DataLoader(u.Dataset(grp.sample(min(len(grp), 100)), args.specie+'/audio/', meta['sr'], meta['sampleDur']),\
                                         batch_size=1, num_workers=4, pin_memory=True)
    for x, idx in tqdm(loader, desc=args.specie + ' ' + label, leave=False):
        x = frontend(x).squeeze().detach()
        plt.figure()
        plt.imshow(x, origin='lower', aspect='auto')
        plt.savefig(f'{args.specie}/annot_pngs/{label}/{idx.item()}')
        plt.close()



