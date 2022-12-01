import os, argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import models, utils as u
import torch

parser = argparse.ArgumentParser()
parser.add_argument("specie", type=str)
parser.add_argument("-frontend", type=str, default='logMel')
parser.add_argument("-nMel", type=int, default=128)
args = parser.parse_args()

meta = models.meta[args.specie]
df = pd.read_csv(f'{args.specie}/{args.specie}.csv')
frontend = models.frontend[args.frontend](meta['sr'], meta['nfft'], meta['sampleDur'], args.nMel)
os.system(f'rm -R {args.specie}/annot_pngs/*')
for label, grp in df.groupby('label'):
    os.system(f'mkdir -p "{args.specie}/annot_pngs/{label}"')
    loader = torch.utils.data.DataLoader(u.Dataset(grp, args.specie+'/audio/', meta['sr'], meta['sampleDur']),\
                                         batch_size=1, num_workers=4, pin_memory=True)
    for x, idx in tqdm(loader, desc=args.specie + ' ' + label, leave=False):
        x = frontend(x).squeeze().detach()
        assert not torch.isnan(x).any(), "Found a NaN in spectrogram... :/"
        plt.figure()
        plt.imshow(x, origin='lower', aspect='auto')
        row = df.loc[idx.item()]
        plt.savefig(f'{args.specie}/annot_pngs/{label}/{row.fn.split(".")[0]}_{row.pos:.2f}.png')
        # plt.savefig(f'{args.specie}/annot_pngs/{label}/{idx.item()}')
        plt.close()



