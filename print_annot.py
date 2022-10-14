from tqdm import tqdm
import os
import models
import matplotlib.pyplot as plt
import pandas as pd
import models
import utils as u
import torch

for specie in models.meta:
    meta = models.meta[specie]
    df = pd.read_csv(f'{specie}/{specie}.csv')
    frontend = models.frontend(meta['sr'], meta['nfft'], meta['sampleDur'])
    for label, grp in df.groupby('label'):
        if os.path.isdir(f'annot_pngs/{specie}/{label}'):
            continue
        os.system(f'mkdir -p annot_pngs/{specie}/{label}')
        loader = torch.utils.data.DataLoader(u.Dataset(grp.sample(min(len(grp), 100)), specie+'/audio/', meta['sr'], meta['sampleDur']), batch_size=1, num_workers=4, pin_memory=True)
        for x, idx in tqdm(loader, desc=specie + ' ' + label, leave=False):
            x = frontend(x).squeeze().detach()
            plt.figure()
            plt.imshow(x, origin='lower', aspect='auto')
            plt.savefig(f'annot_pngs/{specie}/{label}/{idx.item()}')
            plt.close()



