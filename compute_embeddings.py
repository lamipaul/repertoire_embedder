import torch
import models
import numpy as np
import umap
from tqdm import tqdm
import pandas as pd

gpu = torch.device('cuda:1')
modelname = 'AE_marmosset_logMel128_16feat_all-vocs.stdc'
model = torch.nn.Sequential(models.get['frontend_logMel'], models.get['sparrow_encoder'](16), models.get['sparrow_decoder'](16, (4, 2))).eval().to(gpu)
model.load_state_dict(torch.load(modelname))

df = pd.read_pickle('detections.pkl')
#df = df[((df.type!='Noise')&(~df.type.str.contains('-').astype(bool)))]

print('computing encodings')
loader = torch.utils.data.DataLoader(u.Dataset(df, sampleDur=.5), batch_size=16, shuffle=False, num_workers=16, prefetch_factor=8)
with torch.no_grad():
    encodings, idxs = [], []
    for x, idx in tqdm(loader):
        encoding = model[:2](x.to(gpu))
        idxs.extend(idx)
        encodings.extend(encoding.cpu().detach())
idxs = np.array(idxs)
encodings = np.stack(encodings)
print('doing umap...')
X = umap.UMAP(n_jobs=-1).fit_transform(encodings)
np.save('encodings_'+modelname[:-4]+'npy', {'encodings':encodings, 'idx':idxs, 'umap':X})
