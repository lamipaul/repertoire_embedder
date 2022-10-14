import umap, hdbscan
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import torch
from sklearn.metrics import normalized_mutual_info_score
import numpy as np, pandas as pd
import utils as u, models
from tqdm import tqdm
import argparse
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument("specie", type=str)
parser.add_argument("-bottleneck", type=int, default=16)
parser.add_argument("-lr", type=float, default=3e-3)
parser.add_argument("-lr_decay", type=float, default=1e-2)
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-cuda", type=int, default=0)
args = parser.parse_args()

df = pd.read_csv(f'{args.specie}/{args.specie}.csv')
print(f'{len(df)} available vocs')

nepoch = 100
modelname = f'{args.specie}_{args.bottleneck}.stdc'
gpu = torch.device(f'cuda:{args.cuda}')
writer = SummaryWriter('runs/'+modelname)
vgg16 = models.vgg16
vgg16.eval().to(gpu)
meta = models.meta[args.specie]

frontend = models.frontend(meta['sr'], meta['nfft'], meta['sampleDur'])
encoder = models.sparrow_encoder(args.bottleneck)
decoder = models.sparrow_decoder(args.bottleneck, (4, 2))
model = torch.nn.Sequential(frontend, encoder, decoder).to(gpu)

print('Go for model '+modelname)

optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0, lr=args.lr, betas=(0.8, 0.999))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (1-args.lr_decay)**epoch)
loader = torch.utils.data.DataLoader(u.Dataset(df, f'{args.specie}/audio/', meta['sr'], meta['sampleDur']), \
                               batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=u.collate_fn)
MSE = torch.nn.MSELoss()

step = 0
for epoch in range(nepoch):
    model.train()
    for x, name in tqdm(loader, desc=str(epoch), leave=False):
        optimizer.zero_grad()
        label = frontend(x.to(gpu))
        assert not torch.isnan(label).any(), "NaN in spectrogram :'("
        x = encoder(label)
        pred = decoder(x)
        assert not torch.isnan(pred).any(), "found a NaN :'("

        predd = vgg16(pred.expand(pred.shape[0], 3, *pred.shape[2:]))
        labell = vgg16(label.expand(label.shape[0], 3, *label.shape[2:]))

        score = MSE(predd, labell)
        score.backward()
        optimizer.step()
        writer.add_scalar('loss', score.item(), step)
        step += 1

    images = [(e-e.min())/(e.max()-e.min()) for e in label[:8]]
    grid = make_grid(images)
    writer.add_image('target', grid, epoch)
    writer.add_embedding(x.detach(), global_step=step, label_img=label)
    images = [(e-e.min())/(e.max()-e.min()) for e in pred[:8]]
    grid = make_grid(images)
    writer.add_image('reconstruct', grid, epoch)

    torch.save(model.state_dict(), modelname)

    scheduler.step()
    model.eval()
    with torch.no_grad():
        encodings, idxs = [], []
        for x, idx in tqdm(loader, desc='test '+str(epoch), leave=False):
            encoding = model[:2](x.to(gpu))
            idxs.extend(idx)
            encodings.extend(encoding.cpu().detach())
    idxs = np.array(idxs)
    encodings = np.stack(encodings)
    X = umap.UMAP(n_jobs=-1).fit_transform(encodings)
    clusters = hdbscan.HDBSCAN(min_cluster_size=len(df)//100, min_samples=5, core_dist_n_jobs=-1, cluster_selection_method='leaf').fit_predict(X)
    writer.add_scalar('NMI', normalized_mutual_info_score(df.loc[idxs].label, clusters), epoch)




