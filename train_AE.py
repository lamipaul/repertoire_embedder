import umap, hdbscan
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import torch
from sklearn import metrics, cluster
import numpy as np, pandas as pd
from scipy.stats import linregress
import utils as u, models
from tqdm import tqdm
import os, argparse, warnings
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("error")

parser = argparse.ArgumentParser()
parser.add_argument("specie", type=str)
parser.add_argument("-bottleneck", type=int, default=16)
parser.add_argument("-frontend", type=str, default='logMel')
parser.add_argument("-encoder", type=str, default='sparrow_encoder')
parser.add_argument("-nMel", type=int, default=128)
parser.add_argument("-lr", type=float, default=3e-3)
parser.add_argument("-lr_decay", type=float, default=1e-2)
parser.add_argument("-batch_size", type=int, default=128)
parser.add_argument("-cuda", type=int, default=0)
args = parser.parse_args()

df = pd.read_csv(f'{args.specie}/{args.specie}.csv')
print(f'{len(df)} available vocs')

modelname = f'{args.specie}_{args.bottleneck}_{args.frontend}{args.nMel if "Mel" in args.frontend else ""}_{args.encoder}_decod2_BN_nomaxPool.stdc'
gpu = torch.device(f'cuda:{args.cuda}')
writer = SummaryWriter(f'runs2/{modelname}')
os.system(f'cp *.py runs2/{modelname}')
vgg16 = models.vgg16
vgg16.eval().to(gpu)
meta = models.meta[args.specie]

frontend = models.frontend[args.frontend](meta['sr'], meta['nfft'], meta['sampleDur'], args.nMel)
encoder = models.__dict__[args.encoder](*((args.bottleneck // 16, (4, 4)) if args.nMel == 128 else (args.bottleneck // 8, (2, 4))))
decoder = models.sparrow_decoder(args.bottleneck, (4, 4) if args.nMel == 128 else (2, 4))
model = torch.nn.Sequential(frontend, encoder, decoder).to(gpu)

print('Go for model '+modelname)

optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0, lr=args.lr, betas=(0.8, 0.999))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (1-args.lr_decay)**epoch)
loader = torch.utils.data.DataLoader(u.Dataset(df, f'{args.specie}/audio/', meta['sr'], meta['sampleDur']), \
                               batch_size=args.batch_size, shuffle=True, num_workers=8, prefetch_factor=8, collate_fn=u.collate_fn)
MSE = torch.nn.MSELoss()

step, loss = 0, []
for epoch in range(100_000//len(loader)):
    for x, name in tqdm(loader, desc=str(epoch), leave=False):
        optimizer.zero_grad()
        label = frontend(x.to(gpu))
        assert not torch.isnan(label).any(), "NaN in spectrogram :'( "+str(name[torch.isnan(label).any(1).any(1).any(1)])
        x = encoder(label)
        pred = decoder(x)
        assert not torch.isnan(pred).any(), "found a NaN :'("
        predd = vgg16(pred.expand(pred.shape[0], 3, *pred.shape[2:]))
        labell = vgg16(label.expand(label.shape[0], 3, *label.shape[2:]))

        score = MSE(predd, labell)
        score.backward()
        optimizer.step()
        writer.add_scalar('loss', score.item(), step)
        loss.append(score.item())

        if len(loss) > 2000 and np.median(loss[-2000:-1000]) < np.median(loss[-1000:]):
            print('Early stop')
            torch.save(model.state_dict(), f'{args.specie}/{modelname}')
            exit()
        step += 1
        continue
        # TEST ROUTINE
        if step % 500 == 0:
            # Plot reconstructions
            images = [(e-e.min())/(e.max()-e.min()) for e in label[:8]]
            grid = make_grid(images)
            writer.add_image('target', grid, step)
            # writer.add_embedding(x.detach(), global_step=step, label_img=label)
            images = [(e-e.min())/(e.max()-e.min()) for e in pred[:8]]
            grid = make_grid(images)
            writer.add_image('reconstruct', grid, step)

            torch.save(model.state_dict(), f'{args.specie}/{modelname}')
            scheduler.step()

            # Actual test
            model[1:].eval()
            with torch.no_grad():
                encodings, idxs = [], []
                for x, idx in tqdm(loader, desc='test '+str(step), leave=False):
                    encoding = model[:2](x.to(gpu))
                    idxs.extend(idx)
                    encodings.extend(encoding.cpu().detach())
            idxs, encodings = np.array(idxs), np.stack(encodings)
            print('Computing UMAP...', end='')
            try:
                X = umap.UMAP(n_jobs=-1).fit_transform(encodings)
            except UserWarning:
                pass
            print('\rRunning HDBSCAN...', end='')
            clusters = hdbscan.HDBSCAN(min_cluster_size=len(df)//100, min_samples=5, core_dist_n_jobs=-1, cluster_selection_method='leaf').fit_predict(X)
            # df.loc[idxs, 'cluster'] = clusters.astype(int)
            mask = ~df.loc[idxs].label.isna()
            clusters, labels = clusters[mask], df.loc[idxs[mask]].label
            writer.add_scalar('NMI HDBSCAN', metrics.normalized_mutual_info_score(labels, clusters), step)
            try:
                writer.add_scalar('ARI HDBSCAN', metrics.adjusted_rand_score(labels, clusters), step)
            except:
                pass
            writer.add_scalar('Homogeneity HDBSCAN', metrics.homogeneity_score(labels, clusters), step)
            writer.add_scalar('Completeness HDBSCAN', metrics.completeness_score(labels, clusters), step)
            writer.add_scalar('V-Measure HDBSCAN', metrics.v_measure_score(labels, clusters), step)
            
            # print('\rComputing HDBSCAN precision and recall distributions', end='')
            # labelled = df[~df.label.isna()]
            # precs, recs = [], []
            # for l, grp in labelled.groupby('label'):
            #     best = (grp.groupby('cluster').fn.count() / labelled.groupby('cluster').fn.count()).idxmax()
            #     precs.append((grp.cluster==best).sum()/(labelled.cluster==best).sum())
            #     recs.append((grp.cluster==best).sum()/len(grp))
            # writer.add_histogram('HDBSCAN Precisions ', np.array(precs), step)
            # writer.add_histogram('HDBSCAN Recalls ', np.array(recs), step)
            # df.drop('cluster', axis=1, inplace=True)
            # print('\rRunning elbow method for K-Means...', end='')
            # ks = (5*1.2**np.arange(20)).astype(int)
            # distorsions = [cluster.KMeans(n_clusters=k).fit(encodings).inertia_ for k in ks]
            # print('\rEstimating elbow...', end='')
            # errors = [linregress(ks[:i], distorsions[:i]).stderr + linregress(ks[i+1:], distorsions[i+1:]).stderr for i in range(2, len(ks)-2)]
            # k = ks[np.argmin(errors)]
            # writer.add_scalar('Chosen K', k, step)
            # clusters = cluster.KMeans(n_clusters=k).fit_predict(encodings)
            # df.loc[idxs, 'cluster'] = clusters.astype(int)

            # writer.add_scalar('Silhouette', metrics.silhouette_score(encodings, clusters), step)
            # clusters, labels = clusters[mask], df.loc[idxs[mask]].label
            # writer.add_scalar('NMI K-Means', metrics.normalized_mutual_info_score(labels, clusters), step)
            # try:
            #     writer.add_scalar('ARI K-Means', metrics.adjusted_rand_score(labels, clusters), step)
            # except:
            #     pass
            # writer.add_scalar('Homogeneity K-Means', metrics.homogeneity_score(labels, clusters), step)
            # writer.add_scalar('Completeness K-Means', metrics.completeness_score(labels, clusters), step)
            # writer.add_scalar('V-Measure K-Means', metrics.v_measure_score(labels, clusters), step)

            # print('\rComputing K-Means precision and recall distributions', end='') 
            # labelled = df[~df.label.isna()]
            # precs, recs = [], []
            # for l, grp in labelled.groupby('label'):
            #     best = (grp.groupby('cluster').fn.count() / labelled.groupby('cluster').fn.count()).idxmax()
            #     precs.append((grp.cluster==best).sum()/(labelled.cluster==best).sum())
            #     recs.append((grp.cluster==best).sum()/len(grp))
            # writer.add_histogram('K-Means Precisions ', np.array(precs), step)
            # writer.add_histogram('K-Means Recalls ', np.array(recs), step)
            # df.drop('cluster', axis=1, inplace=True)

            print('\r', end='')
            model[1:].train()
        step += 1
