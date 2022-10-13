from torchvision.utils import make_grid
import numpy as np
from torch import nn, utils, device, optim, long, save
import pandas as pd
import utils as u
from tqdm import tqdm
import models
import argparse
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("specie", type=str)
parser.add_argument("-bottleneck", type=int, default=16)
parser.add_argument("-lr", type=float, default=3e-3)
parser.add_argument("-lr_decay", type=float, default=1e-2)
args = parser.parse_args()

df = pd.read_csv(f'{args.specie}/{args.specie}.csv')
print(f'{len(df)} available vocs')

nepoch = 100
batch_size = 64
modelname = f'{args.specie}_{args.bottleneck}'
gpu = device('cuda:1')
writer = SummaryWriter('runs/'+modelname)
vgg16 = u.VGG()
vgg16.eval().to(gpu)
meta = models.meta[args.specie]

frontend = models.frontend(meta['sr'], meta['nfft'], meta['sampleDur'])
encoder = models.sparrow_encoder(args.nfeat)
decoder = models.sparrow_decoder(args.nfeat, (4, 2))
model = nn.Sequential(frontend, encoder, decoder).to(gpu).train()

print('Go for model '+modelname)

optimizer = optim.AdamW(model.parameters(), weight_decay=0, lr=args.lr, betas=(0.8, 0.999))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (1-args.lr_decay)**epoch)
loader = utils.data.DataLoader(u.Dataset(df, f'{args.specie}/audio/', meta['sr'], meta['sampleDur']), \
                               batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=u.collate_fn)
loss_fun = nn.MSELoss()

step = 0
for epoch in range(nepoch):
    for x, name in tqdm(loader, desc=str(epoch), leave=False):
        optimizer.zero_grad()
        label = frontend(x.to(gpu))
        x = encoder(label)
        pred = decoder(x)
        predd = vgg16(pred.expand(pred.shape[0], 3, *pred.shape[2:]))
        labell = vgg16(label.expand(label.shape[0], 3, *label.shape[2:]))

        score = loss_fun(predd, labell)
        score.backward()
        optimizer.step()
        writer.add_scalar('loss', score.item(), step)


        if step%50==0 :
            images = [(e-e.min())/(e.max()-e.min()) for e in label[:8]]
            grid = make_grid(images)
            writer.add_image('target', grid, step)
            writer.add_embedding(x.detach(), global_step=step, label_img=label)
            images = [(e-e.min())/(e.max()-e.min()) for e in pred[:8]]
            grid = make_grid(images)
            writer.add_image('reconstruct', grid, step)

        step += 1
    if epoch % 10 == 0:
        scheduler.step()
    save(model.state_dict(), modelname)

