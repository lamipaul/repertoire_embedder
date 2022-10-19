from torchvision.utils import make_grid
import torch
import pandas as pd
import utils as u, models
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("detections", type=str, help=".csv file with detections to be encoded. Columns filename (path of the soundfile) and pos (center of the detection in seconds) are needed")
parser.add_argument("-audio_folder", type=str, default='./', help="Folder from which to load sound files")
parser.add_argument("-NFFT", type=int, default=1024, help="FFT size for the spectrogram computation")
parser.add_argument("-nMel", type=int, default=128, help="Number of Mel bands for the spectrogram (either 64 or 128)")
parser.add_argument("-SR", type=int, default=44100, help="Sample rate of the samples before spectrogram computation")
parser.add_argument("-sampleDur", type=float, default=1, help="Size of the signal extracts surrounding detections to be encoded")
parser.add_argument("-bottleneck", type=int, default=16, help='size of the auto-encoder\'s bottleneck')
args = parser.parse_args()

df = pd.read_csv(args.detections)
print(f'Training using {len(df)} vocalizations')

nepoch = 100
batch_size = 64 if torch.cuda.is_available() else 16
nfeat = 16
modelname = args.detections[:-4]+'_AE_weights.stdc'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.003
wdL2 = 0.0
writer = SummaryWriter('runs/'+modelname)
vgg16 = models.vgg16.eval().to(device)

frontend = models.frontend(args.SR, args.NFFT, args.sampleDur, args.nMel)
encoder = models.sparrow_encoder(args.bottleneck)
decoder = models.sparrow_decoder(args.bottleneck, (4, 4) if args.nMel == 128 else (2, 4))
model = torch.nn.Sequential(frontend, encoder, decoder).to(device)


optimizer = torch.optim.AdamW(model.parameters(), weight_decay=wdL2, lr=lr, betas=(0.8, 0.999))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : .99**epoch)
loader = torch.utils.data.DataLoader(u.Dataset(df, args.audio_folder, args.SR, args.sampleDur), batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=u.collate_fn)
loss_fun = torch.nn.MSELoss()

print('Go for model '+modelname)
step = 0
for epoch in range(nepoch):
    for x, name in tqdm(loader, desc=str(epoch), leave=False):
        optimizer.zero_grad()
        label = frontend(x.to(device))
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
    torch.save(model.state_dict(), modelname)

