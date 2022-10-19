import utils as u
import models
import numpy as np, pandas as pd, torch
import umap
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Compute the AE projection of vocalizations once it was trained.")
parser.add_argument('modelname', type=str, help='Filename of the AE weights (.stdc)')
parser.add_argument("detections", type=str, help=".csv file with detections to be encoded. Columns filename (path of the soundfile) and pos (center of the detection in seconds) are needed")
parser.add_argument("-audio_folder", type=str, default='./', help="Folder from which to load sound files")
parser.add_argument("-NFFT", type=int, default=1024, help="FFT size for the spectrogram computation")
parser.add_argument("-nMel", type=int, default=128, help="Number of Mel bands for the spectrogram (either 64 or 128)")
parser.add_argument("-bottleneck", type=int, default=16, help='size of the auto-encoder\'s bottleneck')
parser.add_argument("-SR", type=int, default=44100, help="Sample rate of the samples before spectrogram computation")
parser.add_argument("-sampleDur", type=float, default=1, help="Size of the signal extracts surrounding detections to be encoded")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
frontend = models.frontend(args.SR, args.NFFT, args.sampleDur, args.nMel)
encoder = models.sparrow_encoder(args.bottleneck)
decoder = models.sparrow_decoder(args.bottleneck, (4, 4) if args.nMel == 128 else (2, 4))
model = torch.nn.Sequential(frontend, encoder, decoder).to(device)
model.load_state_dict(torch.load(args.modelname, map_location=device))

df = pd.read_csv(args.detections)

print('Computing AE projections...')
loader = torch.utils.data.DataLoader(u.Dataset(df, args.audio_folder, args.SR, args.sampleDur), batch_size=16, shuffle=False, num_workers=8, prefetch_factor=8)
with torch.no_grad():
    encodings, idxs = [], []
    for x, idx in tqdm(loader):
        encoding = model[:2](x.to(device))
        idxs.extend(idx)
        encodings.extend(encoding.cpu().detach())
idxs = np.array(idxs)
encodings = np.stack(encodings)

print('Computing UMAP projections...')
X = umap.UMAP(n_jobs=-1).fit_transform(encodings)
np.save('encodings_'+args.modelname[:-4]+'npy', {'encodings':encodings, 'idx':idxs, 'umap':X})
