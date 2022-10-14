import torchvision.models as torchmodels
from torch import nn
import utils as u
from filterbank import STFT, MelFilter, Log1p

meta = {
  'bengalese_finch1':{
    'sr': 32000,
    'nfft': 512,
    'sampleDur': 0.1
  },
  'bengalese_finch2':{
    'sr': 32000,
    'nfft': 512,
    'sampleDur': 0.1
  },
  'black-headed_grosbeaks':{
    'sr':44100,
    'nfft':512,
    'sampleDur':0.35
  },
  'california_thrashers':{
    'nfft':512,
    'sr': 44100,
    'sampleDur': 0.25
  },
  'cassin_vireo':{
    'sr':44100,
    'nfft':512,
    'sampleDur': 0.5
  },
  'orcas':{
    'nfft': 1024,
    'sr': 22050,
    'sampleDur': 2
  },
  'humpback':{
    'nfft': 512,
    'sr': 11025,
    'sampleDur': 2
  }
}

vgg16 = torchmodels.vgg16(pretrained=True)
vgg16 = vgg16.features[:13]
for nm, mod in vgg16.named_modules():
    if isinstance(mod, nn.MaxPool2d):
        setattr(vgg16, nm,  nn.AvgPool2d(2 ,2))


N_MEL_BANDS= 64

frontend = lambda sr, nfft, sampleDur : nn.Sequential(
  STFT(nfft, int((sampleDur*sr - nfft)/128)),
  MelFilter(sr, nfft, N_MEL_BANDS, 0, sr//2),
  Log1p(7, trainable=False)
)

sparrow_encoder = lambda nfeat : nn.Sequential(
  nn.Conv2d(1, 32, 3, stride=2, bias=False, padding=(1)),
  nn.BatchNorm2d(32),
  nn.LeakyReLU(0.01),
  nn.Conv2d(32, 64, 3, stride=2, bias=False, padding=1),
  nn.BatchNorm2d(64),
  nn.MaxPool2d((1, 2)),
  nn.ReLU(True),
  nn.Conv2d(64, 128, 3, stride=2, bias=False, padding=1),
  nn.BatchNorm2d(128),
  nn.ReLU(True),
  nn.Conv2d(128, 256, 3, stride=2, bias=False, padding=1),
  nn.BatchNorm2d(256),
  nn.ReLU(True),
  nn.Conv2d(256, nfeat, (3, 5), stride=2, padding=(1, 2)),
  nn.AdaptiveMaxPool2d((1,1)),
  u.Reshape(nfeat)
)

sparrow_decoder = lambda nfeat, shape : nn.Sequential(
  nn.Linear(nfeat, nfeat*shape[0]*shape[1]),
  u.Reshape(nfeat, *shape),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(nfeat, 256, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(256),
  nn.ReLU(True),
  nn.Conv2d(256, 256, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(256),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(256, 128, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(128),
  nn.ReLU(True),
  nn.Conv2d(128, 128, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(128),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(128, 64, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(64),
  nn.ReLU(True),
  nn.Conv2d(64, 64, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(64),
  nn.ReLU(True),

  nn.Upsample(scale_factor=(1,2)),
  nn.Conv2d(64, 32, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(32),
  nn.ReLU(True),
  nn.Conv2d(32, 32, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(32),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2 if N_MEL_BANDS == 128 else (1,2)),
  nn.Conv2d(32, 32, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(32),
  nn.ReLU(True),
  nn.Conv2d(32, 32, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(32),
  nn.ReLU(True),

  nn.Upsample(scale_factor=2),
  nn.Conv2d(32, 32, (3, 3), bias=False, padding=1),
  nn.BatchNorm2d(32),
  nn.ReLU(True),
  nn.Conv2d(32, 1, (3, 3), bias=False, padding=1),
  nn.ReLU(True)
)


