import soundfile as sf
import torchvision.models as torchmodels
import torch
from torch import nn
from torch.utils import data
import numpy as np
from scipy.signal import resample


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class Dataset(data.Dataset):
    def __init__(self, df, audiopath, sr, sampleDur, retType=False):
        super(Dataset, self)
        self.audiopath, self.df, self.retType, self.sr, self.sampleDur = audiopath, df, retType, sr, sampleDur

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fs = sf.info(self.audiopath+row.fn).samplerate
        sig, fs = sf.read(self.audiopath+row.fn, start=int((row.pos - self.sampleDur/2)*fs), stop=int((row.pos + self.sampleDur/2)*fs))
        if len(sig) < self.sampleDur:
            print(f'failed with {row.name}')
            return None
        if fs != self.sr:
            sig = resample(sig, int(len(sig)/fs*self.sr))
        if self.retType:
            return torch.Tensor(norm(sig)).float(), row.name, row.label_id
        else:
            return torch.Tensor(norm(sig)).float(), row.name


def norm(arr):
    return (arr - np.mean(arr) ) / np.std(arr)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


def VGG():
    vgg16 = torchmodels.vgg16(pretrained=True)
    vgg16 = vgg16.features[:13]
    for nm, mod in vgg16.named_modules():
        if isinstance(mod, nn.MaxPool2d):
            setattr(vgg16, nm,  nn.AvgPool2d(2 ,2))
    return vgg16
