import soundfile as sf
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
        try:
            info = sf.info(self.audiopath+row.fn)
            dur, fs = info.duration, info.samplerate
            start = int(np.clip(row.pos - self.sampleDur/2, 0, dur - self.sampleDur) * fs)
            sig, fs = sf.read(self.audiopath+row.fn, start=start, stop=start + int(self.sampleDur*fs))
        except:
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
