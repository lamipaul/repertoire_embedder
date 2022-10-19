import soundfile as sf
import torch
import numpy as np
from scipy.signal import resample


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, audiopath, sr, sampleDur):
        super(Dataset, self)
        self.audiopath, self.df, self.sr, self.sampleDur = audiopath, df, sr, sampleDur

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            info = sf.info(self.audiopath+'/'+row.filename)
            dur, fs = info.duration, info.samplerate
            start = int(np.clip(row.pos - self.sampleDur/2, 0, max(0, dur - self.sampleDur)) * fs)
            sig, fs = sf.read(self.audiopath+'/'+row.filename, start=start, stop=start + int(self.sampleDur*fs), always_2d=True)
            sig = sig[:,0]
        except:
            print(f'failed to load sound from row {row.name} with filename {row.filename}')
            return None
        if len(sig) < self.sampleDur * fs:
            sig = np.pad(sig, int(self.sampleDur * fs - len(sig))//2+1, mode='reflect')[:int(self.sampleDur * fs)]
        if fs != self.sr:
            sig = resample(sig, int(len(sig)/fs*self.sr))
        return torch.Tensor(norm(sig)).float(), row.name


def norm(arr):
    return (arr - np.mean(arr) ) / np.std(arr)


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)
