from typing import Union, Tuple
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
        info = sf.info(self.audiopath+row.fn)
        dur, fs = info.duration, info.samplerate
        start = int(np.clip(row.pos - self.sampleDur/2, 0, max(0, dur - self.sampleDur)) * fs)
        sig, fs = sf.read(self.audiopath+row.fn, start=start, stop=start + int(self.sampleDur*fs))
        if sig.ndim == 2:
            sig = sig[:,0]
        if len(sig) < self.sampleDur * fs:
            sig = np.concatenate([sig, np.zeros(int(self.sampleDur * fs) - len(sig))])
        if fs != self.sr:
            sig = resample(sig, int(len(sig)/fs*self.sr))
        if np.std(sig) == 0:
            print('wrong sig '+str(row.name))
        if self.retType:
            return torch.Tensor(norm(sig)).float(), row.name, row.label
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
        return x.reshape(x.shape[0], *self.shape)

class Croper2D(nn.Module):
    def __init__(self, *shape):
        super(Croper2D, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x[:,:,:self.shape[0],:self.shape[1]]




class VQ(nn.Module):
    """
    Quantization layer from *Neural Discrete Representation Learning*
    Args:
        latent_dim (int): number of features along which to quantize
        num_tokens (int): number of tokens in the codebook
        dim (int): dimension along which to quantize
        return_indices (bool): whether to return the indices of the quantized
            code points
    """
    embedding: nn.Embedding
    dim: int
    commitment: float
    initialized: torch.Tensor
    return_indices: bool
    init_mode: str

    def __init__(self,
                 latent_dim: int,
                 num_tokens: int,
                 dim: int = 1,
                 commitment: float = 0.25,
                 init_mode: str = 'normal',
                 return_indices: bool = True,
                 max_age: int = 1000):
        super(VQ, self).__init__()
        self.embedding = nn.Embedding(num_tokens, latent_dim)
        nn.init.normal_(self.embedding.weight, 0, 1.1)
        self.dim = dim
        self.commitment = commitment
        self.register_buffer('initialized', torch.Tensor([0]))
        self.return_indices = return_indices
        assert init_mode in ['normal', 'first']
        self.init_mode = init_mode
        self.register_buffer('age', torch.empty(num_tokens).fill_(max_age))
        self.max_age = max_age

    def update_usage(self, indices):
        with torch.no_grad():
            self.age += 1
            if torch.distributed.is_initialized():
                n_gpu = torch.distributed.get_world_size()
                all_indices = [torch.empty_like(indices) for _ in range(n_gpu)]
                torch.distributed.all_gather(all_indices, indices)
                indices = torch.cat(all_indices)
            used = torch.unique(indices)
            self.age[used] = 0

    def resample_dead(self, x):
        with torch.no_grad():
            dead = torch.nonzero(self.age > self.max_age, as_tuple=True)[0]
            if len(dead) == 0:
                return

            print(f'{len(dead)} dead codes resampled')
            x_flat = x.view(-1, x.shape[-1])
            emb_weight = self.embedding.weight.data
            emb_weight[dead[:len(x_flat)]] = x_flat[torch.randperm(
                len(x_flat))[:len(dead)]].to(emb_weight.dtype)
            self.age[dead[:len(x_flat)]] = 0

            if torch.distributed.is_initialized():
                torch.distributed.broadcast(emb_weight, 0)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        Args:
            x (tensor): input tensor
        Returns:
            quantized tensor, or (quantized tensor, indices) if
            `self.return_indices`
        """
        dim = self.dim
        nb_codes = self.embedding.weight.shape[0]

        codebook = self.embedding.weight
        if (self.init_mode == 'first' and self.initialized.item() == 0 and
                self.training):
            n_proto = self.embedding.weight.shape[0]

            ch_first = x.transpose(dim, -1).contiguous().view(-1, x.shape[dim])
            n_samples = ch_first.shape[0]
            idx = torch.randint(0, n_samples, (n_proto,))[:nb_codes]
            self.embedding.weight.data.copy_(ch_first[idx])
            self.initialized[:] = 1

        needs_transpose = dim != -1 or dim != x.dim() - 1
        if needs_transpose:
            x = x.transpose(-1, dim).contiguous()

        if self.training:
            self.resample_dead(x)

        codes, indices = quantize(x, codebook, self.commitment, self.dim)

        if self.training:
            self.update_usage(indices)

        if needs_transpose:
            codes = codes.transpose(-1, dim)
            indices = indices.transpose(-1, dim)

        if self.return_indices:
            return codes, indices
        else:
            return codes


from torch.autograd import Function


class VectorQuantization(Function):

    @staticmethod
    def compute_indices(inputs_orig, codebook):
        bi = []
        SZ = 10000
        for i in range(0, inputs_orig.size(0), SZ):
            inputs = inputs_orig[i:i + SZ]
            # NxK
            distances_matrix = torch.cdist(inputs, codebook)
            # Nx1
            indic = torch.min(distances_matrix, dim=-1)[1].unsqueeze(1)
            bi.append(indic)
        return torch.cat(bi, dim=0)

    @staticmethod
    def flatten(x):
        code_dim = x.size(-1)
        return x.view(-1, code_dim)

    @staticmethod
    def restore_shapes(codes, indices, target_shape):
        idx_shape = list(target_shape)
        idx_shape[-1] = 1
        return codes.view(*target_shape), indices.view(*idx_shape)

    @staticmethod
    def forward(ctx, inputs, codebook, commitment=0.25, dim=1):
        inputs_flat = VectorQuantization.flatten(inputs)
        indices = VectorQuantization.compute_indices(inputs_flat, codebook)
        codes = codebook[indices.view(-1), :]
        codes, indices = VectorQuantization.restore_shapes(
            codes, indices, inputs.shape)

        ctx.save_for_backward(codes, inputs, torch.tensor([float(commitment)]),
                              codebook, indices)
        ctx.mark_non_differentiable(indices)
        return codes, indices

    @staticmethod
    def backward(ctx, straight_through, unused_indices):
        codes, inputs, beta, codebook, indices = ctx.saved_tensors

        # TODO: figure out proper vq loss reduction
        # vq_loss = F.mse_loss(inputs, codes).detach()

        # gradient of vq_loss
        diff = 2 * (inputs - codes) / inputs.numel()

        commitment = beta.item() * diff

        code_disp = VectorQuantization.flatten(-diff)
        indices = VectorQuantization.flatten(indices)
        code_disp = (torch.zeros_like(codebook).index_add_(
            0, indices.view(-1), code_disp))
        return straight_through + commitment, code_disp, None, None


quantize = VectorQuantization.apply
