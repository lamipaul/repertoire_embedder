import p_tqdm
from soundsig.sound import BioSound
import soundfile as sf
from scipy.signal import resample
import argparse
import pandas as pd, numpy as np
import models

parser = argparse.ArgumentParser()
parser.add_argument("specie", type=str)
args = parser.parse_args()

df = pd.read_csv(f'{args.specie}/{args.specie}.csv')

norm = lambda arr: (arr - np.mean(arr) ) / np.std(arr)

meta = models.meta[args.specie]

feats = ['fund', 'cvfund', 'maxfund', 'minfund', 'meansal', 'meanspect', 'stdspect', 'skewspect',\
     'kurtosisspect', 'entropyspect', 'q1', 'q2', 'q3', 'meantime', 'stdtime', 'skewtime', 'kurtosistime', 'entropytime']

def process(idx):
    row = df.loc[idx]
    info = sf.info(f'{args.specie}/audio/{row.fn}')
    dur, fs = info.duration, info.samplerate
    start = int(np.clip(row.pos - meta['sampleDur']/2, 0, max(0, dur - meta['sampleDur'])) * fs)
    sig, fs = sf.read(f'{args.specie}/audio/{row.fn}', start=start, stop=start + int(meta['sampleDur']*fs))
    if sig.ndim == 2:
        sig = sig[:,0]
    if len(sig) < meta['sampleDur'] * fs:
        sig = np.concatenate([sig, np.zeros(int(self.sampleDur * fs) - len(sig))])
    if fs != meta['sr']:
        sig = resample(sig, int(len(sig)/fs*meta['sr']))
    sound = BioSound(soundWave=norm(sig), fs=fs)
    sound.spectroCalc(max_freq=meta['sr']//2, spec_sample_rate=128//meta['sampleDur'])
    sound.rms = sound.sound.std() 
    sound.ampenv(cutoff_freq = 20, amp_sample_rate = 1000)
    sound.spectrum(f_high=meta['sr']//2 - 1)
    sound.fundest(maxFund = 6000, minFund = 200, lowFc = 200, highFc = 6000, 
                           minSaliency = 0.5, debugFig = 0, 
                           minFormantFreq = 500, maxFormantBW = 500, windowFormant = 0.1,
                           method='Stack')
    
    return [sound.__dict__[f] for f in feats]

res = p_tqdm.p_map(process, df.index[:100], num_cpus=16)

for i, mr in zip(df.index[:100], res):
    for f, r in zip(feats, mr):
        df.loc[i, f] = r

df.to_csv(f'{args.specie}/{args.specie}_biosound.csv', index=False)
