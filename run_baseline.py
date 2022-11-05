from tqdm import tqdm
from soundsig.sound import Biosound
import soundfile as sf
import argparse
import pandas as pd
import models

parser = argparse.ArgumentParser()
parser.add_argument("specie", type=str)
args = parser.parse_args()

df = pd.read_csv(f'{args.specie}/{args.specie}.csv')

def norm(arr):
    return (arr - np.mean(arr) ) / np.std(arr)
meta = models.meta[args.specie]


for idx, row in tqdm(df.iterrows(), total=len(df)):
    info = sf.info(self.audiopath+row.fn)
    dur, fs = info.duration, info.samplerate
    start = int(np.clip(row.pos - meta['sampleDur']/2, 0, max(0, dur - meta['sampleDur'])) * fs)
    sig, fs = sf.read(self.audiopath+row.fn, start=start, stop=start + int(meta['sampleDur']*fs))
    if sig.ndim == 2:
        sig = sig[:,0]
    if len(sig) < meta['sampleDur'] * fs:
        sig = np.concatenate([sig, np.zeros(int(self.sampleDur * fs) - len(sig))])
    sound = BioSound(soundWave=norm(sig), fs=fs)
    sound.spectroCalc(max_freq=meta['sr']//2)
    sound.rms = myBioSound.sound.std() 
    sound.ampenv(cutoff_freq = 20, amp_sample_rate = 1000)
    sound.spectrum(f_high=10000)
    sound.fundest(maxFund = 1500, minFund = 300, lowFc = 200, highFc = 6000, 
                           minSaliency = 0.5, debugFig = 0, 
                           minFormantFreq = 500, maxFormantBW = 500, windowFormant = 0.1,
                           method='Stack')