from parameters import set_params
from utility import DTWAligner

import numpy as np

import librosa
import pyworld
import pysptk

from nnmnkwii.metrics import melcd

from PIL import Image

#パラメーターの設定
hparams = set_params()
# from https://github.com/r9y9/nnmnkwii/blob/8afc05cce5b8a6727ed5d0fb874c1ae4e4039f1e/tests/test_real_datasets.py#L113
fs = hparams.sample_rate
fftlen = pyworld.get_cheaptrick_fft_size(fs)
alpha = pysptk.util.mcepalpha(fs)
order = 25 # メルケプストラムの要素の数?=num_mels
frame_period = hparams.frame_shift_ms # 1フレームあたり何msにしたいか
idx = 0

#関数の設定
def collect_features(x, fs):
    x = x.astype(np.float64)
    f0, timeaxis = pyworld.dio(x, fs, frame_period=frame_period)
    f0 = pyworld.stonemask(x, f0, timeaxis, fs)
    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
    mc = pysptk.sp2mc(spectrogram, order=order, alpha=alpha)
    return mc

def load_wav(path):
    w = librosa.core.load(path, sr=hparams.sample_rate)[0]
    w = librosa.effects.remix(w, intervals=librosa.effects.split(w))
    w = librosa.effects.trim(w, top_db=20)[0]
    return w

def _stft(y):
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def _stft_parameters():
    n_fft = (hparams.num_freq - 1) * 2
    hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
    return n_fft, hop_length, win_length

def _istft(y):
    _, hop_length, win_length = _stft_parameters()
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)

def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x))) * 4 - 3

def to_png(x):
    x = x * 128+128
    x = np.clip(x, 0.0, 255.0)
    x = x.astype('uint8')
    return x

#waveの読み込み
base1 = "../input/miki/2_Honey_Heartbeat_miki.wav"
base2 = "../input/ritsuko/2_Honey_Heartbeat_ritsuko.wav"
w1 = load_wav(base1)
w2 = load_wav(base2)

#スペクトログラムの計算
s1 = _stft(w1)
s2 = _stft(w2)
m = max(s1.shape[-1], s2.shape[-1])
s1 = [np.pad(s1, ((0, 0), (0, m - s1.shape[-1])), mode='edge')]
s2 = [np.pad(s2, ((0, 0), (0, m - s2.shape[-1])), mode='edge')]

#featureの計算
f1 = collect_features(w1, fs).T
f2 = collect_features(w2, fs).T
m = max(f1.shape[-1], f2.shape[-1])
f1 = [np.pad(f1, ((0, 0), (0, m - f1.shape[-1])), mode='edge')]
f2 = [np.pad(f2, ((0, 0), (0, m - f2.shape[-1])), mode='edge')]

#fを使ってalignment
X, Y = f1[idx].T[None], f2[idx].T[None]
s1_aligned, s2_aligned = DTWAligner(verbose=0, dist=melcd).transform((X, Y), (s1[idx].T[None], s2[idx].T[None]))

#alignmentしたデータをpngで保存
sp = np.abs(s1_aligned[0])
sp = sigmoid(sp)
sp = to_png(sp)
Image.fromarray(sp.T[::-1]).save('./output.png')