from parameters import set_params
from utility import DTWAligner

import numpy as np
from tqdm import tqdm

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
    #w = librosa.effects.trim(w, top_db=20)[0]
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
    return (1.0 / (1.0 + np.exp(-x))) * 2 - 1

def to_png(x):
    x = x * 128+128
    x = np.clip(x, 0.0, 255.0)
    x = x.astype('uint8')
    return x


dataDIR = "../input/"
data_range = (41, 42)
print("load dataset start")
print("    from: %s"%dataDIR)
print("    range: [%d, %d)"%(data_range[0], data_range[1]))
for idx in tqdm(range(data_range[0], data_range[1])):
    #waveの読み込み
    w1 = load_wav(dataDIR+"wav/%02d_miki.wav"%idx)
    w2 = load_wav(dataDIR+"wav/%02d_ritsuko.wav"%idx)

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
    X, Y = f1[0].T[None], f2[0].T[None]
    s1_aligned, s2_aligned = DTWAligner(verbose=0, dist=melcd).transform((X, Y), (s1[0].T[None], s2[0].T[None]))

    #s1_aligned = s1[0].T[None]
    #s2_aligned = s2[0].T[None]

    #実数、虚数で2チャンネルの画像にする
    sp1 = np.zeros((3, s1_aligned[0].shape[0], s1_aligned[0].shape[1]), dtype=np.float32)
    sp1[0] = s1_aligned[0].real
    sp1[1] = s1_aligned[0].imag
    sp1 = to_png(sigmoid(sp1))
    sp2 = np.zeros((3, s2_aligned[0].shape[0], s2_aligned[0].shape[1]), dtype=np.float32)
    sp2[0] = s2_aligned[0].real
    sp2[1] = s2_aligned[0].imag
    sp2 = to_png(sigmoid(sp2))

    #alignmentしたデータをpngで保存
    Image.fromarray(sp1.T[::-1]).convert('RGB').save(dataDIR+"png/%02d_miki.png"%idx)
    Image.fromarray(sp2.T[::-1]).convert('RGB').save(dataDIR+"png/%02d_ritsuko.png"%idx)