from parameters import set_params
from utility import DTWAligner

import os
import numpy as np
from tqdm import tqdm

import librosa
import pyworld
import pysptk

from nnmnkwii.metrics import melcd

from PIL import Image

import chainer
from chainer import Variable
from net import Encoder
from net import Decoder

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
    w2 = load_wav(dataDIR+"wav/%02d_ritsuko.wav"%idx)

    #スペクトログラムの計算
    s2 = [_stft(w2)]

    #featureの計算
    f2 = [collect_features(w2, fs).T]

    #s1_aligned = s1[0].T[None]
    #s2_aligned = s2[0].T[None]

    #実数、虚数で2チャンネルの画像にする
    sp2 = np.zeros((3, s2[0].shape[0], s2[0].shape[1]), dtype=np.float32)
    sp2[0] = s2[0].real
    sp2[1] = s2[0].imag
    sp2 = to_png(sigmoid(sp2))
    print(sp2.shape)

    #alignmentしたデータをpngで保存
    sp2 = sp2.T[::-1]
    print(sp2.shape)
    sp2 = sp2.transpose(1, 0, 2)
    print(sp2.shape)
    #Image.fromarray(sp2.T[::-1]).convert('RGB').save(dataDIR+"png/%02d_ritsuko.png"%idx)


ITER = 400000
MODEL_DIR = "../output_ran/"
MUSIC_IDX = 37

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# from https://github.com/keithito/tacotron/blob/08989cc3553b3a916a31f565e4f20e34bf19172f/hparams.py
hparams = AttrDict(
    # Audio:
    num_mels=80,
    num_freq=513,
    sample_rate=24000,
    frame_length_ms=40, # 50/1000[s] * 28000[Hz] = 600個
    frame_shift_ms=10,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,
)
def _stft_parameters():
    n_fft = (hparams.num_freq - 1) * 2
    hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
    return n_fft, hop_length, win_length

def inv_sigmoid(x):#1.0->inf
    x = np.clip(x, -0.9999999, 0.9999999)
    x = -np.log(1/((x+1)/2)-1)
    return x

def _istft(y):
    _, hop_length, win_length = _stft_parameters()
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)

def save_wav(wav, path):
    wav *= 1 / max(0.01, np.max(np.abs(wav)))
    librosa.output.write_wav(path, wav, hparams.sample_rate)

# Set up a neural network
enc = Encoder(in_ch=2)
dec = Decoder(out_ch=2)

# load terained models
chainer.serializers.load_npz(MODEL_DIR + "enc_iter_%d.npz"%ITER, enc)
chainer.serializers.load_npz(MODEL_DIR + "dec_iter_%d.npz"%ITER, dec)


#img = Image.open('../input/png/%d_ritsuko.png'%MUSIC_IDX)
#img = np.asarray(img).astype("f")/128.0-1.0
img = sp2.astype("f")/128.0-1.0
h, w, c = img.shape
print(img.shape)
img = img.transpose(2, 0, 1)

img = img[:2,:,:]
img = img.reshape(1, 2, h, w)
print(img.shape)

ritsuko = np.zeros(img.shape, dtype=np.float32)
for i in tqdm(range(200)):
    y_l = 0
    y_r = 512
    x_l = i * 16
    x_r = x_l + 128
    ritsuko[:,:,y_l:y_r, x_l:x_l+16] = dec(enc(img[:,:,y_l:y_r, x_l:x_r])).data[:,:,:,112:128]


print(ritsuko.shape)
converted_img = np.zeros((3, ritsuko[0][0].shape[0], ritsuko[0][0].shape[1]), dtype=np.float32)
converted_img[0] = ritsuko[0][0]
converted_img[1] = ritsuko[0][1]
converted_img = converted_img.transpose(1, 2, 0)
converted_img = to_png(converted_img)
Image.fromarray(converted_img).convert('RGB').save(MODEL_DIR + 'cvrt_itr%d.png'%ITER)


rrr = inv_sigmoid(ritsuko[0])
#rrr = np.clip(rrr, -20, 5.54)

rrr_ = np.zeros(rrr[0].shape, dtype=np.float32)
rrr_ = rrr[0] + rrr[1] * 1j

rrr_aligned = _istft(rrr_[::-1])

save_wav(rrr_aligned, MODEL_DIR + 'cvrt_itr%d.wav'%ITER)