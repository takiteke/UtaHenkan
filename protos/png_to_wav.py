import librosa
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

import chainer
from chainer import Variable
from net import Encoder
from net import Decoder

ITER = 155000

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
chainer.serializers.load_npz("../output_unrevarb_lam1=10/enc_iter_%d.npz"%ITER, enc)
chainer.serializers.load_npz("../output_unrevarb_lam1=10/dec_iter_%d.npz"%ITER, dec)


img = Image.open('../input/png/37_ritsuko.png')
img = np.asarray(img).astype("f")/128.0-1.0
h, w, c = img.shape
print(img.shape)
img = img.transpose(2, 0, 1)

img = img[:2,:,:]
img = img.reshape(1, 2, h, w)

ritsuko = np.zeros(img.shape, dtype=np.float32)
for i in tqdm(range(30)):
    y_l = 0
    y_r = 512
    x_l = i * 128
    x_r = x_l + 128
    ritsuko[:,:,y_l:y_r, x_l:x_r] = dec(enc(img[:,:,y_l:y_r, x_l:x_r])).data

def to_png(x):
    x = x * 128+128
    x = np.clip(x, 0.0, 255.0)
    x = x.astype('uint8')
    return x

print(ritsuko.shape)
converted_img = np.zeros((3, ritsuko[0][0].shape[0], ritsuko[0][0].shape[1]), dtype=np.float32)
converted_img[0] = ritsuko[0][0]
converted_img[1] = ritsuko[0][1]
converted_img = converted_img.transpose(1, 2, 0)
converted_img = to_png(converted_img)
Image.fromarray(converted_img).convert('RGB').save('../converted_iter%d_unrevarb_lam1=10.png'%ITER)


rrr = inv_sigmoid(ritsuko[0])
#rrr = np.clip(rrr, -20, 5.54)

rrr_ = np.zeros(rrr[0].shape, dtype=np.float32)
rrr_ = rrr[0] + rrr[1] * 1j

rrr_aligned = _istft(rrr_[::-1])

save_wav(rrr_aligned, '../converted_iter%d_unrevarb_lam1=10.wav'%ITER)