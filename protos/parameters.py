

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# from https://github.com/keithito/tacotron/blob/08989cc3553b3a916a31f565e4f20e34bf19172f/hparams.py
def set_params():
    hparams = AttrDict(
        # Audio:
        num_mels=80,
        num_freq=1025,
        sample_rate=24000,
        frame_length_ms=50, # 50/1000[s] * 28000[Hz] = 600個
        frame_shift_ms=12.5,
        preemphasis=0.97,
        min_level_db=-100,
        ref_level_db=20,
    )
    return hparams