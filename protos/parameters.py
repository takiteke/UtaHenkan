

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# from https://github.com/keithito/tacotron/blob/08989cc3553b3a916a31f565e4f20e34bf19172f/hparams.py
def set_params():
    hparams = AttrDict(
        # Audio:
        num_mels=80,
        num_freq=513,
        sample_rate=24000,
        frame_length_ms=40,
        frame_shift_ms=10,
        preemphasis=0.97,
        min_level_db=-100,
        ref_level_db=20,
    )
    return hparams