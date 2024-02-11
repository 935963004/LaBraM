import h5py
import mne
import numpy as np


def preprocessing_cnt(cntFilePath, l_freq=0.1, h_freq=75.0, sfreq:int=200):
    # reading cnt
    raw = mne.io.read_raw_cnt(cntFilePath, preload=True, data_format='int32')
    raw.drop_channels(['M1', 'M2', 'VEO', 'HEO'])
    if 'ECG' in raw.ch_names:
        raw.drop_channels(['ECG'])

    # filtering
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq)
    raw = raw.notch_filter(50.0)
    # downsampling
    raw = raw.resample(sfreq, n_jobs=5)
    eegData = raw.get_data(units='uV')

    return eegData, raw.ch_names


def preprocessing_edf(edfFilePath, l_freq=0.1, h_freq=75.0, sfreq:int=200, drop_channels: list=None, standard_channels: list=None):
    # reading edf
    raw = mne.io.read_raw_edf(edfFilePath, preload=True)
    if drop_channels is not None:
        useless_chs = []
        for ch in drop_channels:
            if ch in raw.ch_names:
                useless_chs.append(ch)
        raw.drop_channels(useless_chs)

    if standard_channels is not None and len(standard_channels) == len(raw.ch_names):
        try:
            raw.reorder_channels(standard_channels)
        except:
            return None, ['a']

    # filtering
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq)
    raw = raw.notch_filter(50.0)
    # downsampling
    raw = raw.resample(sfreq, n_jobs=5)
    eegData = raw.get_data(units='uV')

    return eegData, raw.ch_names

 
def readh5(h5filePath):
    with h5py.File('matrix.h5', 'r', libver='latest', swmr=True) as f:
        dset = f['data']
        shape = dset.shape
        dtype = dset.dtype

        if dset.chunks:
            np_array = np.empty(shape, dtype=dtype)
            dset.read_direct(np_array)
        else: 
            np_array = dset[()]
    return np_array


if __name__ == '__main__':
    print(readh5('./').shape)
