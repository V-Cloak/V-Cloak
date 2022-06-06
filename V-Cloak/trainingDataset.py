
from torch.utils.data.dataset import Dataset
import torch
import numpy as np
import torchaudio


class trainingDataset(Dataset):
    def __init__(
        self, 
        wavs_list, 
        seed=1234, 
        normalize_mean=0.55,
        normalize_sigma=0.05,
        N=41641,
        M=32089,
        masker=None
    ):
        self.wavs_list = wavs_list
        self.length = len(wavs_list)

        self.mean = normalize_mean
        self.sigma = normalize_sigma
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.N = N
        self.M = M
        self.pad = (N-M)//2

        print("Size of dataset: {}".format(self.length))

        self.msker = masker

    def __getitem__(self, index):
        wav_filename = self.wavs_list[index]
        wav1 = self._preprocess(wav_filename)
        theta, original_max_psd = self.msker._compute_masking_threshold(wav1[0, self.pad:-self.pad].numpy())
        theta = torch.FloatTensor(theta.transpose(1, 0))
        original_max_psd = torch.FloatTensor([original_max_psd])

        return wav1, theta, original_max_psd

    def __len__(self):
        return self.length

    def _preprocess(self, wav_filename):
        waveform, sample_rate = torchaudio.load(wav_filename)

        n_total = waveform.size(-1)
        assert n_total >= self.N
        start_idx = np.random.randint(n_total - self.N + 1)
        end_idx = start_idx + self.N
        waveform = waveform[:, start_idx: end_idx]
        waveform = waveform/waveform.abs().max()
        waveform *= torch.clamp(self.mean+self.sigma*torch.randn(1), 
                                min=0.1, max=1)

        return waveform

    def __len__(self):
        return self.length


if __name__ == '__main__':
    import glob
    dataset = trainingDataset(glob.glob("./test.wav"))
    trainLoader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=2,
                                              shuffle=True)
    for i, mfb in enumerate(trainLoader):
        print(mfb.shape)
        print(mfb)
        if i == 3:
            break
