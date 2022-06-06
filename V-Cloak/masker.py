import logging
from typing import TYPE_CHECKING, Optional, Tuple, List

import numpy as np
import scipy
if TYPE_CHECKING:
    import torch


class Masker(object):
    def __init__(
        self,
        device,
        win_length: int = 2048,
        hop_length: int = 512,
        n_fft: int = 2048,
        sample_rate: int = 16000,
    ):

        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.device = device

    def _compute_masking_threshold(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the masking threshold and the maximum psd of the original audio.

        :param x: Samples of shape (seq_length,).
        :return: A tuple of the masking threshold and the maximum psd.
        """
        import librosa

        # First compute the psd matrix
        # Get window for the transformation
        window = scipy.signal.get_window("hann", self.win_length, fftbins=True)

        # Do transformation
        transformed_x = librosa.core.stft(
            y=x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=window, center=False
        )
        transformed_x *= np.sqrt(8.0 / 3.0)

        psd = abs(transformed_x / self.win_length)
        original_max_psd = np.max(psd * psd)
        with np.errstate(divide="ignore"):
            psd = (20 * np.log10(psd)).clip(min=-200)
        psd = 96 - np.max(psd) + psd

        # Compute freqs and barks
        freqs = librosa.core.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        barks = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan(pow(freqs / 7500.0, 2))

        # Compute quiet threshold
        ath = np.zeros(len(barks), dtype=np.float64) - np.inf
        bark_idx = int(np.argmax(barks > 1))
        ath[bark_idx:] = (
            3.64 * pow(freqs[bark_idx:] * 0.001, -0.8)
            - 6.5 * np.exp(-0.6 * pow(0.001 * freqs[bark_idx:] - 3.3, 2))
            + 0.001 * pow(0.001 * freqs[bark_idx:], 4)
            - 12
        )

        # Compute the global masking threshold theta
        theta = []

        for i in range(psd.shape[1]):
            # Compute masker index
            masker_idx = scipy.signal.argrelextrema(psd[:, i], np.greater)[0]

            if 0 in masker_idx:
                masker_idx = np.delete(masker_idx, 0)

            if len(psd[:, i]) - 1 in masker_idx:
                masker_idx = np.delete(masker_idx, len(psd[:, i]) - 1)

            barks_psd = np.zeros([len(masker_idx), 3], dtype=np.float64)
            barks_psd[:, 0] = barks[masker_idx]
            barks_psd[:, 1] = 10 * np.log10(
                pow(10, psd[:, i][masker_idx - 1] / 10.0)
                + pow(10, psd[:, i][masker_idx] / 10.0)
                + pow(10, psd[:, i][masker_idx + 1] / 10.0)
            )
            barks_psd[:, 2] = masker_idx

            for j in range(len(masker_idx)):
                if barks_psd.shape[0] <= j + 1:
                    break

                while barks_psd[j + 1, 0] - barks_psd[j, 0] < 0.5:
                    quiet_threshold = (
                        3.64 * pow(freqs[int(barks_psd[j, 2])] * 0.001, -0.8)
                        - 6.5 * np.exp(-0.6 * pow(0.001 * freqs[int(barks_psd[j, 2])] - 3.3, 2))
                        + 0.001 * pow(0.001 * freqs[int(barks_psd[j, 2])], 4)
                        - 12
                    )
                    if barks_psd[j, 1] < quiet_threshold:
                        barks_psd = np.delete(barks_psd, j, axis=0)

                    if barks_psd.shape[0] == j + 1:
                        break

                    if barks_psd[j, 1] < barks_psd[j + 1, 1]:
                        barks_psd = np.delete(barks_psd, j, axis=0)
                    else:
                        barks_psd = np.delete(barks_psd, j + 1, axis=0)

                    if barks_psd.shape[0] == j + 1:
                        break

            # Compute the global masking threshold
            delta = 1 * (-6.025 - 0.275 * barks_psd[:, 0])

            t_s = []

            for m in range(barks_psd.shape[0]):
                d_z = barks - barks_psd[m, 0]
                zero_idx = int(np.argmax(d_z > 0))
                s_f = np.zeros(len(d_z), dtype=np.float64)
                s_f[:zero_idx] = 27 * d_z[:zero_idx]
                s_f[zero_idx:] = (-27 + 0.37 * max(barks_psd[m, 1] - 40, 0)) * d_z[zero_idx:]
                t_s.append(barks_psd[m, 1] + delta[m] + s_f)

            t_s_array = np.array(t_s)

            theta.append(np.sum(pow(10, t_s_array / 10.0), axis=0) + pow(10, ath / 10.0))

        theta_array = np.array(theta)

        return theta_array, original_max_psd
    
    def _psd_transform(self, delta: "torch.Tensor", original_max_psd: np.ndarray) -> "torch.Tensor":
        """
        Compute the psd matrix of the perturbation.

        :param delta: The perturbation.
        :param original_max_psd: The maximum psd of the original audio.
        :return: The psd matrix.
        """
        import torch  # lgtm [py/repeated-import]

        # Get window for the transformation
        window_fn = torch.hann_window  # type: ignore

        # Return STFT of delta
        delta_stft = torch.stft(
            delta,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False,
            window=window_fn(self.win_length).to(self.device),
            return_complex=True
        ).to(self.device)


        transformed_delta = torch.real(delta_stft)**2 + torch.imag(delta_stft)**2

        # Compute the psd matrix
        psd = ((8.0 / 3.0 / self.win_length) ** 2) * transformed_delta 
        psd = (
            torch.pow(torch.tensor(10.0).type(torch.float64), torch.tensor(9.6).type(torch.float64)).to(
                self.device
            )
            / torch.reshape(original_max_psd, [-1, 1, 1])
            * psd.type(torch.float64)
        )

        return psd

    def _forward_2nd_stage(
        self,
        local_delta_rescale: "torch.Tensor",
        theta_batch: List[np.ndarray],
        original_max_psd_batch: List[np.ndarray],
        real_lengths: np.ndarray,
    ) -> "torch.Tensor":
        """
        The forward pass of the second stage of the attack.

        :param local_delta_rescale: Local delta after rescaled.
        :param theta_batch: Original thresholds.
        :param original_max_psd_batch: Original maximum psd.
        :param real_lengths: Real lengths of original sequences.
        :return: The loss tensor of the second stage of the attack.
        """
        import torch  # lgtm [py/repeated-import]

        # Compute loss for masking threshold
        losses = []
        relu = torch.nn.ReLU()

        for i, _ in enumerate(theta_batch):
            psd_transform_delta = self._psd_transform(
                delta=local_delta_rescale[i, : real_lengths[i]], original_max_psd=original_max_psd_batch[i]
            )


            loss = torch.nanmean(relu(psd_transform_delta - torch.tensor(theta_batch[i]).to(self.device)))
            losses.append(loss)

        losses_stack = torch.stack(losses)

        return losses_stack

    def batch_forward_2nd_stage(
        self,
        local_delta_rescale: "torch.Tensor",
        theta_batch: "torch.Tensor",
        original_max_psd_batch: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        The forward pass of the second stage of the attack.

        :param local_delta_rescale: Local delta after rescaled.
        :param theta_batch: Original thresholds.
        :param original_max_psd_batch: Original maximum psd.
        :return: The loss tensor of the second stage of the attack.
        """
        import torch  # lgtm [py/repeated-import]

        # Compute loss for masking threshold
        relu = torch.nn.ReLU()

        psd_transform_delta = self._psd_transform(
            delta=local_delta_rescale, original_max_psd=original_max_psd_batch
        )

        # psd_transform_delta: [bs, 1025, 59]
        # theta_batch: [bs, 1025, 59]
        loss = torch.mean(relu(psd_transform_delta - theta_batch))

        return loss
