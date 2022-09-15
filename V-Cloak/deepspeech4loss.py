# from art.estimators.speech_recognition import PyTorchDeepSpeech
from new_pytorch_deep_speech import PyTorchDeepSpeech
import numpy as np
from deepspeech_pytorch.model import DeepSpeech
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import torch
import torchaudio
class DeepSpeech4Loss(PyTorchDeepSpeech):
    def __init__(
        self,
        model: Optional["DeepSpeech"] = None,
        pretrained_model: Optional[str] = None,
        device_type: str = "gpu",
        *args,
        **kwargs
    ):
        super().__init__(
            model=model,
            pretrained_model=pretrained_model,
            device_type=device_type,
            *args,
            **kwargs
        )

        self.ppg_criterion = torch.nn.MSELoss()

    def compute_loss(self,
                     x: Union[np.ndarray, torch.Tensor],
                     y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: Loss gradients of the same shape as `x`.
        """
        # x_in = torch.empty(len(x), device=x.device)
        x_in = x

        # Put the model in the training mode, otherwise CUDA can't backpropagate through the model.
        # However, model uses batch norm layers which need to be frozen
        self.DP_model.train()
        self.set_batchnorm(train=False)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x_in, y, fit=False)

        # Transform data into the model input space
        inputs, targets, input_rates, target_sizes, _ = self._transform_model_input(
            x=x_preprocessed, y=y_preprocessed, compute_gradient=False
        )

        # Compute real input sizes
        input_sizes = input_rates.mul_(inputs.size()[-1]).int()

        # Call to DeepSpeech model for prediction
        outputs, output_sizes = self.DP_model(inputs.to(self._device), input_sizes.to(self._device))
        outputs = outputs.transpose(0, 1)

        if self._version == 2:
            outputs = outputs.float()
        else:
            outputs = outputs.log_softmax(-1)

        # Compute the loss
        loss = self.criterion(outputs, targets, output_sizes, target_sizes).to(self._device)
        # loss.backward()
        return loss

    def compute_ppg(self,
                     x: Union[np.ndarray, torch.Tensor], **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: Loss gradients of the same shape as `x`.
        """
        # x_in = torch.empty(len(x), device=x.device)
        x_in = x

        # Put the model in the training mode, otherwise CUDA can't backpropagate through the model.
        # However, model uses batch norm layers which need to be frozen
        self.DP_model.train()
        self.set_batchnorm(train=False)

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x_in, y=None, fit=False)

        # Transform data into the model input space
        inputs, _, input_rates, _, _ = self._transform_model_input(
            x=x_preprocessed, compute_gradient=False
        )

        # Compute real input sizes
        input_sizes = input_rates.mul_(inputs.size()[-1]).int()

        # Call to DeepSpeech model for prediction
        outputs, output_sizes = self.DP_model(inputs.to(self._device), input_sizes.to(self._device))
        # outputs = outputs.transpose(0, 1)

        # if self._version == 2:
        #     outputs = outputs.float()
        # else:
        #     outputs = outputs.log_softmax(-1)

        # Compute the loss
        # loss = self.criterion(outputs, targets, output_sizes, target_sizes).to(self._device)
        # loss.backward()
        return outputs

    def compute_ppgctc_loss(self,
                             x: Union[np.ndarray, torch.Tensor],
                             y: np.ndarray,
                             ppg, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: Loss gradients of the same shape as `x`.
        """
        # x_in = torch.empty(len(x), device=x.device)
        x_in = x

        # Put the model in the training mode, otherwise CUDA can't backpropagate through the model.
        # However, model uses batch norm layers which need to be frozen
        self.DP_model.train()
        self.set_batchnorm(train=False)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x_in, y, fit=False)

        # Transform data into the model input space
        inputs, targets, input_rates, target_sizes, _ = self._transform_model_input(
            x=x_preprocessed, y=y_preprocessed, compute_gradient=False
        )

        # Compute real input sizes
        input_sizes = input_rates.mul_(inputs.size()[-1]).int()

        # Call to DeepSpeech model for prediction
        outputs, output_sizes = self.DP_model(inputs.to(self._device), input_sizes.to(self._device))
        ppg_loss = self.ppg_criterion(outputs, ppg)
        outputs = outputs.transpose(0, 1)

        if self._version == 2:
            outputs = outputs.float()
        else:
            outputs = outputs.log_softmax(-1)

        # Compute the loss
        ctc_loss = self.criterion(outputs, targets, output_sizes, target_sizes).to(self._device)
        # loss.backward()
        return ppg_loss, ctc_loss

    def compute_ppgloss(self,
                             x: Union[np.ndarray, torch.Tensor],
                             ppg, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: Loss gradients of the same shape as `x`.
        """
        # x_in = torch.empty(len(x), device=x.device)
        outputs = self.compute_ppg(x)
        ppg_loss = self.ppg_criterion(outputs, ppg)

        if torch.isnan(outputs).any():
            print('Warning: output ppg has NaN values')
        if torch.isnan(ppg).any():
            print('Warning: groundtruth ppg has NaN values')

        return ppg_loss

if __name__=='__main__':
    wav,sr = torchaudio.load('./test.wav')
    # wav = wav*2**15
    wav = wav.to('cuda')
    wav.requires_grad=True
    print(wav.shape)
    y = np.asarray(["NO WORDS WERE SPOKEN NO LANGUAGE WAS UTTERED SAVE THAT OF WAILING AND \
        HISSING AND THAT SOMEHOW WAS INDISTINCT AS IF IT EXISTED IN FANCY AND NOT IN REALITY"])
    print(y)
    asr_model = DeepSpeech4Loss(pretrained_model="librispeech")
    loss = asr_model.compute_loss(wav,y)
    y_pred = asr_model.predict(wav.detach().cpu().numpy())

    loss.backward()
    # y_pred_scale = asr_model.predict(wav.cpu().numpy()*2**15)
    print(loss)
    print(y_pred)
    # print(y_pred_scale)
    print(wav.grad)