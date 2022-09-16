The source code of V-Cloak.

# Data Preparation

Put the training set, validation set in the `Datasets` folder, e.g., VoxCeleb1

VoxCeleb1 can be downloaded here:
https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html

LibriSpeech can be downloaded here:
https://www.openslr.org/12/

To modify your path to the training set, modify `train.py`:

```python
self.filelist = glob.glob("Datasets/train-clean-100/*/*/*.flac")+glob.glob("Datasets/train-other-500/*/*/*.flac")
```

# ASV model preparation

Put the pre-trained ECAPA-TDNN model in the `ECAPA` folder.
The ECAPA-TDNN ASV can be downloaded from:
https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb

The vanilla ECAPA-TDNN of SpeechBrain has to be adapted to allow backpropagation to the wav input, i.e., to modify the `speechbrain.lobes.features.Fbank` and the config `.yaml` file of the model. Specifically, remove the `torch.no_grad()`. But it seems that the new version of SpeechBrain already allows it..

The code in this repo allows optimizing using multiple ASVs. Put the pre-trained XVECTOR model in the `XVECTOR` folder.

The XVECTOR ASV can be downloaded from:
https://huggingface.co/speechbrain/spkrec-xvect-voxceleb


To optimize with only one ASV, just modify the `train.py`, remove the code related to `self.target_model2`.

# Training

To train V-Cloak from scratch,

```shell
python train.py
```

To load the newest checkpoint,

```shell
python train.py --resume_training_at=newest
```

To validate before training,

```shell
python train.py --resume_training_at=newest --valid_first
```

If the loss cannot converge, tune the weights of each loss term.

# Anonymization

To anonymize audios, specify a folder contain the audios, e.g.,

```shell
python validation.py --validation_path=./Datasets/dev-clean --checkpoint=./model_checkpoint_GPU/_CheckPoint_newest --eps=0.1 --threshold=0.267
```

`validation.py` is limited for *dev-clean* -like folder. If you want to validate your own folder, just modify the below line in the code:

```python
for filepath in glob.glob(validation_path + "/*/*/*.flac")
```

# Code structure

```shell
.
|-- Datasets
|   |-- dev-clean
|   |-- test-clean
|   |-- train-clean-100
|   `-- train-other-500
|-- ECAPA
|   |-- CKPT.yaml
|   |-- brain.ckpt
|   |-- classifier.ckpt
|   |-- counter.ckpt
|   |-- custom.py -> xxxx
|   |-- dataloader-TRAIN.ckpt
|   |-- embedding_model.ckpt
|   |-- hyperparams.yaml
|   |-- label_encoder.ckpt
|   |-- label_encoder.txt
|   |-- mean_var_norm_emb.ckpt
|   |-- normalizer.ckpt
|   `-- optimizer.ckpt
|-- XVECTOR
|   |-- classifier.ckpt
|   |-- custom.py -> xxxx
|   |-- embedding_model.ckpt
|   |-- hyperparams.yaml
|   |-- label_encoder.ckpt
|   `-- mean_var_norm_emb.ckpt
|-- deepspeech4loss.py
|-- ecapa_tdnn_test.py
|-- masker.py
|-- model
|   |-- conv.py
|   |-- crop.py
|   |-- resample.py
|   |-- utils.py
|   `-- waveunet.py
|-- model_checkpoint_GPU
|-- new_pytorch_deep_speech.py
|-- requirements.txt
|-- train.py
|-- trainingDataset.py
`-- validation.py
```

- Datasets
	+ README.md
	+ Put the training set here, e.g., voxceleb1.
- ECAPA
	+ README.md
	+ Download the ECAPA-TDNN here.
- deepspeech4loss.py
	+ Compute the ASR loss (CTC, GPG).
- ecapa_tdnn_test.py
	+ tools of ECAPA-TDNN
- masker.py
	+ Compute the psychoacoustic-based loss.
- model
	+ The anonymizer of V-Cloak.
- new_pytorch_deep_speech.py
	+ A modified DeepSpeech model to allow torch.DataParallel.
- train.py
	+ Train the anonymizer.
- trainingDataset.py
	+ Data preparation
- validation.py
	+ Generate anonymized audios.
- requirements.txt
- test.wav