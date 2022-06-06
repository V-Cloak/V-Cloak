The source code of V-Cloak.

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