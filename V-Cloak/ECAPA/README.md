The ECAPA-TDNN ASV can be downloaded from:
https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb

The vanilla ECAPA-TDNN of SpeechBrain has to be adapted to allow backpropagation to the wav input (modify the speechbrain.lobes.features.Fbank and the config .yaml file of the model).