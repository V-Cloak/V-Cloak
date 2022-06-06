import argparse

import glob
import torchaudio
import torch
import os
import librosa
import numpy as np
import soundfile as sf
from model.waveunet import Waveunet
from ecapa_tdnn_test import ecapa_tndnn

import time

def SNR_energy(signal, noise):
    E1 = torch.sum(torch.pow(signal, 2))
    E2 = torch.sum(torch.pow(noise, 2))

    return 10*torch.log10(E1/E2)

def SNR_max(signal, noise):
    A1 = torch.max(torch.abs(signal))
    A2 = torch.max(torch.abs(noise))

    print("A1: {}, A2: {}".format(A1, A2))

    return 20*torch.log10(A1/A2)

def validation(validation_path='./data/6930',
               generator = None,
               target_model = None,
               device=None,
               output_path_root = None,
               eps = 0.1,
               normalize = True,
               threshold = 0.267,
               M = 41641,
               N = 32089):
    # Set up validation parameters
    if output_path_root is None:
        output_path_root = './converted_sound_tmp_({})'\
                            .format(time.strftime("%Y-%m-%d_%H:%M:%S", 
                                    time.localtime()))
    # Setting up root dataset path and save path
    root_path = os.path.basename(validation_path)
    output_dir = os.path.join(output_path_root, root_path)
    # output_dir_original = output_dir + '_original'
    output_dir_converted = output_dir #+ '_converted'

    all_scores = []
    all_SNR1 = []
    all_SNR2 = []
    print("Generating Adversarial Examples from {}...".format(root_path))
    with torch.no_grad():
        for speaker in os.listdir(validation_path):
            # Setting up sub dataset path and save path
            validation_dir = os.path.join(validation_path, speaker)
            # output_dir_original_speaker =os.path.join(output_dir_original,speaker)
            output_dir_converted_speaker = os.path.join(output_dir_converted,speaker)

            os.makedirs(output_dir_converted_speaker,exist_ok=True)

            for filepath in glob.glob(validation_dir + "/*.flac"):
                waveform, sample_rate = torchaudio.load(filepath)

                if normalize:
                    waveform = (waveform/waveform.abs().max()).to(device)
                else:
                    waveform = (waveform).to(device)
                    print("false")

                pad = (M-N)//2
                n_segment = waveform.size(-1) // N + 1

                waveform_pad= torch.nn.functional.pad(waveform,
                                (pad,N*n_segment+pad-waveform.size(-1)),
                                "constant", 0).to(device)
                adv_delta = []
                for i in range(n_segment):
                    waveform_segment = waveform_pad[:,i*N:i*N+M]
                    waveform_segment2 = waveform_segment[:, pad:-pad]
                    adv_audio1 = generator((waveform_segment.unsqueeze(0), eps+torch.zeros(1,1)))['perturb'].squeeze(0)
                    adv_perturb1 = torch.clamp(adv_audio1-waveform_segment2, min=-eps, max=eps)
                    adv_perturb = torch.clamp(adv_perturb1+waveform_segment2, min=-1, max=1)-waveform_segment2

                    adv_delta.append(adv_perturb)

                adv_delta = torch.hstack(adv_delta)[:,:waveform.size(-1)]
                adv_audio = adv_delta + waveform
                SNR1 = SNR_energy(waveform, adv_delta).item()
                SNR2 = SNR_max(waveform, adv_delta).item()
                all_SNR1.append(SNR1)
                all_SNR2.append(SNR2)

                if target_model is not None:
                    adv_embedding = target_model.classifier.encode_batch(adv_audio)
                    real_embedding = target_model.classifier.encode_batch(waveform)
                    loss_adv = target_model.cosine_score(adv_embedding, real_embedding)
                    all_scores.append(loss_adv.item())

                    print("{},\tscore: {:.4f},\tnorm: {:.2f},\tSNR1: {:.2f},\tSNR2: {:.2f},\tsize: {}"\
                        .format(os.path.basename(filepath), 
                                loss_adv.item(), 
                                torch.norm(adv_delta, 2).item(), 
                                SNR1,
                                SNR2,
                                list(adv_audio.size())))
                adv_audio = adv_audio.cpu()
                torchaudio.save(filepath=os.path.join(output_dir_converted_speaker, os.path.basename(filepath)),
                         src=adv_audio,sample_rate=16000)
        # threshold = 0.5
        results = ("Stats: ASR: {:.2f}%(thres: {:.3f}),\tscore:{:.2f}-{:.2f}({:.2f}),\tSNR(E):{:.2f}-{:.2f}dB,\tSNR(MAX):{:.2f}-{:.2f}dB"\
                .format(100*np.sum(np.array(all_scores)<threshold)/len(all_scores), 
                threshold,
                min(all_scores), max(all_scores), sum(all_scores)/len(all_scores),
                min(all_SNR1),
                max(all_SNR1),
                min(all_SNR2),
                max(all_SNR2)))
        print(results)
        return results

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation_path', default='./data/6930',help="validation dataset path")
    parser.add_argument('--output_root_path', default='./converted_sound_Waveunet_test',help="path to store the validation output")
    parser.add_argument('--checkpoint', default='./model_checkpoint/_CycleGAN_Checkpoint',help="model checkpoint on which validation is done")
    parser.add_argument('--eps', type=float, default=0.1, help="max amplitude of the adversarial noise")
    parser.add_argument('--normalize', type=bool, default=False, help="normalize or not")
    parser.add_argument('--threshold', type=float, default=0.267, help="threshold of ecapa")
    
    args=parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    instruments = ["perturb"]
    features = 32
    levels = 6
    depth = 1
    sr = 16000
    channels = 1
    kernel_size = 5
    output_size = 2.0
    strides = 4
    conv_type = "gn"
    res = "fixed"
    separate = 0
    feature_growth = "double"
    num_features = [features * i for i in range(1, levels + 1)] if feature_growth == "add" else \
        [features * 2 ** i for i in range(0, levels)]
    target_outputs = int(output_size * sr)
    generator = Waveunet(
        num_inputs=channels,
        num_channels=num_features,
        num_outputs=channels,
        instruments=instruments,
        kernel_size=kernel_size,
        target_output_size=target_outputs,
        depth=depth,
        strides=strides,
        conv_type=conv_type,
        res=res,
        separate=separate).to(device)
    generator=torch.nn.DataParallel(generator)
    checkpoint = torch.load(args.checkpoint)
    generator.load_state_dict(checkpoint['model_waveunet'])

    # target attack model
    model_src = './CKPT+2021-02-27+12-48-32+00'
    hparams_src = 'hyperparams.yaml'
    savedir = './CKPT+2021-02-27+12-48-32+00'
    target_model = ecapa_tndnn(model_src=model_src, 
                                hparams_src=hparams_src, 
                                savedir=savedir, 
                                device=device, 
                                is_load_classifier=True)


    validation(validation_path=args.validation_path,
                generator = generator,
                target_model = target_model,
                device=device,
                output_path_root=args.output_root_path,
                eps = args.eps,
                normalize = args.normalize,
                threshold = args.threshold)
