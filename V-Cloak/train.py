import os, glob
import numpy as np
import argparse
import torch
import time
from tqdm import tqdm
import pickle
import jiwer
import torchaudio

# Personal packages
from ecapa_tdnn_test import ecapa_tndnn
from validation import validation
from model.waveunet import Waveunet
from trainingDataset import trainingDataset
from deepspeech4loss import DeepSpeech4Loss
from masker import Masker

GPU = [str(g) for g in [0,1,2,3]]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(GPU)

# def batch_fbank(x, sr):
#     '''

#     :param: x, size of [batch, channel, time]

#     return: mfb, size of [batch, 80, frame]
#     '''
#     mfb_all = []
#     for i in range(x.size(0)):
#         mfb = torchaudio.compliance.kaldi.fbank(x[i], 
#                             high_freq = sr/2,
#                             low_freq = 0,
#                             num_mel_bins = 80,
#                             preemphasis_coefficient = 0.97,
#                             sample_frequency = sr,
#                             use_log_fbank = True,
#                             use_power = True,
#                             window_type = 'povey')
#         mfb_all.append(
#                     torch.transpose(mfb, 0, 1)\
#                     .unsqueeze(0)
#                     )
#     mfb_t = torch.vstack(mfb_all)

#     return mfb_t


class VCloakTraining(object):
    def __init__(self,
                 model_checkpoint,
                 restart_training_at=None,
                 verbose=False,
                 startat=1,
                 validate_first=True):
        self.verbose = verbose
        self.start_epoch = startat
        self.validate_first = validate_first
        self.restart_training_at = restart_training_at
        self.num_epochs = 65  # 5000
        self.mini_batch_size = 256-64  # 1

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Generator waveunet
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

        num_features = [features*i for i in range(1, levels+1)] if feature_growth == "add" else \
                       [features*2**i for i in range(0, levels)]
        target_outputs = int(output_size * sr)
        self.waveunet = Waveunet(
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
                                separate=separate
                            ).to(self.device)
        
        self.N_all = self.waveunet.shapes['input_frames']
        self.N_out = self.waveunet.shapes['output_frames']
        self.pad = (self.N_all-self.N_out)//2

        print("Input frames: {} --> output frames: {}, pad: {}"\
                .format(self.N_all, self.N_out, self.pad))

        self.waveunet = torch.nn.DataParallel(self.waveunet)

        # Optimizer
        g_params = list(self.waveunet.parameters())

        # Initial learning rates
        self.generator_lr = 4e-4

        # Learning rate decay
        self.generator_lr_decay = self.generator_lr / 200000

        # Starts learning rate decay from after this many iterations have passed
        self.start_decay = 14000  # 200000

        self.generator_optimizer = torch.optim.Adam(
            g_params, lr=self.generator_lr, betas=(0.5, 0.999))

        # To Load save previously saved models
        self.modelCheckpoint = model_checkpoint
        os.makedirs(self.modelCheckpoint, exist_ok=True)

        # Storing Discriminatior and Generator Loss
        self.generator_loss_store = []

        self.file_name = 'log_GPU{}_{}.txt'.format(
                ''.join(GPU),
                time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))

        # target victim model (ASV)
        model_src = './ECAPA'
        hparams_src = 'hyperparams.yaml'
        savedir = './ECAPA'
        self.target_model = ecapa_tndnn(model_src=model_src, 
                                        hparams_src=hparams_src, 
                                        savedir=savedir, 
                                        device=self.device, 
                                        is_load_classifier=True)

        model_src = './XVECTOR'
        hparams_src = 'hyperparams.yaml'
        savedir = './XVECTOR'
        self.target_model2 = ecapa_tndnn(model_src=model_src, 
                                        hparams_src=hparams_src, 
                                        savedir=savedir, 
                                        device=self.device, 
                                        is_load_classifier=True)

        # target ASR model
        self.asr_model =DeepSpeech4Loss(pretrained_model='librispeech',
                                        device_type = "gpu",
                                        device=self.device)

        # Preparing Dataset
        self.filelist = glob.glob("Datasets/train-clean-100/*/*/*.flac")+glob.glob("Datasets/train-other-500/*/*/*.flac")
        self.n_samples = len(self.filelist)

        self.msker = Masker(device=self.device)

        self.dataset = trainingDataset(wavs_list=self.filelist, 
                                        seed=1234,
                                        normalize_mean=0.55,
                                        normalize_sigma=0.2,
                                        N=self.N_all,
                                        M=self.N_out,
                                        masker=self.msker)

        if restart_training_at is not None:
            # Training will resume from previous checkpoint`        `
            self.start_epoch = self.loadModel(restart_training_at)
            print("Training resumed, starting at", self.start_epoch)

    def adjust_lr_rate(self, optimizer, name='generator'):
        if name == 'generator':
            self.generator_lr = max(
                0., self.generator_lr - self.generator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.generator_lr

            if self.generator_lr == 0:
                print("WARNING: learning rate of generator is set to 0.")

    def reset_grad(self):
        self.generator_optimizer.zero_grad()

    def train(self):
        if self.validate_first and self.restart_training_at:
            valid_results = validation(generator = self.waveunet, 
                                        target_model = self.target_model,
                                        device = self.device,
                                        eps = 0.1,
                                        M = self.N_all,
                                        N = self.N_out,
                                        output_path_root = './converted_sound_tmp_GPU{}({})'\
                                            .format(
                                            ''.join(GPU),    
                                            time.strftime("%Y-%m-%d_%H:%M:%S", 
                                                        time.localtime()
                                                    )
                                            )
                                        )
        # Training Begins
        for epoch in range(self.start_epoch, self.num_epochs):
            start_time_epoch = time.time()

            num_of_iteration_per_epoch = int((len(self.dataset)/self.mini_batch_size))+1
            print('Total iterations:', num_of_iteration_per_epoch)

            train_loader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                       batch_size=self.mini_batch_size,
                                                       shuffle=True,
                                                       drop_last=False,
                                                       num_workers=32)

            pbar = tqdm(train_loader)

            # min and max of epsilon
            local_min = 0.001
            local_max = 0.15

            self.waveunet.train()

            # train batches
            i = 0   # iteration
            for (real_wav, theta_batch, original_max_psd_batch) in pbar:
                # align center
                real_wav2 = real_wav[:,:,self.pad:-self.pad]

                # Check if decay
                num_iterations = num_of_iteration_per_epoch * epoch + i
                if num_iterations > self.start_decay:
                    self.adjust_lr_rate(
                        self.generator_optimizer, name='generator')

                    if i == 0:
                        print("generator_lr", self.generator_lr)

                real_wav2 = real_wav2.to(self.device).float()

                ######################## Generate adversarial samples ###################################

                # sample a batch of epsilons from a distribution of clamped Gaussian
                mu=(local_min+local_max)/2
                epislon=((local_max-local_min)/2)
                eps = torch.clamp(mu+epislon*torch.randn(real_wav2.size(0), 1),
                            min=local_min, max=local_max).to(self.device)

                # waveunet outputs the adversarial audio
                adv_audio1 = self.waveunet((real_wav, eps))['perturb']

                # the adversarial perturbation
                adv_perturb1 = adv_audio1 - real_wav2

                # make sure that adv_perturb1 < eps
                adv_perturb1 = torch.clamp(adv_perturb1 - eps.unsqueeze(-1), max=0) + eps.unsqueeze(-1)

                # make sure that adv_perturb1 > -eps
                adv_perturb1 = torch.clamp(adv_perturb1 + eps.unsqueeze(-1), min=0) - eps.unsqueeze(-1)

                # make sure that (adv_perturb1 + real_wav2) in [-1, 1]
                adv_perturb = torch.clamp(adv_perturb1 + real_wav2, min=-1, max=1)-real_wav2

                # the finally adversarial audio
                adv_audio = adv_perturb + real_wav2


                ######################### Optimize generator #######################################

                self.reset_grad()
                begin_to_psy = 1
                if epoch >= begin_to_psy:
                    if epoch == begin_to_psy and i == 0:
                        print("Begin to optimize with psychoacoustics.")
                    loss_G_fake = self.msker.batch_forward_2nd_stage(
                                            local_delta_rescale=adv_perturb.squeeze(1),
                                            theta_batch=theta_batch.to(self.device),
                                            original_max_psd_batch=original_max_psd_batch.to(self.device),
                                            )

                    loss_G_fake_max = 1.0e7
                    if torch.isnan(loss_G_fake) or torch.isinf(loss_G_fake):
                        print('Warning: loss_G_fake is {}'.format(loss_G_fake.item()))
                        loss_G_fake = torch.FloatTensor([loss_G_fake_max]).to(self.device)
                    elif loss_G_fake > loss_G_fake_max:
                        loss_G_fake = torch.FloatTensor([loss_G_fake_max]).to(self.device)
            
                else:
                    loss_G_fake = torch.FloatTensor([0]).to(self.device)

                # calculate perturbation norm
                loss_perturb = torch.mean(torch.norm(adv_perturb, 2, dim=(-1, -2)))

                # calculate target model loss
                if self.target_model is not None:

                    ##################### ASV model ##########################################
                    # embeddings before and after generation
                    adv_A_embedding = self.target_model.classifier.encode_batch(adv_audio.squeeze(1))
                    real_A_embedding = self.target_model.classifier.encode_batch(real_wav2.squeeze(1))

                    # the second ASV, if not, comment them
                    adv_A_embedding2 = self.target_model2.classifier.encode_batch(adv_audio.squeeze(1))
                    real_A_embedding2 = self.target_model2.classifier.encode_batch(real_wav2.squeeze(1))
                    
                    # perturbation bound
                    c = -0.2
                    score = self.target_model.cosine_score(adv_A_embedding, real_A_embedding)
                    
                    # the second ASV, if not, comment them
                    score2 = self.target_model2.cosine_score(adv_A_embedding2, real_A_embedding2)

                    loss_adv = torch.max(score-c, 
                                        torch.zeros_like(score, device=self.device))
                    
                    # the second ASV, if not, comment them
                    loss_adv += torch.max(score2-0.8, 
                                        torch.zeros_like(score, device=self.device))

                    # loss_adv /= 2

                else:
                    print("Warining: self.target_model is None")
                    loss_adv = 0

                if self.asr_model is not None:
                    # real phonetic posteriorgram (ppg) / graphemic posteriorgram (gpg)
                    ppg_real = self.asr_model.compute_ppg(real_wav2.squeeze(1).detach())

                    ppg_loss = self.asr_model.compute_ppgloss(adv_audio.squeeze(1), ppg_real)
                    # real transcription
                    y = self.asr_model.predict(real_wav2.squeeze(1).detach().cpu().numpy(), 
                                                batch_size=real_wav2.size(0))

                    # combine ppg loss and ctc loss
                    # ppg_loss, ctc_loss = self.asr_model.compute_ppgctc_loss(
                    #                                             adv_audio.squeeze(1),
                    #                                             y,
                    #                                             ppg_real)
                    # ctc_loss = ctc_loss/real_wav2.size(0)

                    # trade-off between ppg and ctc
                    if torch.isnan(ppg_loss):
                        ppg_loss = torch.FloatTensor([20]).to(self.device)
                    
                    asr_alpha = 1 # 0.88
                    loss_asr = asr_alpha*ppg_loss # + (1-asr_alpha)*ctc_loss
                    ctc_loss = torch.FloatTensor([0]).to(self.device)


                # generator loss terms

                # norm term
                perturb_loss_lambda = 0.1

                # ASV term
                adv_loss_lambda = 20*1.5

                if score < 0.15:
                    # discriminator term
                    G_loss_lambda = 100e-8 # 100e-6

                    # ASR term
                    asr_loss_lambda = 1

                elif score < 0.3:
                    G_loss_lambda = 70e-8 # 70e-6
                    asr_loss_lambda = 0.8

                else:
                    G_loss_lambda = 10e-8 # 1e-6
                    asr_loss_lambda = 0.5
            
                assert not torch.isnan(loss_G_fake), "loss_G_fake is NaN"
                assert not torch.isnan(loss_adv), "loss_adv is NaN"
                assert not torch.isnan(loss_perturb), "loss_perturb is NaN"
                assert not torch.isnan(loss_asr), "loss_asr is NaN"


                loss_G = G_loss_lambda * loss_G_fake + \
                        adv_loss_lambda * loss_adv + \
                        perturb_loss_lambda * loss_perturb +\
                        asr_loss_lambda * loss_asr 

                loss_G.backward()

                self.generator_optimizer.step()

                ######################### Optimize discriminator #######################################

                # print information of this iteration.

                if (i) % 1 == 0:
                    y_pred = self.asr_model.predict(adv_audio.squeeze(1).detach().cpu().numpy(), 
                                                batch_size=adv_audio.size(0))
                    y = list(y)
                    y_pred = list(y_pred)

                    y_i = 0
                    while y_i < len(y):
                        if len(y[y_i].strip()) == 0:
                            y.pop(y_i)
                            y_pred.pop(y_i)
                        else:
                            y_i += 1
                    # while y.count('') > 0:
                    #     inx = y.index('')
                    #     y.pop(inx)
                    #     y_pred.pop(inx)

                    try:
                        wer = jiwer.wer((y), (y_pred))
                    except ValueError as e:
                        print(e)
                        print(y)
                        wer = 1

                    pbar.set_description("Ep:{}, GLoss: {:.4f}, cos: {:.2f}, {:.2f}({:.2f}), asr: {:.2f}+{:.2f}({:.2f}, wer {:.2f}), G_fake: {:.2f}({:.2f}), norm: {:.2f}({:.2f})".format(
                        epoch, 
                        loss_G.item(), 
                        score.item(), score2.item(), (adv_loss_lambda*loss_adv).item(),
                        ppg_loss.item(), ctc_loss.item(), (asr_loss_lambda*loss_asr).item(), wer,
                        loss_G_fake.item(), (G_loss_lambda*loss_G_fake).item(),
                        loss_perturb.item(), (perturb_loss_lambda*loss_perturb).item()
                    ))
                
                # update iteration
                i += 1

            # save model
            if epoch % 1 == 0:
                end_time = time.time()
                store_to_file = "Ep:{}, GLoss: {:.4f}, cos: {:.2f}({:.2f}), asr: {:.2f}({:.2f}), G_fake: {:.2f}({:.2f}), norm: {:.2f}({:.2f}), Time: {:.2f}\n\n".format(
                    epoch,
                    loss_G.item(), 
                    score.item(), (adv_loss_lambda*loss_adv).item(),
                    loss_asr.item(), (asr_loss_lambda*loss_asr).item(),
                    loss_G_fake.item(), (G_loss_lambda*loss_G_fake).item(),
                    loss_perturb.item(), (perturb_loss_lambda*loss_perturb).item(),
                    end_time - start_time_epoch)
                self.store_to_file(store_to_file)
                print("Epoch: {} Generator Loss: {:.4f}, Time: {:.2f}\n\n".format(
                    epoch, loss_G.item(), end_time - start_time_epoch))

                # Save the Entire model
                print("Saving model Checkpoint  ......")
                store_to_file = "Saving model Checkpoint  ......"
                self.store_to_file(store_to_file)

                savetime =  time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) 
                ckpt_name = '_VCloak_CheckPoint_{}_e{}'.format(savetime, str(epoch))
                self.saveModelCheckPoint(epoch, '{}'.format(
                    self.modelCheckpoint + ckpt_name))
                self.saveModelCheckPoint(epoch, '{}'.format(
                    self.modelCheckpoint + '_VCloak_CheckPoint_newest'))
                print("Model Saved!")

            # generate validation data
            if epoch % 1 == 0:
                # Validation Set
                validation_start_time = time.time()
                valid_results = validation(generator = self.waveunet, 
                                            target_model = self.target_model,
                                            device = self.device,
                                            eps = local_max,
                                            M = self.N_all,
                                            N = self.N_out,
                                            output_path_root = './converted_sound_tmp_GPU{}({})'\
                                                .format(
                                                ''.join(GPU),    
                                                time.strftime("%Y-%m-%d_%H:%M:%S", 
                                                            time.localtime()
                                                        )
                                                )
                                            )
                self.store_to_file(valid_results)

                validation_end_time = time.time()
                store_to_file = "Time taken for validation Set: {}".format(
                    validation_end_time - validation_start_time)
                self.store_to_file(store_to_file)

                print("Time taken for validation Set: {}".format(
                    validation_end_time - validation_start_time))


    def savePickle(self, variable, fileName):
        with open(fileName, 'wb') as f:
            pickle.dump(variable, f)

    def loadPickleFile(self, fileName):
        with open(fileName, 'rb') as f:
            return pickle.load(f)

    def store_to_file(self, doc):
        doc = doc + "\n"
        with open(self.file_name, "a") as myfile:
            myfile.write(doc)

    def saveModelCheckPoint(self, epoch, PATH):
        torch.save({
            'epoch': epoch,
            'generator_loss_store': self.generator_loss_store,
            'model_waveunet': self.waveunet.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
        }, PATH)

    def loadModel(self, PATH):
        checkPoint = torch.load(PATH)

        self.waveunet.load_state_dict(
            state_dict=checkPoint['model_waveunet'])

        self.generator_optimizer.load_state_dict(
            state_dict=checkPoint['generator_optimizer'])
        epoch = int(checkPoint['epoch']) + 1
        self.generator_loss_store = checkPoint['generator_loss_store']
        return epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train adversarial example generator V-Cloak")

    ############################### specify training options #######################################

    model_checkpoint = './model_checkpoint_GPU/'
    resume_training_at = None

    parser.add_argument('--model_checkpoint', type=str,
                        help="location where you want to save the model", default=model_checkpoint)
    parser.add_argument('--resume_training_at', type=str,
                        help="Location of the pre-trained model to resume training",
                        default=resume_training_at)
    parser.add_argument('--startat', type=int,
                        help="start epoch",
                        default=1)
    parser.add_argument('--verbose', action="store_true", help="print the network shape")
    parser.add_argument('--valid_first', action="store_true", help="validation before training")

    argv = parser.parse_args()

    model_checkpoint = argv.model_checkpoint

    if argv.resume_training_at == 'newest':
        resume_training_at = './model_checkpoint_GPU/_VCloak_CheckPoint_newest'
    else:
        resume_training_at = argv.resume_training_at
        
    verbose = argv.verbose

    vcloak = VCloakTraining(
                            model_checkpoint=model_checkpoint,
                            restart_training_at=resume_training_at,
                            verbose = verbose,
                            startat = argv.startat,
                            validate_first=argv.valid_first
                            )
    vcloak.train()
