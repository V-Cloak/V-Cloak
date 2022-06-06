import torchaudio
import os
import torch
from speechbrain.pretrained import EncoderClassifier


class ecapa_tndnn:

    def __init__(self, model_src, hparams_src, savedir, device, is_load_classifier=False):
        self.model_src = model_src
        self.hparams_src = hparams_src
        self.savedir = savedir
        self.device = device
        if is_load_classifier:
            self.classifier = self.load_classifier()
        else:
            self.classifier = None

    def load_classifier(self):
        classifier = EncoderClassifier.from_hparams(source=self.model_src, hparams_file=self.hparams_src,
                                                    savedir=self.savedir, freeze_params=True, run_opts={"device": self.device, "data_parallel_backend": True})
        return classifier

    def compute_emb(self, mcep, wav_lens=None, normalize=False):

        if wav_lens is None:
            wav_lens = torch.ones(
                mcep.shape[0], device=self.classifier.device)

        embeddings = self.classifier.mods.embedding_model(
            mcep, wav_lens)

        if normalize:
            embeddings = self.classifier.hparams.mean_var_norm_emb(
                embeddings, torch.ones(
                    embeddings.shape[0], device=self.classifier.device)
            )
        return embeddings
    
    def cosine_score(self,emb1,emb2):

        # Cosine similarity initialization
        similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        score = similarity(emb1, emb2)

        mean_score=torch.mean(score,dim=0,keepdim=False)[0]

        return mean_score

    def cosine_score_raw(self,emb1,emb2):

        # Cosine similarity initialization
        similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        score = similarity(emb1, emb2)

        return score

if __name__=='__main__':
    model_src = './ECAPA'
    hparams_src = 'hyperparams.yaml'
    savedir = './ECAPA'
    
    wavs, fs =torchaudio.load("./test.wav")
    et = ecapa_tndnn(model_src=model_src, hparams_src=hparams_src, savedir=savedir,is_load_classifier=True)
    embeddings = et.classifier.encode_batch(wavs)
    wav_lens = torch.ones(
                    wavs.shape[0], device=et.classifier.device)
    feats = et.classifier.modules.compute_features(wavs)
    feats = et.classifier.modules.mean_var_norm(feats, wav_lens)
    feats = feats[:, :128, :]
    print("feats shape",feats.shape)
    print(et.compute_emb(feats))
    print(et.cosine_score(embeddings,embeddings,1))




