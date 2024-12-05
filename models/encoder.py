import numpy as np
import torch
import torch.nn as nn

from .encoders import AudioEncoder
from .encoders import TextEncoder



class CustomPENGI(nn.Module):
    def __init__(self,args,pengi):
        super().__init__()

        self.args = args
        pengi_args  = pengi.args
        self.pengi_args = pengi_args
        self.audio_encoder = AudioEncoder(
                    pengi_args.audioenc_name, pengi_args.out_emb, pengi_args.d_proj,
                    pengi_args.sampling_rate, pengi_args.window_size, pengi_args.hop_size, pengi_args.mel_bins, pengi_args.fmin, pengi_args.fmax, pengi_args.classes_num, 
                    pengi_args.specaug, pengi_args.mixup, pengi_args.use_pretrained_audioencoder, pengi_args.freeze_audio_encoder_weights,
                    pengi_args.use_precomputed_melspec, pengi_args.pretrained_audioencoder_path)

        # load the weights of the pengi pre-trained audio and text encoders
        print("\n\nPALM: loading the weights of the pengi pre-trained audio and text encoders ...\n\n")
        self.audio_encoder.load_state_dict(pengi.model.audio_encoder.state_dict())
        self.audio_encoder.eval()
        self.device = args.device

    def forward(self, audio):

        audio_features = self.audio_encoder(audio)[0] # audio_features shape [n_audio_files, 1024]
        return audio_features


