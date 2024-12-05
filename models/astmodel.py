import os
import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Resize
from transformers import ASTConfig, ASTModel, AutoFeatureExtractor
from datasets import load_dataset
# dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example", download_mode='force_redownload')
# audio_sample = dataset["train"]["audio"][0]["array"]


# 加载预训练的AST模型配置
config = ASTConfig.from_pretrained('/data/gaoyunlong/model/AST/')

# 加载预训练的AST模型
model = ASTModel.from_pretrained('/data/gaoyunlong/model/AST/', config=config)
feature_extractor = AutoFeatureExtractor.from_pretrained('/data/gaoyunlong/model/AST/')
# 将模型移到GPU（如果有）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 设置模型为评估模式
model.eval()
file_path = "/data/gaoyunlong/dataset/Audio/dataset_FSC89/audio/novel_test/soundscape_99_379260.wav"
waveform, sr = torchaudio.load(file_path)



# 确保采样率一致
if sr != 16000:
    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

# 提取梅尔频谱图
mel_spectrogram = librosa.feature.melspectrogram(
    y=waveform.numpy()[0],  # Assuming mono audio
    sr=16000,
    n_mels=128,
    n_fft=1024,
    hop_length=512
)

# 转换为对数尺度
mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

# 调整时间步长
target_time_steps = 1024
current_time_steps = mel_spectrogram.shape[1]

if current_time_steps < target_time_steps:
    # 如果时间步长不足，进行填充
    pad_width = target_time_steps - current_time_steps
    mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
elif current_time_steps > target_time_steps:
    # 如果时间步长超过目标值，进行裁剪
    mel_spectrogram = mel_spectrogram[:, :target_time_steps]

# 转换为 PyTorch 张量
mel_spectrogram = torch.tensor(mel_spectrogram).unsqueeze(0)  # Add batch dimension

# 查看最终形状
print(f"Mel spectrogram shape: {mel_spectrogram.shape}")
# # 确保采样率一致
# if sr != 16000:
#     waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

# # 提取梅尔频谱图
# mel_spectrogram = librosa.feature.melspectrogram(
#     y=waveform.numpy()[0],  # Assuming mono audio
#     sr=16000,
#     n_mels=128,
#     n_fft=1024,
#     hop_length=512
# )
# mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
# mel_spectrogram = torch.tensor(mel_spectrogram).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

# # inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")

# import pdb
# pdb.set_trace()
# inputs = mel_spectrogram
# [Batch, 1024, 128
# inputs = torch.randn(1, 1024, 128)


def train_transforms(batch):
    """Apply train_transforms across a batch."""
    subsampled_wavs = []
    for audio in batch[data_args.audio_column_name]:
        wav = random_subsample(
            audio["array"], max_length=data_args.max_length_seconds, sample_rate=feature_extractor.sampling_rate
        )
        subsampled_wavs.append(wav)
    inputs = feature_extractor(subsampled_wavs, sampling_rate=feature_extractor.sampling_rate)
    output_batch = {model_input_name: inputs.get(model_input_name)}
    output_batch["labels"] = list(batch[data_args.label_column_name])

    return output_batch


inputs = mel_spectrogram
inputs = inputs.to(device)
outputs = model(inputs)
import pdb
pdb.set_trace()
# outputs.cpu().numpy()
# cls_tokens = self.cls_token.expand(batch_size, -1, -1)
# distillation_tokens = self.distillation_token.expand(batch_size, -1, -1)
# torch.Size([1, 1214, 768])

import pdb
pdb.set_trace()
