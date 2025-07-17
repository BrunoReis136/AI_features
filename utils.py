import os
import librosa
import numpy as np
from tqdm import tqdm

def convert_wav_to_mel(input_dir,
                       output_dir,
                       sr=16000,
                       n_fft=1024,
                       hop_length=256,
                       n_mels=80,
                       power=1.0,
                       to_db=True,
                       verbose=True):
  
    os.makedirs(output_dir, exist_ok=True)
    file_list = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    iterator = tqdm(file_list, desc="Convertendo WAV → Mel") if verbose else file_list

    for filename in iterator:
        filepath = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".wav", ".npy"))

        y, _ = librosa.load(filepath, sr=sr)

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power
        )

        if to_db:
            mel = librosa.power_to_db(mel, ref=np.max)

        np.save(output_path, mel)


