import os
import glob
import torch
import librosa
import numpy as np

def load_mels_with_labels_tuples(
    folder="../../data/",
    sr=22050,
    n_mels=128,
    hop_length=512,
    n_fft=2048,
    device="cpu"
):
    """
    Load all mp3 files as mel spectrograms and return a list of (tensor, label) tuples.
    
    Label is extracted as everything before the first '--' in the filename.
    
    Returns:
        List[Tuple[torch.FloatTensor, str]]  # (mel_matrix, label)
    """

    mp3_files = glob.glob(os.path.join(folder, "*.mp3"))
    if len(mp3_files) == 0:
        raise ValueError("No mp3 files found in folder.")

    mel_tensors = []
    labels = []
    min_T = float("inf")

    for path in mp3_files:
        filename = os.path.basename(path)
        print(f"processing {filename}")
        label = filename.split("--")[0]
        labels.append(label)
        print(f"labeling as {label}") 

        y, _ = librosa.load(
            path,
            sr=sr,
            mono=True,
            res_type="kaiser_fast"
        )

        if y.size == 0:
            print(f"⚠️ Skipping empty file: {filename}")
            continue

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)

        mel_tensor = torch.tensor(mel_db, dtype=torch.float32)
        mel_tensors.append(mel_tensor)
        min_T = min(min_T, mel_tensor.shape[1])

    dataset = [
        (mel[:, :min_T].to(device), label)
        for mel, label in zip(mel_tensors, labels)
    ]

    return dataset
