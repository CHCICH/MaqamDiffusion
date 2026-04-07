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
    device="cuda" if torch.cuda.is_available() else "cpu",
    with_labels=True,
    max_frames=1000,
    saved_path="dataset_updated"
):
    """
    Load all mp3 files as mel spectrograms and return a list of (tensor, label) tuples.
    
    Label is extracted as everything before the first '--' in the filename.
    Only keeps the first max_frames time steps (~2 min at default hop_length).
    
    Returns:
        List[Tuple[torch.FloatTensor, str]]  # (mel_matrix, label)
    """

    mp3_files = glob.glob(os.path.join(folder, "*.mp3"))
    if len(mp3_files) == 0:
        raise ValueError("No mp3 files found in folder.")

    mel_tensors = []
    labels = []
    music_start_time_list = []
    counter = 0
    for path in mp3_files:
        music_start_time_list.append(counter)
        filename = os.path.basename(path)
        label = filename.split("--")[0]
        

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
        n_mels, T = mel_db.shape  
        for i in range(0, T, max_frames):
            print(f"processing {filename} version {i//max_frames + 1} with label {label}")
            chunk = mel_db[:, i:i+max_frames]
            if chunk.shape[1] < max_frames:
                pad = max_frames - chunk.shape[1]
                chunk = np.concatenate([chunk, -80*np.ones((n_mels, pad))], axis=1)

            mel_tensor = torch.tensor(chunk, dtype=torch.float32)
            mel_tensors.append(mel_tensor)
            labels.append(label)
            counter += 1

    if not with_labels:
        dataset = mel_tensors
    else:
        dataset = [
            (mel.to(device), label)
            for mel, label in zip(mel_tensors, labels)
        ]
    
    torch.save(dataset, f"../../json_data/{saved_path}.pt")

    return dataset,music_start_time_list



