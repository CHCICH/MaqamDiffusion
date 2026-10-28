import glob
import json
import os
import librosa
import numpy as np
import torch


def load_mels_with_labels_tuples(
    folder="../../data/",
    sr=22050,
    n_mels=128,
    hop_length=512,
    n_fft=2048,
    with_labels=True,
    max_frames=1024,
    saved_path="dataset_updated",
    labeling="first",
    is_Sub_Batching=True,
    batching_size=1000,
):
  """Load all mp3 files as mel spectrograms and save them into multiple .pt shard files

  whenever the buffer size limit is reached.
  """
  mp3_files = glob.glob(os.path.join(folder, "*.mp3"))
  if len(mp3_files) == 0:
    raise ValueError("No mp3 files found in folder.")

  # Ensure output directory exists
  output_dir = "../../json_data"
  os.makedirs(output_dir, exist_ok=True)

  songs_loaded = None
  if labeling == "bigger_set":
    data_loaded = json.load(open("../../arabic_1000_songs_.json", "r"))
    songs_loaded = data_loaded["songs"]

  music_start_time_list = []
  counter = 0
  part_idx = 0

  current_mels = []
  current_labels = []

  def save_shard(mels, labels_list, idx):
    if not mels:
      return
    if with_labels:
      dataset = [(mel, label) for mel, label in zip(mels, labels_list)]
    else:
      dataset = mels

    file_path = os.path.join(output_dir, f"{saved_path}_part_{idx}.pt")
    torch.save(dataset, file_path)
    print(
        f"💾 [Buffer Full] Saved shard {idx} with {len(mels)} samples to"
        f" {file_path}"
    )

  for path in mp3_files:
    music_start_time_list.append(counter)
    filename = os.path.basename(path)

    if labeling == "first":
      label = filename.split("--")[0]
    elif labeling == "bigger_set" and folder == "../../data_set/":
      un_clear_hash_name_of_label = filename.split(" ")[-1]
      cleared_hash_name_of_label = un_clear_hash_name_of_label.split(".")[0][
          1:-1
      ]
      song_searched = list(
          filter(
              lambda x: x["link"] is not None
              and "=" in x["link"]
              and x["link"].split("=")[1] == cleared_hash_name_of_label,
              songs_loaded,
          )
      )
      label = song_searched[0]["maqam_family"]

    y, _ = librosa.load(path, sr=sr, mono=True, res_type="kaiser_fast")
    if y.size == 0:
      print(f"⚠️ Skipping empty file: {filename}")
      continue

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)
    n_mels_dim, T = mel_db.shape

    for i in range(0, T, max_frames):
      chunk = mel_db[:, i : i + max_frames]
      if chunk.shape[1] < max_frames:
        pad = max_frames - chunk.shape[1]
        chunk = np.concatenate([chunk, -80 * np.ones((n_mels_dim, pad))], axis=1)

      mel_tensor = torch.tensor(chunk, dtype=torch.float32)

      current_mels.append(mel_tensor)
      current_labels.append(label)
      counter += 1

      if is_Sub_Batching:
        fullness_pct = (len(current_mels) / batching_size) * 100
        print(
            f"🔄 Buffer Status: {len(current_mels)}/{batching_size}"
            f" ({fullness_pct:.1f}% full)",
            end="\r",
        )

        if len(current_mels) >= batching_size:
          print()  # Clear line before saving log
          save_shard(current_mels, current_labels, part_idx)
          part_idx += 1
          current_mels = []
          current_labels = []

  if current_mels:
    print()
    if is_Sub_Batching:
      save_shard(current_mels, current_labels, part_idx)
    else:
      if with_labels:
        dataset = [(mel, label) for mel, label in zip(current_mels, current_labels)]
      else:
        dataset = current_mels
      file_path = os.path.join(output_dir, f"{saved_path}.pt")
      torch.save(dataset, file_path)
      print(
          f"💾 Saved final dataset with {len(current_mels)} samples to"
          f" {file_path}"
      )

def load_global_frequency_distribution_tuples(
    folder="../../data/",
    sr=22050,
    target_bins=1000,
    hop_length=512,
    n_fft=2048,
    with_labels=True,
    saved_path="dataset_global_fft",
    output_dir="../../json_data",
    labeling="first",
    is_Sub_Batching=False,
    batching_size=100,
):
  """Load all mp3 files, compute the global frequency distribution across the whole song 
  (collapsed over time), standardize to target_bins (1000 frequency bins), 
  and save them into PyTorch .pt shard files or a single file.
  """
  mp3_files = glob.glob(os.path.join(folder, "*.mp3"))
  if len(mp3_files) == 0:
    raise ValueError("No mp3 files found in folder.")

  # Ensure output directory exists
  os.makedirs(output_dir, exist_ok=True)

  songs_loaded = None
  if labeling == "bigger_set":
    data_loaded = json.load(open("../../arabic_1000_songs_.json", "r"))
    songs_loaded = data_loaded["songs"]

  music_start_time_list = []
  counter = 0
  part_idx = 0

  current_profiles = []
  current_labels = []

  def save_shard(profiles, labels_list, idx):
    if not profiles:
      return
    if with_labels:
      dataset = [(profile, label) for profile, label in zip(profiles, labels_list)]
    else:
      dataset = profiles

    file_path = os.path.join(output_dir, f"{saved_path}_part_{idx}.pt")
    torch.save(dataset, file_path)
    print(
        f"💾 [Buffer Full] Saved shard {idx} with {len(profiles)} frequency profiles to"
        f" {file_path}"
    )

  for path in mp3_files:
    music_start_time_list.append(counter)
    filename = os.path.basename(path)

    # Labeling logic preserved from your function
    if labeling == "first":
      label = filename.split("--")[0]
    elif labeling == "bigger_set" and folder == "../../data_set/":
      un_clear_hash_name_of_label = filename.split(" ")[-1]
      cleared_hash_name_of_label = un_clear_hash_name_of_label.split(".")[0][
          1:-1
      ]
      song_searched = list(
          filter(
              lambda x: x["link"] is not None
              and "=" in x["link"]
              and x["link"].split("=")[1] == cleared_hash_name_of_label,
              songs_loaded,
          )
      )
      label = song_searched[0]["maqam_family"]
    else:
      label = "unknown"

    y, _ = librosa.load(path, sr=sr, mono=True, res_type="kaiser_fast")
    if y.size == 0:
      print(f"⚠️ Skipping empty file: {filename}")
      continue

    # Compute mel-spectrogram (n_mels set to target_bins so frequency axis is 1000)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=target_bins
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)
    
    # Collapse time dimension by taking the mean across all time frames (axis=1)
    # This yields a 1D array of shape (target_bins,) representing frequency vs overall magnitude
    freq_profile = np.mean(mel_db, axis=1)

    # Convert to PyTorch 1D tensor of shape (1000,)
    profile_tensor = torch.tensor(freq_profile, dtype=torch.float32)

    current_profiles.append(profile_tensor)
    current_labels.append(label)
    counter += 1

    if is_Sub_Batching:
      fullness_pct = (len(current_profiles) / batching_size) * 100
      print(
          f"🔄 Buffer Status: {len(current_profiles)}/{batching_size}"
          f" ({fullness_pct:.1f}% full)",
          end="\r",
      )

      if len(current_profiles) >= batching_size:
        print()  # Clear line before saving log
        save_shard(current_profiles, current_labels, part_idx)
        part_idx += 1
        current_profiles = []
        current_labels = []

  # Save remaining or full dataset
  if current_profiles:
    print()
    if is_Sub_Batching:
      save_shard(current_profiles, current_labels, part_idx)
    else:
      if with_labels:
        dataset = [(profile, label) for profile, label in zip(current_profiles, current_labels)]
      else:
        dataset = current_profiles
      file_path = os.path.join(output_dir, f"{saved_path}.pt")
      torch.save(dataset, file_path)
      print(
          f"💾 Saved final dataset with {len(current_profiles)} frequency profiles to"
          f" {file_path}"
      )

  return music_start_time_list