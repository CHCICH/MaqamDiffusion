import os
import json

import torch
from check import setup_torch
from models.Data_Load import (
    DataLoader_AutoEncoder_Classifier,
    Dataset_Autoencoder_Classifier,
)
from models.Autencoder import AutoEncoder, Classifier


from helper_training_functions import (
    converter_class_idx,
    convert_label_list,
    test_model,
    train_classifier,
    train,
    train_contrasitve_model,
    train_and_test_per_epoch,
)


device = setup_torch()
print(f"Current directory: {os.getcwd()}")

data_tensor = torch.load("json_data/dataset_updated.pt", map_location=device)

dataset = Dataset_Autoencoder_Classifier(data_tensor, normalize=True)

train_size = 900
test_size = len(dataset) - train_size
gen = torch.Generator(device=device)
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size], generator=gen
)

dataLoader = DataLoader_AutoEncoder_Classifier(
    train_dataset, batch_size=40, shuffle=True
)
dataLoader_test = DataLoader_AutoEncoder_Classifier(
    test_dataset, batch_size=30, shuffle=False
)


LR_rate = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.5]

# Train the model


# end of training

latent_autoencoder = AutoEncoder()
#   latent_autoencoder.load_state_dict(torch.load("model_weights.pth", map_location=device))
#  latent_autoencoder.to(device)
#    latent_autoencoder.eval()
classifier = Classifier(2048, 8)
# classifier.load_state_dict(torch.load("classifier_weight.pth", map_location=device))
# classifier.to(device)
# classifier.eval()

# correct_ones = 0
# total = 0
latent_classifier = False

if latent_classifier:
    train_a, test_a, train_l, test_l = train_and_test_per_epoch(
        classifier, 100, dataLoader, dataLoader_test
    )
    final_data = [train_a, test_a, train_l, test_l]

    with open("best_epoch.json", "w") as f:
        json.dump(final_data, f)

else:
    weights = [0.1]
    absolute_final_data = []
    for weight in weights:
        train_ac, test_ac = train_contrasitve_model(
            100, dataLoader, weight, dataLoader_test
        )
        final_data = [train_ac, test_ac]
        absolute_final_data.append(final_data)

    with open("real_contrastive_now.json", "w") as f:
        json.dump(absolute_final_data, f)
