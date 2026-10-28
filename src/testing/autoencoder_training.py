import os
import json
import matplotlib
import matplotlib.pyplot as plt
import math

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
    # test_model,
    train,
    train_contrasitve_model,
    train_and_test_per_epoch,
    train_contrasitve_model_updated_to_account_for_partitions
)


device = setup_torch()
print(f"Current directory: {os.getcwd()}")
list_of_partition = os.listdir('json_data') 
list_of_partition.sort()


print(f"the current elements inside of json data: {list_of_partition}")





def train_partition(partition:str,type_of_testing:str = "training",epoch_counter=20):
    print(f"training on partition {partition}")
    data_tensor = torch.load(f"json_data/{partition}", map_location=device)


    dataset = Dataset_Autoencoder_Classifier(data_tensor, normalize=True)
    print(len(dataset))
    train_size = None

    if(type_of_testing == "training"):
        train_size = len(dataset)
    else:
        train_size = 0

    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    dataLoader = DataLoader_AutoEncoder_Classifier(
        train_dataset, batch_size=40, shuffle=True
    )
    dataLoader_test = DataLoader_AutoEncoder_Classifier(
        test_dataset, batch_size=30, shuffle=False
    )

    weights = [0.2]
    absolute_final_data = []

    for weight in weights:
        train_ac, test_ac = train_contrasitve_model(
            epoch_count=epoch_counter, dataLoader=dataLoader, weight_L=weight, dataLoader_test=dataLoader_test,type_="train" if type_of_testing == "training" else "test"
        )
        final_data = [train_ac, test_ac]
        absolute_final_data.append(final_data)

    if type_of_testing == "testing":
        with open("real_contrastive_new_without_weights.json", "w") as f:
            json.dump(absolute_final_data, f)
        # if len(absolute_final_data[0][0]) == 100:
        #     matplotlib.pyplot.plot(range(1, 101), absolute_final_data[0][0], label="Train Accuracy")

        # if len(absolute_final_data[0][1]) == 100:
        #     matplotlib.pyplot.plot(range(1, 101), absolute_final_data[0][1], label="Test Accuracy")
        # matplotlib.pyplot.xlabel("Epochs")
        # matplotlib.pyplot.ylabel("Accuracy")
        # matplotlib.pyplot.title("Train and Test Accuracy over Epochs")
        # matplotlib.pyplot.legend()
        # plt.show()
        print(absolute_final_data[0][1][-1])

    #NOTE: change the dimension of the latent space please 
epochs = 100
partition_batching = False 
if partition_batching:
    for partition in list_of_partition:
        list_of_test = ["dataset_updated_labled_Local.pt_part_17.pt", 
                        "dataset_updated_labled_Local.pt_part_18.pt" ,
                        "dataset_updated_labled_Local.pt_part_19.pt",
                        "dataset_updated_labled_Local.pt_part_20.pt"]

        list_of_sanity_check = [] #["dataset_updated_labled_Local.pt_part_12.pt",
                                #"dataset_updated_labled_Local.pt_part_13.pt"]


        if partition in list_of_sanity_check:
            train_partition(partition=partition,type_of_testing="training")
            train_partition(partition=partition,type_of_testing="testing", epoch_counter=1)

        elif partition in list_of_test:
            print("we are testing now !")
            train_partition(partition=partition,type_of_testing="testing", epoch_counter=1)
else:
    mse_loss, ce_loss = train_contrasitve_model_updated_to_account_for_partitions(epoch_count=epochs, weight_L=0.5, dataLoader_test=None,type_="training")
    epoch_space = [i for i in range(1,epochs + 1)]
    plt.plot(epoch_space, mse_loss)
    plt.plot(epoch_space, ce_loss)
    plt.show()
    