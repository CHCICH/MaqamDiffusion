import json
import matplotlib.pyplot as plt

with open("./results/data_hyperparam_autoencoder.json", "r") as file:
    data = json.load(file)

final_Loss = [final_loss[-1] for final_loss in data]
final_Loss.pop()
lr_rate = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01]
epoch_space = [i for i in range(1, 251)]

for i in range(len(lr_rate)):
    plt.title(f"lr={lr_rate[i]}")
    plt.plot(epoch_space, data[i])
    plt.show()
