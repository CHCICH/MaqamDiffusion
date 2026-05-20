import matplotlib.pyplot as plt
import json


data_collected = None

with open("real_contrastive_now.json", "r") as f:
    data_collected = json.load(f)

for i in range(len(data_collected)):
    train_acc, test_acc = data_collected[i]

    epoch_space = [i + 1 for i in range(len(train_acc))]

    plt.plot(epoch_space, train_acc)
    plt.plot(epoch_space, test_acc)
    plt.show()
