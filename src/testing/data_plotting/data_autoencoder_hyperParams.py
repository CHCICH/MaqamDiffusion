import matplotlib.pyplot as plt
import json


data_collected = None

with open("best_epoch.json", "r") as f:
    data_collected = json.load(f)


train_acc, test_acc, train_loss, test_loss = data_collected

epoch_space = [i + 1 for i in range(len(train_acc))]

plt.plot(epoch_space, train_acc)

plt.plot(epoch_space, test_acc)
plt.show()

plt.plot(epoch_space, train_loss)

plt.plot(epoch_space, test_loss)

plt.show()
