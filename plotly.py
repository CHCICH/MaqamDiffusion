import matplotlib.pyplot as plt
import json


with open("loss_graph_new.json", "r") as file:
    data = json.load(file)

epoch_space = [i for i in range(1, 301)]
plt.plot(epoch_space, data)

plt.show()
