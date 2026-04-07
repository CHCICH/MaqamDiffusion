import torch
from check import setup_torch
from models.Data_Load import DataLoader_AutoEncoder, Dataset_
from models.Autencoder import AutoEncoder, Classifier

# Dummy image-like data: batch=4, channels=1, height=256, width=256

device = setup_torch()


data_tensor = torch.load("../../json_data/dataset_updated.pt", map_location=device)
data_tensor = [t.flatten() for t in data_tensor]
data_tensor = [t.reshape((128, 1000)) for t in data_tensor]
data_tensor = torch.stack(data_tensor)


dataset = Dataset_(data_tensor, normalize=True)

train_size = 1200
test_size = len(dataset) - train_size
gen = torch.Generator(device=device)
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size], generator=gen
)

dataLoader = DataLoader_AutoEncoder(train_dataset, batch_size=40, shuffle=True)
dataLoader_test = DataLoader_AutoEncoder(test_dataset, batch_size=30, shuffle=False)


def test_model(model, dataLoader, Loss_fn, optimizer, num_epochs):

    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    num_batches = 0

    output_data = []

    with torch.no_grad():
        for batch in dataLoader:
            batch = batch.to(device)
            output = model(batch)
            total_mse += Loss_fn(output, batch).item()
            total_mae += torch.mean(torch.abs(output - batch)).item()
            num_batches += 1
            output_data.append(output.cpu().numpy())

    avg_mse = total_mse / max(1, num_batches)
    avg_mae = total_mae / max(1, num_batches)
    return avg_mse, avg_mae


def train(epoch, lr_rate, dataLoader, Loss_fn, optimizer):

    latent_model = AutoEncoder()
    latent_model.to(device)

    if Loss_fn == "MSE":
        Loss_fn = torch.nn.MSELoss()
    elif Loss_fn == "MAE":
        Loss_fn = torch.nn.L1Loss()
    elif Loss_fn == "BCE":
        Loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Invalid Loss Function")
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(latent_model.parameters(), lr=lr_rate)
    elif optimizer == "AdamW":
        optimizer = torch.optim.AdamW(latent_model.parameters(), lr=lr_rate)
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(latent_model.parameters(), lr=lr_rate)
    else:
        raise ValueError("Invalid Optimizer")

    schdeluer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    num_epochs = epoch
    for epoch in range(num_epochs):
        for batch in dataLoader:
            batch = batch.to(device)
            output = latent_model(batch)
            loss = Loss_fn(output, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    return latent_model, Loss_fn, optimizer, num_epochs


lr_rate = 0.02
epoch = 300
latent_model, Loss_fn, optimizer, num_epochs = train(
    epoch, lr_rate, dataLoader, Loss_fn="MSE", optimizer="Adam"
)


# here we are going to start the training for the Classifier in other words the bottle neck classifier


def train_classifier(epoch, lr_rate, dataLoader, Loss_fn, optimizer, input_size):
    latent_classifier = Classifier(input_size, 8)
    latent_classifier.to(device)

    schdeluer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    num_epochs = epoch
    for epoch in range(num_epochs):
        for batch in dataLoader:
            batch = batch.to(device)
            output = latent_model(batch)
            loss = Loss_fn(output, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
