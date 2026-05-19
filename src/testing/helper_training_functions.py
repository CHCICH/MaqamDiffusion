import torch
from check import setup_torch
from models.Autencoder import AutoEncoder, Classifier

device = setup_torch()


def converter_class_idx(class_idx, in_mapping=False):
    inverse_class_mapping = {
        "bayat": 1,
        "hijaz": 7,
        "hijazkar": 6,
        "kurd": 0,
        "nahawand": 4,
        "rast": 3,
        "saba": 5,
        "segah": 2,
    }

    class_mapping = {
        1: "bayat",
        7: "hijaz",
        6: "hijazkar",
        0: "kurd",
        4: "nahawand",
        3: "rast",
        5: "saba",
        2: "segah",
    }

    if in_mapping:
        return inverse_class_mapping.get(class_idx, -1)
    else:
        return class_mapping.get(class_idx, -1)


def convert_label_list(list_label):
    new_list = []

    for label in list_label:
        new_elem = int(converter_class_idx(label, True))
        y = [0, 0, 0, 0, 0, 0, 0, 0]
        y[new_elem] = 1
        new_list.append(y)

    return torch.tensor(new_list, dtype=torch.float32)


def train_classifier(epoch, lr_rate, dataLoader, Loss_fn, optimizer, input_size):

    latent_classifier = Classifier(input_size, 8)

    latent_classifier.to(device)

    latent_autoencoder = AutoEncoder()

    latent_autoencoder.load_state_dict(
        torch.load("model_weights.pth", map_location=device)
    )

    latent_autoencoder.to(device)

    latent_autoencoder.eval()

    if optimizer == "Adam":
        optimizer = torch.optim.Adam(latent_classifier.parameters(), lr=lr_rate)

    elif optimizer == "AdamW":
        optimizer = torch.optim.AdamW(latent_classifier.parameters(), lr=lr_rate)

    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(latent_classifier.parameters(), lr=lr_rate)

    else:
        raise ValueError("Invalid Optimizer")

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.955)

    num_epochs = epoch

    loss_value = []

    for epoch in range(num_epochs):
        latent_classifier.train()

        for batch in dataLoader:
            images, labels = batch

            images = images.to(device)

            labels = convert_label_list(labels).to(device)

            with torch.no_grad():
                encoded_images = latent_autoencoder.encode_latent(images)

                encoded_images = torch.flatten(encoded_images, start_dim=1)

            output = latent_classifier(encoded_images)

            loss = Loss_fn(output, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        scheduler.step()

        print(
            f"Cross_Entropy lr = {lr_rate} Loss {epoch + 1}/{num_epochs} Loss is L ={loss.item()} "
        )

        loss_value.append(loss.item())

    torch.save(latent_classifier.state_dict(), "classifier_weight.pth")

    return loss_value


# here I just wanted to add a comment just to make sure to later on refactor the code
# and make it more modular and reusable, but I will keep it as is for now since I want to focus on the training process and not on the code structure


def train_contrasitve_model(epoch_count, dataLoader, weight_L):
    autoencoder = AutoEncoder().to(device)
    classifier = Classifier(16384, 8).to(device)

    optimizer_autoencoder = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    scheduler_autoencoder = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_autoencoder, gamma=0.955
    )
    scheduler_classifier = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_classifier, gamma=0.955
    )

    mse_loss = torch.nn.MSELoss()
    ce_loss = torch.nn.CrossEntropyLoss()

    loss_value = []

    for epoch in range(epoch_count):
        epoch_total = 0
        epoch_mse = 0
        epoch_ce = 0
        batch_count = 0

        for batch in dataLoader:
            images, labels = batch
            images = images.to(device)

            labels = convert_label_list(labels).to(device)

            output_autoencoder = autoencoder(images)

            encoded_images = autoencoder.encode_latent(images)
            encoded_images = torch.flatten(encoded_images, start_dim=1)

            output_classifier = classifier(encoded_images)

            loss_mse = mse_loss(output_autoencoder, images)
            loss_ce = ce_loss(output_classifier, labels)

            loss = weight_L * loss_mse + loss_ce

            optimizer_autoencoder.zero_grad()
            optimizer_classifier.zero_grad()

            loss.backward()

            optimizer_autoencoder.step()
            optimizer_classifier.step()

            epoch_total += loss.item()
            epoch_mse += loss_mse.item()
            epoch_ce += loss_ce.item()
            batch_count += 1

        scheduler_autoencoder.step()
        scheduler_classifier.step()

        loss_value.append(
            {
                "total": epoch_total / batch_count,
                "mse": epoch_mse / batch_count,
                "ce": epoch_ce / batch_count,
            }
        )

    return loss_value


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

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.955)
    num_epochs = epoch
    epochList = []

    for epoch in range(num_epochs):
        latent_model.train()

        for batch in dataLoader:
            images, labels = batch
            images = images.to(device)

            output = latent_model(images)
            loss = Loss_fn(output, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        epochList.append(loss.item())
        print(f"lr = {lr_rate} Loss {epoch + 1}/{num_epochs} Loss is L ={loss.item()} ")

    return latent_model, Loss_fn, optimizer, num_epochs, epochList


def test_model(classifier, latent_autoencoder, dataLoader_test):
    with torch.inference_mode():
        for batch in dataLoader_test:
            images, labels = batch
            images = images.to(device)
            labels = convert_label_list(labels).to(device)

            encoded_images = latent_autoencoder.encode_latent(images)
            encoded_images = torch.flatten(encoded_images, start_dim=1)

            output_classifier = classifier(encoded_images)

            predicted_label = torch.argmax(output_classifier, dim=1)
            real_label = torch.argmax(labels, dim=1)

            correct_ones += (predicted_label == real_label).sum().item()
            total += labels.size(0)

    accuarcy = correct_ones / total

    print(f"Correct: {correct_ones}/{total}")
    print(f"Accuracy: {accuarcy * 100:.2f}%")


def train_and_test_per_epoch(classifer, epoch_max, dataLoader, dataLoader_test):
    latent_model, Loss_fn, optimizer, num_epochs, epochList = train(
        150, 1e-3, dataLoader, Loss_fn="MSE", optimizer="Adam"
    )
    torch.save(latent_model.state_dict(), "model_weights.pth")
    test_accuracy = []
    train_accuracy = []
    test_loss = []
    train_loss = []
    loss_fn_classifer = torch.nn.CrossEntropyLoss()
    optimizer_classifier = torch.optim.Adam(classifer.parameters(), lr=1e-3)
    scheduler_classifier = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_classifier, gamma=0.955
    )

    for _ in range(epoch_max):
        batch_count = 0
        epoch_total = 0
        correct_ones = 0
        total = 0
        for batch in dataLoader:
            images, labels = batch
            images = images.to(device)
            labels = convert_label_list(labels).to(device)
            encoded_images = latent_model(images)
            output = classifer(encoded_images)
            Loss = loss_fn_classifer(output, labels)

            true_label = torch.argmax(labels, dim=1)
            predicted_label = torch.argmax(output, dim=1)
            correct_ones += (true_label == predicted_label).sum().item()
            total += labels.size(0)
            optimizer_classifier.zero_grad()
            Loss.backward()
            optimizer_classifier.step()
            epoch_total += Loss.item()
            batch_count += 1
        scheduler_classifier.step()

        train_accuracy.append(correct_ones / total)
        train_loss.append(epoch_total / batch_count)

        with torch.inference_mode():
            batch_count = 0
            epoch_total = 0
            correct_ones = 0
            total = 0
            for batch in dataLoader_test:
                images, labels = batch
                images = images.to(device)
                labels = convert_label_list(labels).to(device)
                encoded_images = latent_model(images)
                output = classifer(encoded_images)
                Loss = loss_fn_classifer(output, labels)
                true_label = torch.argmax(labels, dim=1)
                predicted_label = torch.argmax(output, dim=1)

                correct_ones += (predicted_label == true_label).sum().item()
                total += labels.size(0)
                epoch_total += Loss.item()
                batch_count += 1
            test_accuracy.append(correct_ones / total)
            test_loss.append(epoch_total / batch_count)

    return train_accuracy, test_accuracy, train_loss, test_loss
