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


# here I just wanted to add a comment just to make sure to later on refactor the code
# and make it more modular and reusable, but I will keep it as is for now since I want to focus on the training process and not on the code structure


def train_contrasitve_model(epoch_count, dataLoader, weight_L, dataLoader_test):
    autoencoder = AutoEncoder().to(device)
    classifier = Classifier(2048, 8).to(device)

    optimizer_autoencoder = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    optimizer_classifier = torch.optim.AdamW(
        classifier.parameters(), lr=1e-3, weight_decay=1e-3
    )

    scheduler_autoencoder = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_autoencoder, gamma=0.955
    )
    scheduler_classifier = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_classifier, gamma=0.955
    )

    mse_loss = torch.nn.MSELoss()
    ce_loss = torch.nn.CrossEntropyLoss()

    loss_value = []
    train_accuracy = []
    test_accuracy = []

    for epoch in range(epoch_count):
        epoch_total = 0
        epoch_mse = 0
        epoch_ce = 0
        batch_count = 0
        correct_ones = 0
        total = 0
        autoencoder.train()
        classifier.train()
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
            true_label = torch.argmax(labels, dim=1)
            predicted_label = torch.argmax(output_classifier, dim=1)
            correct_ones += (true_label == predicted_label).sum().item()
            batch_count += 1
            total += labels.size(0)
        train_accuracy.append(correct_ones / total)
        scheduler_autoencoder.step()
        scheduler_classifier.step()

        # here the infrence begins
        epoch_total = 0
        epoch_mse = 0
        epoch_ce = 0
        batch_count = 0
        correct_ones = 0
        total = 0
        autoencoder.eval()
        classifier.eval()
        with torch.inference_mode():
            for batch in dataLoader_test:
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

                epoch_total += loss.item()
                epoch_mse += loss_mse.item()
                epoch_ce += loss_ce.item()
                true_label = torch.argmax(labels, dim=1)
                predicted_label = torch.argmax(output_classifier, dim=1)
                correct_ones += (true_label == predicted_label).sum().item()
                batch_count += 1
                total += labels.size(0)
            test_accuracy.append(correct_ones / total)

    return train_accuracy, test_accuracy


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
    optimizer_classifier = torch.optim.AdamW(
        classifer.parameters(), lr=1e-3, weight_decay=1e-3
    )
    scheduler_classifier = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_classifier, gamma=0.955
    )

    for _ in range(epoch_max):
        classifer.train()
        batch_count = 0
        epoch_total = 0
        correct_ones = 0
        total = 0
        for batch in dataLoader:
            images, labels = batch
            images = images.to(device)
            labels = convert_label_list(labels).to(device)
            encoded_images = latent_model.encode_latent(images)
            encoded_images = torch.flatten(encoded_images, start_dim=1)
            print(encoded_images.shape)
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
        classifer.eval()
        with torch.inference_mode():
            batch_count = 0
            epoch_total = 0
            correct_ones = 0
            total = 0
            for batch in dataLoader_test:
                images, labels = batch
                images = images.to(device)
                labels = convert_label_list(labels).to(device)
                encoded_images = latent_model.encode_latent(images)
                encoded_images = torch.flatten(encoded_images, start_dim=1)
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
