import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from OxyGenie.learn import EUNet, SimuDataset

# Charger le dataset complet
dataset = SimuDataset("dataset2")  # , transform)

# Définir les tailles pour train/test
train_size = int(0.8 * len(dataset))  # 80% pour l'entraînement
test_size = len(dataset) - train_size  # 20% pour le test

# Split des données
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Création des DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# batch_size = 32

# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Créer l'instance du modèle
model = EUNet()
# summary(model, input_size=(16, 1, 256*2, 256*2))

# Définir une fonction de perte et un optimiseur
criterion = nn.MSELoss()  # Perte MSE pour la reconstruction d'image
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Entraînement
num_epochs = 6
for epoch in range(num_epochs):
    # Entraînement
    model.train()
    running_train_loss = 0.0
    with tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train", unit="batch") as t:
        for i, (inputs, targets) in enumerate(t):
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs[0], inputs[1])

            # Calcul de la perte
            loss = criterion(outputs, targets)

            # Backward pass et mise à jour des poids
            loss.backward()
            optimizer.step()

            # Mise à jour des statistiques
            b_loss = loss.item()
            running_train_loss += b_loss
            avg_train_loss = running_train_loss / (i + 1)
            t.set_postfix({"Train Loss": avg_train_loss, "Batch Loss": b_loss})

    # Evaluation
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        with tqdm(test_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Test", unit="batch") as t:
            for i, (inputs, targets) in enumerate(t):
                # Forward pass uniquement
                outputs = model(inputs[0], inputs[1])

                # Calcul de la perte
                loss = criterion(outputs, targets)

                # Mise à jour des statistiques
                running_test_loss += loss.item()
                avg_test_loss = running_test_loss / (i + 1)
                t.set_postfix({"Test Loss": avg_test_loss})

    # Afficher un résumé de l'epoch
    print(
        f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")


torch.save(model.state_dict(), "model_weights_6E.pth")
