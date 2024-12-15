
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
import random
import torchvision.transforms.functional as TF


class SimuDataset(Dataset):
    """
    A PyTorch dataset class for loading and preprocessing simulation data.

    This dataset handles input features (`X_1`, `X_2`) and corresponding target outputs (`Y`),
    applying specified transformations and augmentations during loading.
    
    """
    
    def __init__(self, dataset_path, transform=None, random_flip=True):
        """
        x_dir : répertoire contenant les images (X/)
        y_dir : répertoire contenant les résultats (Y/)
        transform : transformations à appliquer aux images (par exemple, normalisation)
        """
        self.x1_dir = os.path.join(dataset_path, "X_1")
        self.x2_dir = os.path.join(dataset_path, "X_2")
        self.y_dir = os.path.join(dataset_path, "Y")
        # Liste des fichiers d'images
        self.x1_files = sorted(os.listdir(self.x1_dir))
        self.x2_files = sorted(os.listdir(self.x2_dir))
        # Liste des fichiers de résultats
        self.y_files = sorted(os.listdir(self.y_dir))
        # Transformations à appliquer (par exemple, normalisation)
        self.transform = transform
        self.random_flip = random_flip

    def __len__(self):
        """Retourne le nombre d'échantillons"""
        return len(self.x1_files)

    def __getitem__(self, idx):
        """
        Charge une image et son résultat associé
        idx : indice de l'échantillon
        """
        # Charger l'image et le résultat
        x1_path = os.path.join(self.x1_dir, self.x1_files[idx])
        x2_path = os.path.join(self.x2_dir, self.x2_files[idx])
        y_path = os.path.join(self.y_dir, self.y_files[idx])

        # Charger l'image (convertir en float32 si nécessaire)
        x1 = np.load(x1_path).astype(np.float32)
        x2 = np.load(x2_path).astype(np.float32)
        # Charger le résultat (en float32)
        y = np.load(y_path).astype(np.float32)

        # x2 = -np.log(x2)
        # y = np.log(y)
        # Trouver les valeurs min et max pour chaque image
        x_min, x_max = np.min(x1), np.max(x1)

        x2[0] = x2[0] / 10
        x2[1] = x2[1] * 100
        # y_min, y_max = np.min(y), np.max(y)
        # self.y_min = y_min
        # self.y_max = y_max

        # Appliquer la normalisation Min-Max
        x1 = (x1 - x_min) / (x_max - x_min)
        # y = (y - y_min) / (y_max - y_min)
        y = y / 100
        # Convertir en image PIL après normalisation
        x1 = Image.fromarray(x1)
        x1 = x1.resize((512, 512))
        # x2 = TF.to_tensor(x2)
        y = Image.fromarray(y)

        # Ajouter des dimensions supplémentaires si nécessaire (par exemple, pour les images en niveaux de gris)
        # Si l'image est 2D, on ajoute une dimension supplémentaire pour représenter un canal (par ex. (H, W, 1))
        # if x.ndim == 2:  # Si l'image est en niveaux de gris
        # x = np.expand_dims(x, axis=-1)
        # y = np.expand_dims(y, axis=-1)

        # Convertir en tenseurs PyTorch
        # x = torch.tensor(x).permute(2, 0, 1)  # Permuter pour que l'ordre des dimensions soit (C, H, W)
        # y = torch.tensor(y).permute(2, 0, 1)

        # Appliquer les transformations si spécifié

        if self.random_flip:
            x1, y = self.hvflip(x1, y)
        else:
            x1 = TF.to_tensor(x1)
            y = TF.to_tensor(y)

        if self.transform:
            x1 = self.transform(x1)
            y = self.transform(y)

        return (x1, x2), y

    def hvflip(self, x, y):
        """
        Applies random horizontal and vertical flips to inputs and targets.
        
        """
        # Random horizontal flipping
        if random.random() > 0.5:
            x = TF.hflip(x)
            y = TF.hflip(y)

        # Random vertical flipping
        if random.random() > 0.5:
            x = TF.vflip(x)
            y = TF.vflip(y)

        # Transform to tensor
        x = TF.to_tensor(x)
        y = TF.to_tensor(y)

        return x, y

    def descale(self, y_scaled):
        """
        Converts normalized target data back to its original range.
        """
        # Vérifiez que les valeurs min et max existent
        if not hasattr(self, 'y_min') or not hasattr(self, 'y_max'):
            raise ValueError(
                "Les valeurs 'y_min' et 'y_max' doivent être définies pour descale.")

        # Calcul de la désnormalisation
        y_original = y_scaled * (self.y_max - self.y_min) + self.y_min
        return y_original


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = SimuDataset("dataset2", random_flip=True)
    (X1, X2), Y = dataset[1]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    print(X2)

    ax[0].imshow(X1[0])
    ax[1].imshow(Y[0])

    plt.show()
    plt.figure()
    plt.hist(Y[0].numpy().flatten(), bins=200)
    plt.show()
