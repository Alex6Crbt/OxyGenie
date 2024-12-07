
from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import torchvision.transforms.functional as TF


class ImageDataset(Dataset):
    def __init__(self, dataset_path, transform=None, random_flip=True):
        self.x_dir = os.path.join(dataset_path, "X")
        self.y_dir = os.path.join(dataset_path, "Y")
        # Liste des fichiers d'images
        self.x_files = sorted(os.listdir(self.x_dir))
        # Liste des fichiers de résultats
        self.y_files = sorted(os.listdir(self.y_dir))
        # Transformations à appliquer (par exemple, normalisation)
        self.transform = transform
        self.random_flip = random_flip

    def __len__(self):
        """Retourne le nombre d'échantillons"""
        return len(self.x_files)

    def __getitem__(self, idx):
        """
        Charge une image et son résultat associé
        idx : indice de l'échantillon
        """
        # Charger l'image et le résultat
        x_path = os.path.join(self.x_dir, self.x_files[idx])
        y_path = os.path.join(self.y_dir, self.y_files[idx])

        # Charger l'image (convertir en float32 si nécessaire)
        x = np.load(x_path).astype(np.float32)
        # Charger le résultat (en float32)
        y = np.load(y_path).astype(np.float32)

        x = -np.log(x)
        # y = np.log(y)
        # Trouver les valeurs min et max pour chaque image
        x_min, x_max = np.min(x), np.max(x)
        # y_min, y_max = np.min(y), np.max(y)
        self.x_min = x_min
        self.x_max = x_max
        # self.y_min = y_min
        # self.y_max = y_max

        # Appliquer la normalisation Min-Max
        x = (x - x_min) / (x_max - x_min)
        # y = (y - y_min) / (y_max - y_min)
        y = y / 100
        # Convertir en image PIL après normalisation
        x = Image.fromarray(x)
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
            x, y = self.hvflip(x, y)
        else:
            x = TF.to_tensor(x)
            y = TF.to_tensor(y)

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y

    def hvflip(self, x, y):
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

    def descale(self, x_scaled):
        """
        Dé-normalise les données mises à l'échelle entre [0, 1] à leur plage d'origine.
        x_scaled : Tensor ou ndarray normalisé (entre 0 et 1)
        """
        # Vérifiez que les valeurs min et max existent
        if not hasattr(self, 'x_min') or not hasattr(self, 'x_max'):
            raise ValueError(
                "Les valeurs 'x_min' et 'x_max' doivent être définies pour descale.")

        # Calcul de la dé-normalisation
        x_original = np.exp(x_scaled * (self.x_max - self.x_min) + self.x_min)
        return x_original
