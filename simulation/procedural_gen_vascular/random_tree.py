#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 23:00:58 2024

@author: alexis
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line
from skimage.morphology import dilation
from tqdm import tqdm

# Classe pour la génération de branches
class BranchGen:
    def __init__(self, L_p, angle_p, n_branch, L_max=None):
        self.L_p = L_p  # Paramètres de longueur (moyenne, variation)
        self.angle_p = angle_p  # Paramètres d'angle (initial, écart-type)
        self.n_branch = n_branch  # Nombre de branches
        self.L_max = L_max  # Longueur maximale pour chaque branche
    
    def __call__(self, grid, branch):
        new_branchs = []
        grid_size_x, grid_size_y = grid.shape
        new_grid = grid.copy()
        L_mean, L_range = self.L_p
        angle_i, sigma_alpha = self.angle_p
        
        assert self.n_branch <= len(branch)
        
        L_i = np.random.choice(len(branch), size=self.n_branch, replace=True)
        for i in L_i:
            x_i, y_i = branch[i]
            new_branch = [(x_i, y_i)]
            angle = np.random.uniform(0, angle_i)
            L_tot = 0
            while True:
                L = L_mean + np.random.uniform(-L_range, L_range)
                angle += sigma_alpha * np.random.randn()

                x_f, y_f = int(x_i + L * np.sin(angle)), int(y_i - L * np.cos(angle))

                if x_f < 0 or y_f < 0 or x_f >= grid_size_x or y_f >= grid_size_y:
                    break

                rr, cc = line(y_i, x_i, y_f, x_f)
                new_grid[rr, cc] = 255//2
                new_branch.append((x_f, y_f))

                L_tot += L
                if self.L_max is not None and L_tot > self.L_max:
                    break

                x_i, y_i = x_f, y_f
            
            new_branchs.append(new_branch)
            
        all_new_branch = sum(new_branchs, [])
        
        return new_grid, all_new_branch
    def __repr__(self):

        return f"BranchGen(\n\t\t L_p = {self.L_p}, angle_p = ({self.angle_p[0]:0.2f}, {self.angle_p[1]:0.2f}),\n\t\t n_branch={self.n_branch}, L_max={self.L_max},\n\t)"


# Classe pour la dilatation
class DilationN:
    def __init__(self, n_iter=1):
        self.n_iter = n_iter  # Nombre d'itérations de dilatation
    
    def __call__(self, grid, branch):
        new_grid = grid.copy()
        for _ in range(self.n_iter):
            new_grid = dilation(new_grid)
            
        return new_grid, branch  # La dilatation n'affecte pas les branches
    def __repr__(self):
        return f"DilationN(n_iter={self.n_iter})"

class PGVNet:
    def __init__(self, n_iter):
        self.dilation = DilationN(n_iter)
    def __call__(self, grid, branch):
        # print(self.grid.shape)
        ngrid = np.ones(grid.shape)*255*(grid>2)
        M, _ = self.dilation(grid,[])
        m1 = M>grid   
        ngrid[m1]  = 255//2
        return ngrid, branch
    def __repr__(self):
        return f"PGVNet(n_iter={self.dilation.n_iter})"

# Classe Pipeline qui orchestre les opérations
class PGPipeline:
    def __init__(self, operations):
        self.operations = operations
    
    def __call__(self, grid, branch):
        for operation in self.operations:
            grid, branch = operation(grid, branch)
            
        return grid, branch
    def __repr__(self):
        operations_repr = ",\n    ".join(repr(op) for op in self.operations)
        return f"PGPipeline([\n    {operations_repr}\n])"

def sp_ratio(grid):
    ngrid = np.zeros(grid.shape)
    ngrid[grid>2] = 1
    dilngrid = DilationN(1)(ngrid,[])[0]
    perim = dilngrid[((dilngrid==1) & (ngrid==0))]
    surf = ngrid
    return 4*np.pi*np.sum(surf)/np.sum(perim)**2
    

def simple_generation(N=1, grid_size = 1000, Lc = 30, lrang = 10, initial_angle = 0, alpha = np.pi / 4):
    start_x, start_y = grid_size // 2, grid_size - 10  # Bas-centre
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    
    # Création de la séquence d'opérations
    sequence = PGPipeline([
        BranchGen((Lc, lrang), (initial_angle, alpha / 5), 1),
        DilationN(3),
        BranchGen((Lc, lrang), (2 * np.pi, alpha / 3), 4),
        DilationN(1),
        BranchGen((Lc, lrang), (2 * np.pi, alpha / 3), 8, 500),
        DilationN(1),
        BranchGen((Lc, lrang), (2 * np.pi, alpha / 2), 8, 100),
        DilationN(1),
        PGVNet(5),
    ])
    # Initialiser un tableau pour stocker les N images générées
    grids = np.empty((N, grid_size, grid_size), dtype=np.uint8)
    
    for i in tqdm(range(N)):
        # Générer une image avec le pipeline
        ngrid, _ = sequence(grid.copy(), [(start_x, start_y)])
        grids[i] = ngrid  # Stocker l'image générée dans le tableau

    return grids

if __name__=="__main__":
    plt.style.use("bmh")
    from matplotlib.colors import LinearSegmentedColormap
    # Définir les couleurs
    colors = ["white", "#4B0082", "#E6BEFF"]
    # Créer une colormap avec des transitions linéaires
    custom_cmap_lavande = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    
    
    # --- Paramètres globaux ---
    Lc = 30         # Longueur moyenne des segments
    lrang = 10      # Variation de longueur
    grid_size = 1000 # Taille de la grille
    alpha = np.pi / 4 # Amplitude de la variation angulaire
    start_x, start_y = grid_size // 2, grid_size - 10  # Bas-centre
    initial_angle = 0


    # --- Initialisation de la grille ---
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    
    # Création de la séquence d'opérations
    sequence = PGPipeline([
        BranchGen((Lc, lrang), (initial_angle, alpha / 5), 1),
        DilationN(3),
        BranchGen((Lc, lrang), (2 * np.pi, alpha / 3), 4),
        DilationN(1),
        BranchGen((Lc, lrang), (2 * np.pi, alpha / 3), 8, 500),
        DilationN(1),
        BranchGen((Lc, lrang), (2 * np.pi, alpha / 2), 8, 100),
        DilationN(1),
        PGVNet(5),
    ])
    
    # Exécution de la séquence
    # ngrid, nbranchs = sequence(grid, [(start_x, start_y)])
    # Affichage du résultat
    # plt.imshow(ngrid)
    # plt.show()
    
    # Générer n images différentes
    n = 3
    fig, axes = plt.subplots(n, n, figsize=(10, 10))
    
    for i in range(n**2):
        np.random.seed(i)  # Modifier la graine aléatoire pour chaque image
        ngrid, nbranchs = sequence(grid.copy(), [(start_x, start_y)])  # Initialisation avec une liste de branches vide
        ax = axes[i // n, i % n]
        ax.imshow(ngrid, cmap=custom_cmap_lavande)
        ax.axis("tight")
        ax.xaxis.set_visible(False)  # Cache uniquement l'axe x
        ax.yaxis.set_visible(False)  # Cache uniquement l'axe y
        ax.set_title(f"Gen n°{i+1}, circularité = {sp_ratio(ngrid)*100:0.1f}%")
    
    plt.tight_layout()
    plt.show()
