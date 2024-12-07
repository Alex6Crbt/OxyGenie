#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:07:50 2024

@author: alexis
"""

from OxyGenie.diffusion import *
import OxyGenie.pgvnet as pgv
from OxyGenie.pgvnet import *

import matplotlib.pyplot as plt
import numpy as np
plt.style.use("bmh")

# plt.style.use("dark_background")
np.random.seed(42)
param = {
    "D": 1e-5, "k": 10, "Lx": 0.01, "Ly": 0.01, "T": 0.3, "nt": 8000, "nx": 250, "ny": 250,
    "initial_concentration": 100.0, "speed": 10, "step": 10,
}

# Initialisation des paramètres physiques et d'echantillonage
simparams = SimulationParams(**param)

# Simulation à partir du réseau vasculaire simulé :
Vnet = pgv.simple_generation()

L_result = run_simulation(simparams, FromPGVNet(Vnet[0]), C_0_cst=True)

# Simulation à partir d'une image :
# L_result = run_simulation(simparams, FromIMG("/Users/alexis/Downloads/IMG_38EDAE940454-1.jpeg"), C_0_cst=False)

# # Concentration initiale, coefficient de diffusion, coefficient d'absorbption custom
# C = np.zeros((simparams.nx,simparams.ny))
# C[120:130, 120:130] = 1

# X, Y = np.meshgrid(np.linspace(-1, 1, simparams.ny), np.linspace(-1, 1, simparams.nx))
# D = np.exp(-((X)**2+(Y-1)**2)/1.5)

# k = np.ones((simparams.nx, simparams.ny))

# # Simulation customisé :
# L_result = run_simulation(simparams, FromCustom(C, D, k), C_0_cst=True)


# Affichage

# Simu_plot.simple(simparams, L_result[-1], simparams.D_mat)

# Simu_plot.anim(simparams, L_result)




#%%


# plt.style.use("bmh")
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
n = 5
fig, axes = plt.subplots(1, n, figsize=(25, 5))

for i in range(n):
    np.random.seed(i)  # Modifier la graine aléatoire pour chaque image
    ngrid, nbranchs = sequence(grid.copy(), [(start_x, start_y)])  # Initialisation avec une liste de branches vide
    ax = axes[i]#[i // n, i % n]
    ax.imshow(run_simulation(simparams, FromPGVNet(ngrid), C_0_cst=True)[-1], cmap="Purples")
    # ax.axis("equal")
    ax.xaxis.set_visible(False)  # Cache uniquement l'axe x
    ax.yaxis.set_visible(False)  # Cache uniquement l'axe y
    # ax.set_title(f"Gen n°{i+1}, circularité = {sp_ratio(ngrid)*100:0.1f}%")

plt.tight_layout()
plt.show()

