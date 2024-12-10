from OxyGenie.diffusion import *
import OxyGenie.pgvnet as pgv
# from OxyGenie.pgvnet import *

import matplotlib.pyplot as plt
import numpy as np
plt.style.use("bmh")

# plt.style.use("dark_background")
np.random.seed(42)

params = {
    "D": 1e-5, "k": 8, "Lx": 0.05, "Ly": 0.05, "T": 0.5, "nt": 2500, "nx": 256 * 2, "ny": 256 * 2,
    "initial_concentration": 100.0, "speed": 1, "step": 10,
}

# Initialisation des paramètres physiques et d'echantillonage
simparams = SimulationParams(**params)

# Simulation à partir du réseau vasculaire simulé :
Vnet = pgv.simple_generation(grid_size=1280)

L_result = run_simulation(simparams, FromPGVNet(Vnet[0]), C_0_cst=True)

# Simulation à partir d'une image :
# L_result = run_simulation(simparams, FromIMG("/Users/alexis/Downloads/IMG_38EDAE940454-1.jpeg"), C_0_cst=False)


# # Simulation customisé :

# # Concentration initiale, coefficient de diffusion, coefficient d'absorbption custom
# C = np.zeros((simparams.nx,simparams.ny))
# C[120:130, 120:130] = 1

# X, Y = np.meshgrid(np.linspace(-1, 1, simparams.ny), np.linspace(-1, 1, simparams.nx))
# D = np.exp(-((X)**2+(Y-1)**2)/1.5)

# k = np.ones((simparams.nx, simparams.ny))

# L_result = run_simulation(simparams, FromCustom(C, D, k), C_0_cst=True)


# Affichage

Simu_plot.simple(simparams, L_result[-1], simparams.D_mat)

Simu_plot.anim(simparams, L_result)
