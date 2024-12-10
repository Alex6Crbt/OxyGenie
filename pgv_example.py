import OxyGenie.pgvnet as pgv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.style.use("bmh")

# Définir les couleurs
colors = ["white", "#4B0082", "#E6BEFF"]
# Créer une colormap avec des transitions linéaires
custom_cmap_lavande = LinearSegmentedColormap.from_list(
    "custom_cmap", colors, N=256)


# Paramètres globaux
Lc = 30         # Longueur moyenne des segments
lrang = 10      # Variation de longueur
grid_size = 1000  # Taille de la grille
alpha = np.pi / 4  # Amplitude de la variation angulaire
start_x, start_y = grid_size // 2, grid_size - 10  # Bas-centre
initial_angle = 0


# Initialisation de la grille
grid = np.zeros((grid_size, grid_size), dtype=np.uint8)

# Création de la séquence d'opérations
sequence = pgv.PGPipeline([
    pgv.BranchGen((Lc, lrang), (initial_angle, alpha / 5), 1),
    pgv.DilationN(3),
    pgv.BranchGen((Lc, lrang), (2 * np.pi, alpha / 3), 4),
    pgv.DilationN(1),
    pgv.BranchGen((Lc, lrang), (2 * np.pi, alpha / 3), 8, 500),
    pgv.DilationN(1),
    pgv.BranchGen((Lc, lrang), (2 * np.pi, alpha / 2), 8, 100),
    pgv.DilationN(1),
    pgv.PGVNet(5),
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
    # Initialisation avec une liste de branches vide
    ngrid, nbranchs = sequence(grid.copy(), [(start_x, start_y)])
    ax = axes[i]  # [i // n, i % n]
    # ax.imshow(run_simulation(simparams, FromPGVNet(
    #     ngrid), C_0_cst=True)[-1], cmap="Purples")
    # ax.axis("equal")
    ax.imshow(ngrid, cmap="Purples")
    ax.xaxis.set_visible(False)  # Cache uniquement l'axe x
    ax.yaxis.set_visible(False)  # Cache uniquement l'axe y
    ax.set_title(f"Gen n°{i+1}, circularité = {pgv.sp_ratio(ngrid)*100:0.1f}%")

plt.tight_layout()
plt.show()
