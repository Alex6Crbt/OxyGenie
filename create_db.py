import OxyGenie.diffusion as simu
import OxyGenie.pgvnet as pgv
import numpy as np
import os

# np.random.seed(42)

# Dossiers de sauvegarde
output_dir = 'dataset2/'

# Créer les dossiers si nécessaire
os.makedirs(output_dir + 'X_1/', exist_ok=True)
os.makedirs(output_dir + 'X_2/', exist_ok=True)
os.makedirs(output_dir + 'Y/', exist_ok=True)

# Exemple de boucle pour générer et enregistrer les données
num_samples = 1000  # Nombre d'exemples à générer


for i in range(num_samples):

    # On choisi un k entre 1 et 10 aléatoirement, float
    k_random = 9 * np.random.rand() + 1

    T_simu_rp = -2 / 9 * k_random + 2.5 - 2 / 9

    nt_efficient = int(T_simu_rp * 5000)

    print(k_random, T_simu_rp, nt_efficient)

    params = {
        "D": 1e-5, "k": k_random, "Lx": 0.05, "Ly": 0.05, "T": T_simu_rp, "nt": nt_efficient, "nx": 256 * 2, "ny": 256 * 2,
        "initial_concentration": 100.0, "speed": 10, "step": 10,
    }
    simuparams = simu.SimulationParams(**params)

    # On genere un réseau de vascularisation aléatoire, np.array : (256, 256)
    V_net = pgv.simple_generation(grid_size=1280)[0]

    sp = pgv.sp_ratio(V_net)

    X_1 = V_net.copy()
    X_2 = np.array([k_random, sp])
    print(X_2)

    # On lance la simu
    L_result = simu.run_simulation(simuparams, simu.FromPGVNet(
        V_net), C_0_cst=True, save_last_only=True)

    Y = L_result[-1]  # Dernier résultat de simulation (état stationaire)

    # Sauvegarde
    np.save(f'{output_dir}X_1/img_{i:04d}.npy', X_1)
    np.save(f'{output_dir}X_2/img_{i:04d}.npy', X_2)
    np.save(f'{output_dir}Y/img_{i:04d}.npy', Y)

    print(f"Échantillon {i + 1} enregistré.")

print("Fin !")
