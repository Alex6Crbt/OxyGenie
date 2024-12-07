import OxyGenie.diffusion as simu
import OxyGenie.pgvnet as pgv
import numpy as np
import os

# np.random.seed(42)

param = {
    "D": 1e-5, "k": 10, "Lx": 0.01, "Ly": 0.01, "T": 0.3, "nt": 8000, "nx": 256, "ny": 256,
    "initial_concentration": 100.0, "speed": 10, "step": 10,
}


# Dossiers de sauvegarde
output_dir = 'dataset/'

# Créer les dossiers si nécessaire
os.makedirs(output_dir + 'X/', exist_ok=True)
os.makedirs(output_dir + 'Y/', exist_ok=True)

num_samples = 800 # Nombre d'exemples à générer

for i in range(0, num_samples):
    # Génération de l'entrée Vnet et des paramètres de simulation
    Vnet = pgv.simple_generation()
    simparams = simu.SimulationParams(**param)
    
    # simulation
    L_result = simu.run_simulation(simparams, simu.FromPGVNet(Vnet[0]), C_0_cst=True, save_last_only=True)
    
    # Extraction des données
    image = simparams.D_mat  # Carte de diffusion
    result = L_result[-1]  # Dernier résultat de simulation
    # physical_vector = simparams.get_parameters()  # Exemple de récupération des paramètres (adaptez selon votre code)

    # Sauvegarder l'image
    np.save(f'{output_dir}X/img_{i:03d}.npy', image)

    # Sauvegarder les paramètres physiques associés
    # np.save(f'{output_dir}X/params_{i:03d}.npy', physical_vector)

    # Sauvegarder le résultat de simulation
    np.save(f'{output_dir}Y/result_{i:03d}.npy', result)

    print(f"Échantillon {i + 1} enregistré.")