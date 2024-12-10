import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import OxyGenie.diffusion as simu
import OxyGenie.pgvnet as pgv
from OxyGenie.learn import EUNet
# %%

model = EUNet()

model.load_state_dict(torch.load("model2_weights_6E.pth"))

# %%
# with torch.inference_mode()
# from matplotlib.colors import LogNorm, SymLogNorm


n = 5
lk = np.linspace(1, 10, n)
fig, ax = plt.subplots(n, n, figsize=(12, 12))

for j in tqdm(range(n)):
    mat = np.zeros((512, 512))
    step = 40 * (j + 1)
    mat[:, ::step] = 255
    mat[::step, :] = 255

    for i in range(n):
        params = [float(lk[i]), 5e-3]
        out = model.predict(mat.astype(np.float32), params)
        ax[i, j].imshow(out, vmin=0, vmax=120)  # , norm= LogNorm())
        ax[i, j].axis("off")
        # ax[i,j].set_title(f"params={params}")
plt.show()

# %%


k_random = 10  # 9*np.random.rand()+1 # On choisi un k entre 1 et 10 aléatoirement, float

T_simu_rp = -2 / 9 * k_random + 2.5 - 2 / 9

nt_efficient = int(T_simu_rp * 5000)

print(k_random, T_simu_rp, nt_efficient)

params = {
    "D": 1e-5, "k": k_random, "Lx": 0.05, "Ly": 0.05, "T": 0.25, "nt": 2500, "nx": 256 * 2, "ny": 256 * 2,
    "initial_concentration": 100.0, "speed": 1, "step": 10,
}
simuparams = simu.SimulationParams(**params)

# On genere un réseau de vascularisation aléatoire, np.array : (256, 256)
V_net = pgv.simple_generation(grid_size=1280)[0]

sp = pgv.sp_ratio(V_net)

X_1 = V_net.copy()
X_2 = [float(k_random), float(sp)]
print(X_2)

# plt.imshow(X_1)
# plt.show()

out = model.predict(V_net.astype(np.float32), X_2)
out[out > 100] = 100

# On lance la simu
L_result = simu.run_simulation(simuparams, simu.FromPGVNet(
    V_net), C_0_cst=True, save_last_only=False, C0=out)

Y = L_result[-1]  # Dernier résultat de simulation (état stationaire)

f, a = plt.subplots(1, 2)
a[0].imshow(out)
a[1].imshow(Y)
plt.show()


simu.Simu_plot.simple(simuparams, L_result[-1], simuparams.D_mat)

simu.Simu_plot.anim(simuparams, L_result)
