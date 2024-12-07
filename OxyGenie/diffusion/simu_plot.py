import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from tqdm import tqdm
from .simu_func import N_tot


class Simu_plot:
    def __init__(self):
        plt.style.use("dark_background")
    @classmethod
    def simple(self, params, C_result, D=None):
        if D is None:
            fig, ax1 = plt.subplots()
            im = ax1.imshow(C_result, extent=(0, params.Lx, 0, params.Ly), origin='lower', cmap='hot')
            plt.colorbar(im, ax=ax1, label="Concentration")
            ax1.set_xlabel("$x \\ (m)$")
            ax1.set_ylabel("$y \\ (m)$")
            ax1.set_title(f"Concentration finale après ${params.T:.1e} \\ s$")
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            im = ax1.imshow(C_result, extent=(0, params.Lx, 0, params.Ly), origin='lower', cmap='hot')
            plt.colorbar(im, ax=ax1, label="Concentration")
            ax1.set_xlabel("$x \\ (m)$")
            ax1.set_ylabel("$y \\ (m)$")
            ax1.set_title(f"Concentration finale après ${params.T:.1e} \\ s$")
            im2 = ax2.imshow(D, extent=(0, params.Lx, 0, params.Ly), origin='lower', cmap="coolwarm",norm=LogNorm())
            plt.colorbar(im2, ax=ax2, label="Coefficient de diffusion")
            ax2.set_xlabel("$x \\ (m)$")
            ax2.set_ylabel("$y \\ (m)$")
            ax2.set_title("Coefficient de diffusion")
            plt.show()
    
    @classmethod
    def champ_vect(self, params, L_result, i, s_ech = 5):
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(L_result[i],  extent=(0, params.Lx, 0, params.Ly), origin='lower', cmap='viridis')  # Carte de concentration
        fig.colorbar(im, ax=ax, label="Concentration")
        
        # Calcul des gradients pour la frame courante
        if i<0:
            dC_dy, dC_dx = np.gradient(-L_result[i-1] + L_result[i])  # Gradients temporels
            plt.title(f"Champ vectoriel des flux de concentration à T = {(params.nt-i)*params.dt:0.1e}s")
        else:
            dC_dy, dC_dx = np.gradient(-L_result[i] + L_result[i+1])  # Gradients temporels
            plt.title(f"Champ vectoriel des flux de concentration à T = {i*params.dt:0.1e}s")

        
        # Norme des vecteurs (éviter division par zéro avec 1e-10)
        # c = np.max(dC_dx**2 + dC_dy**2)
        dC_dx = dC_dx/(params.dx*params.dt)
        dC_dy = dC_dy/(params.dy*params.dt)

        c = np.sqrt((dC_dx)**2 + (dC_dy)**2)+1e-10
        
        X, Y = np.linspace(0, params.Lx, params.nx), np.linspace(0, params.Ly, params.ny)
        X, Y = np.meshgrid(X, Y)
        # Sous-échantillonnage des résultats   
        if s_ech==1:
            X, Y, pdC_dx, pdC_dy, pc = X[::s_ech, ::s_ech], Y[::s_ech, ::s_ech], dC_dx[::s_ech, ::s_ech], dC_dy[::s_ech, ::s_ech], c[::s_ech, ::s_ech]
            q = ax.quiver(X, Y, pdC_dx, pdC_dy, pc, cmap='plasma')#,   scale=1e-0/(2*Lx),)

        else:
            X, Y, pdC_dx, pdC_dy, pc = X[::s_ech, ::s_ech], Y[::s_ech, ::s_ech], dC_dx[::s_ech, ::s_ech]/c[::s_ech, ::s_ech], dC_dy[::s_ech, ::s_ech]/c[::s_ech, ::s_ech], c[::s_ech, ::s_ech]
            q = ax.quiver(X, Y, pdC_dx, pdC_dy, pc, cmap='plasma',   scale=1e-0/(2*params.Lx),)

        
        fig.colorbar(q, ax=ax, label="Flux de concentration")

        plt.xlabel("$x \\ (m)$")
        plt.ylabel("$y \\ (m)$")
        plt.show()
        
    @classmethod
    def anim(self, params, L_result):
        speed = params.speed
        S = np.empty(len(L_result))  # Allocation d'un tableau pour tous les éléments
        for i in tqdm(range(len(L_result))):
            S[i] = N_tot(L_result[i], params.dx, params.dy)

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        (ax1, ax2) , (ax3, ax4) = axs

        im = ax1.imshow(L_result[0], extent=(0, params.Lx, 0, params.Ly), origin='lower', cmap='hot')#, norm=LogNorm())
        plt.colorbar(im, ax=ax1, label="Concentration")

        ax1.set_xlabel("$x \\ (m)$")
        ax1.set_ylabel("$y \\ (m)$")
        ax1.set_title(f"Concentration finale après ${params.T:0.1e} \\ s$")

        line1, = ax2.plot(S)
        ax2.set_xlim(0, params.nt)
        ax2.set_ylim(min(S)*0.99, max(S)*1.01)
        ax2.set_xlabel("$n$ itérations")
        ax2.set_ylabel("$N_{tot}$")
        ax2.set_title("$N_{tot}$ en fonction de l'itération")
        ax2.grid(True, color='gray', linestyle='--', linewidth=0.5)


        xc = L_result[0][params.nx//2, :]
        line2, = ax3.plot([k*params.dx for k in range(params.nx)], xc,'r', linestyle="--", marker="2", label="X")
        yc = L_result[0][:, params.ny//2]
        line3, = ax3.plot([k*params.dy for k in range(params.ny)], yc, "y", linestyle="--", marker="2", label="Y")

        ax3.set_xlim(0, params.Lx)
        # ax3.set_ylim(min(xc)*0.99, max(xc)*1.01)
        ax3.set_xlabel("$x \\ (m)$")
        ax3.set_ylabel("$C$")
        ax3.set_title("coupe de $C$ en x et y")
        ax3.grid(True, color='gray', linestyle='--', linewidth=0.5)
        ax3.legend()

        # Initialisation des listes pour stocker les résultats
        Ltc = np.empty(params.nt//params.step)
        Ltfx = np.empty(params.nt//params.step)
        Ltfy = np.empty(params.nt//params.step)
        
        # Itération unique pour remplir les trois listes
        for k in range(params.nt//params.step):
            frame = L_result[k]
            Ltc[k] = frame[params.nx // 2, params.ny // 2]
            Ltfx[k] = frame[params.nx // 2, 0]
            Ltfy[k] = frame[0, params.ny // 2]
            

        # Ltc = [L_result[k][params.nx//2, params.ny//2] for k in range(params.nt)]
        # Ltfx = [L_result[k][params.nx//2, 0] for k in range(params.nt)]
        # Ltfy = [L_result[k][0, params.ny//2] for k in range(params.nt)]

        line4, = ax4.plot([k*params.dt*params.step for k in range(params.nt//params.step)], Ltc, label="Centre")
        line5, = ax4.plot([k*params.dt*params.step for k in range(params.nt//params.step)], Ltfx, label="Bord X Gauche")
        line6, = ax4.plot([k*params.dt*params.step for k in range(params.nt//params.step)], Ltfy, label="Bord Y Bas")

        # ax4.set_ylim(min(Ltc)*0.99, max(Ltc)*1.01)
        ax4.set_xlabel("$t \\ (s)$")
        ax4.set_ylabel("$C$")
        ax4.set_title("Evolution temporelle")
        ax4.grid(True, color='gray', linestyle='--', linewidth=0.5)
        ax4.legend()

        # Fonction d'animation
        def update(frame):
            # print(len(Ltc[:speed*frame]), len([k*params.dt*params.step for k in range(speed*frame)]))
            # print(frame, len(L_result)//speed)
            """Met à jour l'image pour chaque étape de l'animation."""
            im.set_array(L_result[speed*frame])
            line1.set_data([k*params.step for k in range(speed*frame+1)], S[:speed*frame + 1])
            line2.set_data([k*params.dx for k in range(params.nx)], L_result[speed*frame][params.nx//2, :])
            line3.set_data([k*params.dy for k in range(params.ny)], L_result[speed*frame][:, params.ny//2])
            line4.set_data([k*params.dt*params.step for k in range(speed*frame)], Ltc[:speed*frame])
            line5.set_data([k*params.dt*params.step for k in range(speed*frame)], Ltfx[:speed*frame])
            line6.set_data([k*params.dt*params.step for k in range(speed*frame)], Ltfy[:speed*frame])
            return [im, line1, line2, line3, line4, line5, line6,]

        # Création de l'animation avec `FuncAnimation`
        ani = FuncAnimation(fig, update, interval=5, frames=len(L_result)//speed, blit=True, repeat=True)

        # Afficher l'animation
        plt.show()
        return ani
    
    @classmethod
    def anim_vect(self, params, L_result, s_ech = 5):
        speed, nt, nx, ny = params.speed, params.nt, params.nx, params.ny

        fig, ax1 = plt.subplots(figsize=(10,10))
        # ax3 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
        im = ax1.imshow(L_result[len(L_result)//2], extent=(0, params.Lx, 0, params.Ly), origin='lower', cmap='viridis')#, norm=LogNorm())
        fig.colorbar(im, ax=ax1, label="Concentration")

        # Initialisation du champ vectoriel
        dC_dy, dC_dx = np.gradient(-L_result[-(nt//params.step)//4-1]+L_result[-(nt//params.step)//2])
        c = np.sqrt(dC_dx**2+dC_dy**2)+1e-10

        s=s_ech

        X, Y = np.linspace(0, params.Lx, nx), np.linspace(0, params.Ly, ny)
        X, Y = np.meshgrid(X, Y)
        X, Y, pdC_dx, pdC_dy, pc = X[::s, ::s], Y[::s, ::s], dC_dx[::s, ::s], dC_dy[::s, ::s], c[::s, ::s]

        quiver = ax1.quiver(X, Y, pdC_dx/pc, pdC_dy/pc, pc, cmap='plasma', scale=1e-0/(2*params.Lx),)
        ax1.set_xlabel("$x \\ (m)$")
        ax1.set_ylabel("$y \\ (m)$")
        ax1.set_title(f"Concentration finale après ${params.T:0.1e} \\ s$")

        # Préallocation des tableaux pour optimiser la mémoire
        anidc = np.empty((nt, *pdC_dx.shape, 2))  # Stocke (pdC_dx, pdC_dy) pour chaque frame
        anic = np.empty((nt, *pdC_dx.shape))      # Stocke pc pour chaque frame

        # Calcul des gradients et stockage optimisé
        for i in tqdm(range(nt//params.step)):
            # if i%5==0:
            
            # Calcul des gradients pour la frame courante
            dC_dy, dC_dx = np.gradient(-L_result[i] + L_result[i+1])  # Gradients complets

            # Norme des vecteurs (éviter division par zéro avec 1e-10)
            c = np.max(dC_dx**2 + dC_dy**2)
            cc = np.sqrt(dC_dx**2 + dC_dy**2)+1e-10
            # Sous-échantillonnage des résultats
            pdC_dx, pdC_dy, pc = dC_dx[::s, ::s], dC_dy[::s, ::s], cc[::s, ::s]
            # pdC_dx, pdC_dy, pc = dC_dx[::s, ::s], dC_dy[::s, ::s], cc[]

            # Stockage dans les tableaux préalloués
            anidc[i, :, :, 0] = pdC_dx/pc#np.sign(pdC_dx)*np.log(1+np.abs(pdC_dx/c))
            anidc[i, :, :, 1] = pdC_dy/pc#np.sign(pdC_dy)*np.log(1+np.abs(pdC_dy/c))
            anic[i] = pc



        # Fonction d'animation
        def update(frame):
            """Met à jour l'image pour chaque étape de l'animation."""
            im.set_array(L_result[speed*frame])
            dC_dy, dC_dx = anidc[speed * frame, :, :, 1], anidc[speed * frame, :, :, 0]
            pc = anic[speed*frame]
            # im.set_array((dC_dy**2 + dC_dx**2)/c)
            quiver.set_UVC(dC_dx, dC_dy, 10*pc/np.max(pc))  # Met à jour les données du quiver
            # colorbar.update_normal(im) 
            return [im, quiver]

        ani = FuncAnimation(fig, update, interval=5, frames=len(L_result)//speed, blit=True, repeat=True)

        plt.show()
        return ani