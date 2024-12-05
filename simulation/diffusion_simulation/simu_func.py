import numpy as np
from PIL import Image
from tqdm import tqdm

def crit_stab(alphax, alphay):
    stab_coef = np.max(alphax) + np.max(alphay)
    if stab_coef >= 0.5:
        raise ValueError(f"Critère de stabilité non respecté (>0.5): {stab_coef:.2e}")
    else:
        print(f"Critère de stabilité respecté (<0.5): {stab_coef:.2e}")

def N_tot(C, dx, dy):
    l = 1
    Na = 6.02214076e23
    return np.sum(C[l:-l,l:-l])/(dx*dy*Na)

def init_simulation(params, vasc_path = "/Users/alexis/Downloads/IMG_38EDAE940454-1.jpeg"):
    
    if vasc_path is not None:
        # Initialisation de la concentration en O2
        # Im = Image.open("/Users/alexis/Downloads/IMG_0495.jpg")
        nx, ny = params.nx, params.ny
        Im = Image.open(vasc_path)
        image_reduite = Im.resize((nx, ny))
        M = np.array(image_reduite)
        m1 = M[:,:,0]<125 # concentration dans les vaissaux
        m2 = M[:,:,2]<125 # concentration dans les vaissaux + znd
        m3 = M[:,:,2]<125
        params.m1 = m1
        m3[m1==1] = 0 # concentration dans les znd
        C = params.initial_concentration*np.ones((nx, ny))*m1
        # Initialisation du coefficient de diffusion : 
        # D = params.D/100*np.ones((nx, ny))*m1+params.D*np.ones((nx, ny))*~m2+params.D/1000*np.ones((nx, ny))*m2
        D = params.D*np.ones((nx, ny))*m1+params.D/100*np.ones((nx, ny))*~m2+params.D/1000*np.ones((nx, ny))*m2
        params.D_mat = D
        dDx = np.zeros((nx, ny))
        dDx[1:-1,1:-1] = D[2:,1:-1]-D[1:-1,1:-1] # Attention claculer la dérivé de D
        dDy = np.zeros((nx, ny))
        dDy[1:-1,1:-1] = D[1:-1, 2:]-D[1:-1,1:-1]  # Attention claculer la dérivé de D
        # Initialisation de l'absobtion/production en O2
        k = np.zeros((nx, ny))
        k[~m2] = params.k/np.sum(~m2)
        k[m3] = params.k/np.sum(m3)*10
        k[m1] = -params.k/(np.sum(m1)+np.sum(m3)/10)
        params.k_mat = k
        # Initialisation des préfacteur de l'équation de diffusion discrète
        alphax = D*params.dt/(params.dx)**2
        alphay = D*params.dt/(params.dy)**2
        betax = dDx*params.dt/(params.dx)**2
        betay = dDy*params.dt/(params.dy)**2
        return C, D, k, alphax, alphay, betax, betay

    else:
        # Initialisation de la concentration en O2
        center_x, center_y = params.nx // 2, params.ny // 2
        size = params.initial_source_size
        concentration = params.initial_concentration
        C = np.zeros((params.nx,params.ny))
        C[center_x - size:center_x + size, center_y - size:center_y + size] = concentration
        # Initialisation du coefficient de diffusion : 
        X, Y = np.meshgrid(np.linspace(-1, 1, params.ny), np.linspace(0, 2, params.nx))
        D2 = np.exp(-(X**2+Y**2))
        D = params.D*D2
        params.D_mat = D
        dDx = np.zeros((nx, ny))
        dDx[1:-1,1:-1] = D[2:,1:-1]-D[1:-1,1:-1] # Attention claculer la dérivé de D
        dDy = np.zeros((nx, ny))
        dDy[1:-1,1:-1] = D[1:-1, 2:]-D[1:-1,1:-1]  # Attention claculer la dérivé de D
        # Initialisation de l'absobtion/production en O2
        k = np.zeros((params.nx, params.ny))
        k[params.nx//2-5:params.nx//2+5, params.ny//2-10:params.ny//2-5] = params.k
        k[params.nx//2, params.ny//2+10] = -params.k
        params.k_mat = k
        # Initialisation des préfacteur de l'équation de diffusion discrète
        alphax = D*params.dt/(params.dx)**2
        alphay = D*params.dt/(params.dy)**2
        betax = dDx*params.dt/(params.dx)**2
        betay = dDy*params.dt/(params.dy)**2
        return C, D, k, alphax, alphay, betax, betay

def F(C, k, cax, cay, cbx, cby):
    return -k[1:-1, 1:-1]*C[1:-1, 1:-1] \
        + cbx*(C[2:, 1:-1]-C[1:-1, 1:-1]) \
        + cby*(C[1:-1, 2:]-C[1:-1, 1:-1]) \
        + cax*(C[2:, 1:-1]+C[:-2, 1:-1]-2*C[1:-1, 1:-1])\
        + cay*(C[1:-1, 2:]+C[1:-1, :-2]-2*C[1:-1, 1:-1])

def compute_step_rk4(C, D, k, cax, cay, cbx, cby, dt):

    k1, k2, k3, k4 = np.zeros_like(C), np.zeros_like(C), np.zeros_like(C), np.zeros_like(C)

    # Créer une copie pour la mise à jour en place
    C_new = C.copy()
    k1[1:-1, 1:-1] = F(C, k, cax, cay, cbx, cby)
    k2[1:-1, 1:-1] = F(C + 0.5 * dt * k1, k, cax, cay, cbx, cby)
    k3[1:-1, 1:-1] = F(C + 0.5 * dt * k2, k, cax, cay, cbx, cby)
    k4[1:-1, 1:-1] = F(C + dt * k3, k, cax, cay, cbx, cby)
    
    # Calculer la diffusion pour chaque point interne de la grille
    C_new[1:-1, 1:-1] = C[1:-1, 1:-1] + (dt / 6) * (k1[1:-1, 1:-1] + 2*k2[1:-1, 1:-1] + 2*k3[1:-1, 1:-1] + k4[1:-1, 1:-1])# dt*F(C)
    
    # Conditions aux bords (ici, on utilise des bords fixes à zéro)
    C_new[0, :] = C_new[1, :]
    C_new[-1, :] = C_new[-2, :]
    C_new[:, 0] = C_new[:, 1]
    C_new[:, -1] = C_new[:, -2]
    
    return C_new
    
def compute_step_euler(C, D, k, cax, cay, cbx, cby, dt):
    C_new = C.copy()

    # C_new[1:-1, 1:-1] = C[1:-1, 1:-1]*(1-k[1:-1, 1:-1]*dt) + betax[1:-1, 1:-1]*(C[2:, 1:-1]-C[1:-1, 1:-1]) + betay[1:-1, 1:-1]*(C[1:-1, 2:]-C[1:-1, 1:-1]) + alphax[1:-1, 1:-1]*(C[2:, 1:-1]+C[:-2, 1:-1]-2*C[1:-1, 1:-1])+alphay[1:-1, 1:-1]*(C[1:-1, 2:]+C[1:-1, :-2]-2*C[1:-1, 1:-1])
    C_new[1:-1, 1:-1] = C[1:-1, 1:-1]*(1-k[1:-1, 1:-1]*dt) \
        + cbx*(C[2:, 1:-1]-C[1:-1, 1:-1]) \
        + cby*(C[1:-1, 2:]-C[1:-1, 1:-1]) \
        + cax*(C[2:, 1:-1]+C[:-2, 1:-1]-2*C[1:-1, 1:-1])\
        + cay*(C[1:-1, 2:]+C[1:-1, :-2]-2*C[1:-1, 1:-1])
        
    # C_new[1:-1, 1:-1] = C[1:-1, 1:-1] + dt*F(C, k, cax, cay, cbx, cby)
    
    # C_new = params.initial_concentration*np.ones((params.nx, params.ny))*params.m1 + C_new*~params.m1
    
    # Conditions aux bords (ici, on utilise la condition de flux nul)
    C_new[0, :] = C_new[1, :]
    C_new[-1, :] = C_new[-2, :]
    C_new[:, 0] = C_new[:, 1]
    C_new[:, -1] = C_new[:, -2]

    return C_new

class FromPGVNet:
    def __init__(self, grid):
        self.grid = grid
    
    def __call__(self, params):
        grid = self.grid
        nx, ny = params.nx, params.ny
        assert ((nx<=grid.shape[0]) & (ny<=grid.shape[1]))
        Im = Image.fromarray(grid)
        image_reduite = Im.resize((nx, ny))
        M = np.array(image_reduite)
        # params.Mmat = M
        C = params.initial_concentration*np.ones((nx, ny))*((M>=200))
        params.C_ini = C
        print("Added initial C to params : params.C_ini")
        # Initialisation du coefficient de diffusion : 
        D = params.D*np.ones((nx, ny))*(M>200) + params.D/50*np.ones((nx, ny))*((M<=200) & (M>1))+ params.D/10*np.ones((nx, ny))*(M<=1)#+params.D/1000*np.ones((nx, ny))*m2
        params.D_mat = D
        print("Added Diff mat to params : params.D_mat")
        dDx = np.zeros((nx, ny))
        dDx[1:-1,1:-1] = D[2:,1:-1]-D[1:-1,1:-1] # Attention claculer la dérivé de D
        dDy = np.zeros((nx, ny))
        dDy[1:-1,1:-1] = D[1:-1, 2:]-D[1:-1,1:-1]  # Attention claculer la dérivé de D
        # Initialisation de l'absobtion/production en O2
        k = np.zeros((nx, ny))
        k[M<1] = params.k
        params.k_mat = k
        print("Added k mat to params : params.k_mat")
        # Initialisation des préfacteur de l'équation de diffusion discrète
        alphax = D*params.dt/(params.dx)**2
        alphay = D*params.dt/(params.dy)**2
        betax = dDx*params.dt/(params.dx)**2
        betay = dDy*params.dt/(params.dy)**2
            
        return C, D, k, alphax, alphay, betax, betay


class FromIMG:
    def __init__(self, vasc_path):
        self.vasc_path = vasc_path
        
    def __call__(self, params):
        # Initialisation de la concentration en O2
        # Im = Image.open("/Users/alexis/Downloads/IMG_0495.jpg")
        nx, ny = params.nx, params.ny
        Im = Image.open(self.vasc_path)
        image_reduite = Im.resize((nx, ny))
        M = np.array(image_reduite)
        m1 = M[:,:,0]<125 # concentration dans les vaissaux
        m2 = M[:,:,2]<125 # concentration dans les vaissaux + znd
        m3 = M[:,:,2]<125
        # params.m1 = m1
        m3[m1==1] = 0 # concentration dans les znd
        C = params.initial_concentration*np.ones((nx, ny))*m1
        params.C_ini = C
        print("Added initial C to params : params.C_ini")
        # Initialisation du coefficient de diffusion : 
        # D = params.D/100*np.ones((nx, ny))*m1+params.D*np.ones((nx, ny))*~m2+params.D/1000*np.ones((nx, ny))*m2
        D = params.D*np.ones((nx, ny))*m1+params.D/100*np.ones((nx, ny))*~m2+params.D/1000*np.ones((nx, ny))*m2
        params.D_mat = D
        print("Added Diff mat to params : params.D_mat")
        dDx = np.zeros((nx, ny))
        dDx[1:-1,1:-1] = D[2:,1:-1]-D[1:-1,1:-1] # Attention claculer la dérivé de D
        dDy = np.zeros((nx, ny))
        dDy[1:-1,1:-1] = D[1:-1, 2:]-D[1:-1,1:-1]  # Attention claculer la dérivé de D
        # Initialisation de l'absobtion/production en O2
        k = np.zeros((nx, ny))
        k[~m2] = params.k/np.sum(~m2)
        k[m3] = params.k/np.sum(m3)*10
        k[m1] = -params.k/(np.sum(m1)+np.sum(m3)/10)
        params.k_mat = k
        print("Added k mat to params : params.k_mat")
        # Initialisation des préfacteur de l'équation de diffusion discrète
        alphax = D*params.dt/(params.dx)**2
        alphay = D*params.dt/(params.dy)**2
        betax = dDx*params.dt/(params.dx)**2
        betay = dDy*params.dt/(params.dy)**2
        return C, D, k, alphax, alphay, betax, betay
        
class FromCustom:
   def __init__(self, C_init, D_init, k_init):
       self.C_init = C_init # Attention les entrés prennent des valeurs de pixels dans [0, 1]
       self.D_init = D_init # Attention les entrés prennent des valeurs de pixels dans [0, 1]
       self.k_init = k_init # Attention les entrés prennent des valeurs de pixels dans [-1, 1]
       
   def __call__(self, params):
       # Initialisation de la concentration en O2
       C = params.initial_concentration*self.C_init
       params.C_ini = C
       print("Added initial C to params : params.C_ini")
       # Initialisation du coefficient de diffusion : 
       D = params.D*self.D_init
       params.D_mat = D
       print("Added Diff mat to params : params.D_mat")
       dDx = np.zeros((params.nx, params.ny))
       dDx[1:-1,1:-1] = D[2:,1:-1]-D[1:-1,1:-1] # Attention claculer la dérivé de D
       dDy = np.zeros((params.nx, params.ny))
       dDy[1:-1,1:-1] = D[1:-1, 2:]-D[1:-1,1:-1]  # Attention claculer la dérivé de D
       # Initialisation de l'absobtion/production en O2
       k = params.k*self.k_init
       params.k_mat = k
       print("Added k mat to params : params.k_mat")
       # Initialisation des préfacteur de l'équation de diffusion discrète
       alphax = D*params.dt/(params.dx)**2
       alphay = D*params.dt/(params.dy)**2
       betax = dDx*params.dt/(params.dx)**2
       betay = dDy*params.dt/(params.dy)**2
       return C, D, k, alphax, alphay, betax, betay

def run_simulation(params, init_method, C_0_cst=True, save_last_only = False):
    C, D, k, alphax, alphay, betax, betay = init_method(params)
    
    if C_0_cst:
        C_i = C.copy()
    else:
        C_i = None
    
    dt = params.dt
    crit_stab(alphax, alphay)
    cax = alphax[1:-1, 1:-1]
    cay = alphay[1:-1, 1:-1]
    cbx = betax[1:-1, 1:-1]
    cby = betay[1:-1, 1:-1]  
    L_result = [C.copy()]
    print(f"\n Quantité de matière totale initiale : {N_tot(L_result[0], params.dx, params.dy):e} mol")
    for i in tqdm(range(params.nt)):
        C = compute_step_euler(C, D, k, cax, cay, cbx, cby, dt)
        if C_i is not None:
            C[C_i!=0] = C_i[C_i!=0]
        if not save_last_only:
            if i%params.step==0:
                L_result.append(C)
    if save_last_only:
        L_result.append(C)
    print(f"\n Quantité de matière totale finale : {N_tot(L_result[-1], params.dx, params.dy):e} mol")
    return L_result


