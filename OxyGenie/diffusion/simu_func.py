import numpy as np
from PIL import Image
from tqdm import tqdm


def crit_stab(alphax, alphay):
    r"""
    Evaluates the Courant–Friedrichs–Lewy (CFL) condition for numerical stability with the maximum values of `alphax` and `alphay`.

    The function checks if the sum of the maximum values of `alphax` and `alphay` exceeds
    the stability threshold (0.5). If the criterion is not met, a `ValueError` is raised.

    Parameters
    ----------
    alphax : 2D np.ndarray
        Here\: :math:`\alpha_x = \frac{D \cdot \Delta t}{(\Delta x)^2}`
    alphay : 2D np.ndarray
        Here\: :math:`\alpha_y = \frac{D \cdot \Delta t}{(\Delta y)^2}`

    Raises
    ------
    ValueError
        If the stability criterion is not respected (stab_coef >= 0.5). The error message includes
        the computed stability coefficient.

    Returns
    -------
    None.
        Prints a message indicating whether the stability criterion is respected.
    """
    stab_coef = np.max(alphax) + np.max(alphay)
    if stab_coef >= 0.5:
        raise ValueError(
            f"Critère de stabilité non respecté (>0.5): {stab_coef:.2e}")
    else:
        print(f"Critère de stabilité respecté (<0.5): {stab_coef:.2e}")


def N_tot(C, dx, dy):
    r"""
    Calculates the total number of particles in a 2D concentration field.

    .. warning::
        
        -Avogadro's number is used to convert molar concentration
        to the number of particles.

        -The boundary of width `l = 1` is excluded from the sum to avoid edge effects from the boundary condintions.


    Parameters
    ----------
    C : 2D np.ndarray
        The 2D array representing the concentration field (in moles per square meter).
        The boundary region is excluded from the calculation.
    dx : float
        The grid spacing in the x-direction (in meters).
    dy : float
        The grid spacing in the y-direction (in meters).

    Returns
    -------
    float
        The total number of particles within the central region of the grid.
        The result is dimensionless.
    """
    l = 1
    Na = 6.02214076e23
    return np.sum(C[l:-l, l:-l]) * (dx * dy * Na)


def init_simulation(params, vasc_path="/Users/alexis/Downloads/IMG_38EDAE940454-1.jpeg"):

    if vasc_path is not None:
        # Initialisation de la concentration en O2
        # Im = Image.open("/Users/alexis/Downloads/IMG_0495.jpg")
        nx, ny = params.nx, params.ny
        Im = Image.open(vasc_path)
        image_reduite = Im.resize((nx, ny))
        M = np.array(image_reduite)
        m1 = M[:, :, 0] < 125  # concentration dans les vaissaux
        m2 = M[:, :, 2] < 125  # concentration dans les vaissaux + znd
        m3 = M[:, :, 2] < 125
        params.m1 = m1
        m3[m1 == 1] = 0  # concentration dans les znd
        C = params.initial_concentration * np.ones((nx, ny)) * m1
        # Initialisation du coefficient de diffusion :
        # D = params.D/100*np.ones((nx, ny))*m1+params.D*np.ones((nx, ny))*~m2+params.D/1000*np.ones((nx, ny))*m2
        D = params.D * np.ones((nx, ny)) * m1 + params.D / 100 * np.ones(
            (nx, ny)) * ~m2 + params.D / 1000 * np.ones((nx, ny)) * m2
        params.D_mat = D
        dDx = np.zeros((nx, ny))
        # Attention claculer la dérivé de D
        dDx[1:-1, 1:-1] = D[2:, 1:-1] - D[1:-1, 1:-1]
        dDy = np.zeros((nx, ny))
        # Attention claculer la dérivé de D
        dDy[1:-1, 1:-1] = D[1:-1, 2:] - D[1:-1, 1:-1]
        # Initialisation de l'absobtion/production en O2
        k = np.zeros((nx, ny))
        k[~m2] = params.k / np.sum(~m2)
        k[m3] = params.k / np.sum(m3) * 10
        k[m1] = -params.k / (np.sum(m1) + np.sum(m3) / 10)
        params.k_mat = k
        # Initialisation des préfacteur de l'équation de diffusion discrète
        alphax = D * params.dt / (params.dx)**2
        alphay = D * params.dt / (params.dy)**2
        betax = dDx * params.dt / (params.dx)**2
        betay = dDy * params.dt / (params.dy)**2
        return C, D, k, alphax, alphay, betax, betay

    else:
        # Initialisation de la concentration en O2
        center_x, center_y = params.nx // 2, params.ny // 2
        size = params.initial_source_size
        concentration = params.initial_concentration
        C = np.zeros((params.nx, params.ny))
        C[center_x - size:center_x + size, center_y -
            size:center_y + size] = concentration
        # Initialisation du coefficient de diffusion :
        X, Y = np.meshgrid(np.linspace(-1, 1, params.ny),
                           np.linspace(0, 2, params.nx))
        D2 = np.exp(-(X**2 + Y**2))
        D = params.D * D2
        params.D_mat = D
        dDx = np.zeros((nx, ny))
        # Attention claculer la dérivé de D
        dDx[1:-1, 1:-1] = D[2:, 1:-1] - D[1:-1, 1:-1]
        dDy = np.zeros((nx, ny))
        # Attention claculer la dérivé de D
        dDy[1:-1, 1:-1] = D[1:-1, 2:] - D[1:-1, 1:-1]
        # Initialisation de l'absobtion/production en O2
        k = np.zeros((params.nx, params.ny))
        k[params.nx // 2 - 5:params.nx // 2 + 5, params.ny //
            2 - 10:params.ny // 2 - 5] = params.k
        k[params.nx // 2, params.ny // 2 + 10] = -params.k
        params.k_mat = k
        # Initialisation des préfacteur de l'équation de diffusion discrète
        alphax = D * params.dt / (params.dx)**2
        alphay = D * params.dt / (params.dy)**2
        betax = dDx * params.dt / (params.dx)**2
        betay = dDy * params.dt / (params.dy)**2
        return C, D, k, alphax, alphay, betax, betay


def F(C, k, cax, cay, cbx, cby):
    r"""
    Computes the rate of change of the concentration field `C` based on the diffusion and reaction terms.

    This function calculates the right-hand side of the diffusion-reaction equation using finite differences 
    for the diffusion terms and a simple linear reaction term.
    
    .. warning::
        This function is only working for RK4 compute step scheme
        
    
    """
    return -k[1:-1, 1:-1] * C[1:-1, 1:-1] \
        + cbx * (C[2:, 1:-1] - C[1:-1, 1:-1]) \
        + cby * (C[1:-1, 2:] - C[1:-1, 1:-1]) \
        + cax * (C[2:, 1:-1] + C[:-2, 1:-1] - 2 * C[1:-1, 1:-1])\
        + cay * (C[1:-1, 2:] + C[1:-1, :-2] - 2 * C[1:-1, 1:-1])


def compute_step_rk4(C, D, k, cax, cay, cbx, cby, dt):
    r"""
    Performs one step of the 4th-order Runge-Kutta (RK4) method to update the concentration field.

    The RK4 method is a numerical technique for solving ordinary differential equations (ODEs). 
    This function applies it to the concentration field `C` in a 2D grid, accounting for diffusion and absorbtion.
    
    .. note::
        The boundary conditions are set to be "no flux" (Neumann).

    Parameters
    ----------
    C : 2D np.ndarray
        The current concentration field.

    D : 2D np.ndarray
        The diffusion coefficient field, with the same shape as `C`.

    k : 2D np.ndarray
        The absorbtion rate field, with the same shape as `C`.

    cax : 2D np.ndarray
        The precomputed prefactor for the diffusion term in the x-direction.

    cay : 2D np.ndarray
        The precomputed prefactor for the diffusion term in the y-direction.

    cbx : 2D np.ndarray
        The precomputed prefactor for the diffusion flux term in the x-direction.

    cby : 2D np.ndarray
        The precomputed prefactor for the diffusion flux term in the y-direction.

    dt : float
        The time step (`dt`) used for the RK4 method to update the concentration.

    Returns
    -------
    2D np.ndarray
        The updated concentration field `C_new` after one RK4 step, with the same shape as `C`.

    """
    k1, k2, k3, k4 = np.zeros_like(C), np.zeros_like(
        C), np.zeros_like(C), np.zeros_like(C)

    # Créer une copie pour la mise à jour en place
    C_new = C.copy()
    k1[1:-1, 1:-1] = F(C, k, cax, cay, cbx, cby)
    k2[1:-1, 1:-1] = F(C + 0.5 * dt * k1, k, cax, cay, cbx, cby)
    k3[1:-1, 1:-1] = F(C + 0.5 * dt * k2, k, cax, cay, cbx, cby)
    k4[1:-1, 1:-1] = F(C + dt * k3, k, cax, cay, cbx, cby)

    # Calculer la diffusion pour chaque point interne de la grille
    C_new[1:-1, 1:-1] = C[1:-1, 1:-1] + (dt / 6) * (
        k1[1:-1, 1:-1] + 2 * k2[1:-1, 1:-1] + 2 * k3[1:-1, 1:-1] + k4[1:-1, 1:-1])  # dt*F(C)

    # Conditions aux bords (ici, on utilise des bords fixes à zéro)
    C_new[0, :] = C_new[1, :]
    C_new[-1, :] = C_new[-2, :]
    C_new[:, 0] = C_new[:, 1]
    C_new[:, -1] = C_new[:, -2]

    return C_new


def compute_step_euler(C, D, k, cax, cay, cbx, cby, dt):
    r"""
    Performs one step of the Euler method to update the concentration field.

    The update is based on the following form of the diffusion equation:

    :math:`\frac{\partial C(\textbf{r}, t)}{\partial t} = \text{div}(D(\textbf{r}) \cdot \overrightarrow{\text{grad}}(C(\textbf{r}, t)) - k(\textbf{r}) \cdot C(\textbf{r}, t)`

    where the diffusion terms are discretized using finite differences (forward for flux terms and central for diffusion terms).

    .. note::
        The boundary conditions are set to be "no flux" (Neumann).
    
    .. warning::
        The stability criterion needs to be respected.

    Parameters
    ----------
    C : 2D np.ndarray
        The current concentration field.

    D : 2D np.ndarray
        The diffusion coefficient field, with the same shape as `C`.

    k : 2D np.ndarray
        The absorbtion rate field, with the same shape as `C`.

    cax : 2D np.ndarray
        The precomputed prefactor for the diffusion term in the x-direction.

    cay : 2D np.ndarray
        The precomputed prefactor for the diffusion term in the y-direction.

    cbx : 2D np.ndarray
        The precomputed prefactor for the diffusion flux term in the x-direction.

    cby : 2D np.ndarray
        The precomputed prefactor for the diffusion flux term in the y-direction.

    dt : float
        The time step (`dt`) used for the RK4 method to update the concentration.

    Returns
    -------
    2D np.ndarray
        The updated concentration field `C_new` after one RK4 step, with the same shape as `C`.

    """
    C_new = C.copy()

    # C_new[1:-1, 1:-1] = C[1:-1, 1:-1]*(1-k[1:-1, 1:-1]*dt) + betax[1:-1, 1:-1]*(C[2:, 1:-1]-C[1:-1, 1:-1]) + betay[1:-1, 1:-1]*(C[1:-1, 2:]-C[1:-1, 1:-1]) + alphax[1:-1, 1:-1]*(C[2:, 1:-1]+C[:-2, 1:-1]-2*C[1:-1, 1:-1])+alphay[1:-1, 1:-1]*(C[1:-1, 2:]+C[1:-1, :-2]-2*C[1:-1, 1:-1])
    C_new[1:-1, 1:-1] = C[1:-1, 1:-1] * (1 - k[1:-1, 1:-1] * dt) \
        + cbx * (C[2:, 1:-1] - C[1:-1, 1:-1]) \
        + cby * (C[1:-1, 2:] - C[1:-1, 1:-1]) \
        + cax * (C[2:, 1:-1] + C[:-2, 1:-1] - 2 * C[1:-1, 1:-1])\
        + cay * (C[1:-1, 2:] + C[1:-1, :-2] - 2 * C[1:-1, 1:-1])

    # C_new[1:-1, 1:-1] = C[1:-1, 1:-1] + dt*F(C, k, cax, cay, cbx, cby)

    # C_new = params.initial_concentration*np.ones((params.nx, params.ny))*params.m1 + C_new*~params.m1

    # Conditions aux bords (ici, on utilise la condition de flux nul)
    C_new[0, :] = C_new[1, :]
    C_new[-1, :] = C_new[-2, :]
    C_new[:, 0] = C_new[:, 1]
    C_new[:, -1] = C_new[:, -2]

    return C_new


class FromPGVNet:
    r"""
    A class to initialize concentration, diffusion coefficients, and reaction rates from a vascular network generated by the pgvnet sub-module.

    .. note::
        - The provided grid is resized to match the grid size defined by `params.nx` and `params.ny`.
        - The concentration (`C`), diffusion coefficient (`D`), and reaction rate (`k`) fields are initialized 
          based on values in the grid.
        - Diffusion coefficient and absorbtion rate are adjusted based on pixel values, where:
            - Areas with higher grid values represent regions with higher diffusion coefficients.
            - Regions with pixel values below a threshold represent areas with production or absorption rates.
        - Return via `__call__`

    Parameters
    ----------
    grid : 2D np.ndarray
        Vascular network generated by the pgvnet sub-module. The grid values are
        used to define regions with different diffusion and absorbtion properties based on thresholds.

    Returns
    -------
    tuple
        - `C` : 2D np.ndarray
            Initialized concentration field based on grid values.
        - `D` : 2D np.ndarray
            Diffusion coefficient field initialized based on grid values.
        - `k` : 2D np.ndarray
            Absorbtion rate field initialized based on grid values.
        - `alphax` : 2D np.ndarray
            Prefactor for diffusion in the x-direction.
        - `alphay` : 2D np.ndarray
            Prefactor for diffusion in the y-direction.
        - `betax` : 2D np.ndarray
            Prefactor for the diffusion flux term in the x-direction.
        - `betay` : 2D np.ndarray
            Prefactor for the diffusion flux term in the y-direction


    .. code-block:: python
    
        from OxyGenie.diffusion import *
        import OxyGenie.pgvnet as pgv
        
        grid = pgv.simple_generation(grid_size=1280)[0]
        params = {
            "D": 1e-5, "k": 8, "Lx": 0.05, "Ly": 0.05, "T": 0.5, 
            "nt": 2500, "nx": 256 * 2, "ny": 256 * 2,
            "initial_concentration": 100.0, "step": 10,
        }
        simparams = SimulationParams(**params)
        from_pgvnet = FromPGVNet(grid)
        C, D, k, alphax, alphay, betax, betay = from_pgvnet(params)
        
    """

    def __init__(self, grid):
        self.grid = grid

    def __call__(self, params):
        grid = self.grid
        nx, ny = params.nx, params.ny
        assert ((nx <= grid.shape[0]) & (ny <= grid.shape[1]))
        Im = Image.fromarray(grid)
        image_reduite = Im.resize((nx, ny))
        M = np.array(image_reduite)
        # params.Mmat = M
        C = params.initial_concentration * np.ones((nx, ny)) * ((M >= 200))
        params.C_ini = C
        print("Added initial C to params : params.C_ini")
        # Initialisation du coefficient de diffusion :
        D = params.D / 2 * np.ones((nx, ny)) * (M > 200) + params.D / 3 * np.ones((nx, ny)) * ((M <= 200) & (
            M > 1)) + params.D * np.ones((nx, ny)) * (M <= 1)  # +params.D/1000*np.ones((nx, ny))*m2
        params.D_mat = D
        print("Added Diff mat to params : params.D_mat")
        dDx = np.zeros((nx, ny))
        # Attention claculer la dérivé de D
        dDx[1:-1, 1:-1] = D[2:, 1:-1] - D[1:-1, 1:-1]
        dDy = np.zeros((nx, ny))
        # Attention claculer la dérivé de D
        dDy[1:-1, 1:-1] = D[1:-1, 2:] - D[1:-1, 1:-1]
        # Initialisation de l'absobtion/production en O2
        k = np.zeros((nx, ny))
        k[M < 1] = params.k
        params.k_mat = k
        print("Added k mat to params : params.k_mat")
        # Initialisation des préfacteur de l'équation de diffusion discrète
        alphax = D * params.dt / (params.dx)**2
        alphay = D * params.dt / (params.dy)**2
        betax = dDx * params.dt / (params.dx)**2
        betay = dDy * params.dt / (params.dy)**2

        return C, D, k, alphax, alphay, betax, betay


class FromIMG:
    r"""
    A class to initialize concentration, diffusion coefficients, and reaction rates from an image.


    .. note::
        - The provided image is resized to match the grid size defined by `params.nx` and `params.ny`.
        - The concentration (`C`), diffusion coefficient (`D`), and reaction rate (`k`) fields are initialized 
          based on chanel values in the grid.
        - Return via `__call__`
    
    Parameters
    ----------
    
    vasc_path : str
        Path to the image file containing spatial information. The image is used to define regions with different 
        diffusion and reaction properties based on the channel pixel values.
    
    Returns
    -------
    tuple
        - `C` : 2D np.ndarray
            Initialized concentration field based on grid values.
        - `D` : 2D np.ndarray
            Diffusion coefficient field initialized based on grid values.
        - `k` : 2D np.ndarray
            Absorbtion rate field initialized based on grid values.
        - `alphax` : 2D np.ndarray
            Prefactor for diffusion in the x-direction.
        - `alphay` : 2D np.ndarray
            Prefactor for diffusion in the y-direction.
        - `betax` : 2D np.ndarray
            Prefactor for the diffusion flux term in the x-direction.
        - `betay` : 2D np.ndarray
            Prefactor for the diffusion flux term in the y-direction
    
    
    .. code-block:: python
    
        from OxyGenie.diffusion import *
        
        params = {
            "D": 1e-5, "k": 8, "Lx": 0.05, "Ly": 0.05, "T": 0.5, 
            "nt": 2500, "nx": 256 * 2, "ny": 256 * 2,
            "initial_concentration": 100.0, "step": 10,
        }
        simparams = SimulationParams(**params)
        from_IMG = FromIMG("/path.jpg")
        C, D, k, alphax, alphay, betax, betay = from_IMG(params)
        
    """
    
    def __init__(self, vasc_path):
        self.vasc_path = vasc_path

    def __call__(self, params):
        # Initialisation de la concentration en O2
        # Im = Image.open("/Users/alexis/Downloads/IMG_0495.jpg")
        nx, ny = params.nx, params.ny
        Im = Image.open(self.vasc_path)
        image_reduite = Im.resize((nx, ny))
        M = np.array(image_reduite)
        m1 = M[:, :, 0] < 125  # concentration dans les vaissaux
        m2 = M[:, :, 2] < 125  # concentration dans les vaissaux + znd
        m3 = M[:, :, 2] < 125
        # params.m1 = m1
        m3[m1 == 1] = 0  # concentration dans les znd
        C = params.initial_concentration * np.ones((nx, ny)) * m1
        params.C_ini = C
        print("Added initial C to params : params.C_ini")
        # Initialisation du coefficient de diffusion :
        # D = params.D/100*np.ones((nx, ny))*m1+params.D*np.ones((nx, ny))*~m2+params.D/1000*np.ones((nx, ny))*m2
        D = params.D * np.ones((nx, ny)) * m1 + params.D / 100 * np.ones(
            (nx, ny)) * ~m2 + params.D / 1000 * np.ones((nx, ny)) * m2
        params.D_mat = D
        print("Added Diff mat to params : params.D_mat")
        dDx = np.zeros((nx, ny))
        # Attention claculer la dérivé de D
        dDx[1:-1, 1:-1] = D[2:, 1:-1] - D[1:-1, 1:-1]
        dDy = np.zeros((nx, ny))
        # Attention claculer la dérivé de D
        dDy[1:-1, 1:-1] = D[1:-1, 2:] - D[1:-1, 1:-1]
        # Initialisation de l'absobtion/production en O2
        k = np.zeros((nx, ny))
        k[~m2] = params.k / np.sum(~m2)
        k[m3] = params.k / np.sum(m3) * 10
        k[m1] = -params.k / (np.sum(m1) + np.sum(m3) / 10)
        params.k_mat = k
        print("Added k mat to params : params.k_mat")
        # Initialisation des préfacteur de l'équation de diffusion discrète
        alphax = D * params.dt / (params.dx)**2
        alphay = D * params.dt / (params.dy)**2
        betax = dDx * params.dt / (params.dx)**2
        betay = dDy * params.dt / (params.dy)**2
        return C, D, k, alphax, alphay, betax, betay


class FromCustom:
    """
    A class to initialize concentration, diffusion coefficients, and reaction rates from an custom distributions.

    .. warning::    
        - `C_init` and `D_init` values must be normalized to [0, 1].
        - `k_init` values must be normalized to [-1, 1].


    Parameters
    ----------
    C_init : 2D np.ndarray
        Initial concentration distribution scaled in the range [0, 1].
    D_init : 2D np.ndarray
        Initial diffusion coefficient distribution scaled in the range [0, 1].
    k_init : 2D np.ndarray
        Initial reaction rate distribution scaled in the range [-1, 1].

    Returns
    -------
    tuple
        - `C` : 2D np.ndarray
            Initialized concentration field based on grid values.
        - `D` : 2D np.ndarray
            Diffusion coefficient field initialized based on grid values.
        - `k` : 2D np.ndarray
            Absorbtion rate field initialized based on grid values.
        - `alphax` : 2D np.ndarray
            Prefactor for diffusion in the x-direction.
        - `alphay` : 2D np.ndarray
            Prefactor for diffusion in the y-direction.
        - `betax` : 2D np.ndarray
            Prefactor for the diffusion flux term in the x-direction.
        - `betay` : 2D np.ndarray
            Prefactor for the diffusion flux term in the y-direction
    
    
    .. code-block:: python
    
        from OxyGenie.diffusion import *
        
        params = {
            "D": 1e-5, "k": 8, "Lx": 0.05, "Ly": 0.05, "T": 0.5, 
            "nt": 2500, "nx": 256 * 2, "ny": 256 * 2,
            "initial_concentration": 100.0, "step": 10,
        }
        simparams = SimulationParams(**params)
        C = np.zeros((simparams.nx,simparams.ny))
        C[120:130, 120:130] = 1

        X, Y = np.meshgrid(np.linspace(-1, 1, simparams.ny), np.linspace(-1, 1, simparams.nx))
        D = np.exp(-((X)**2+(Y-1)**2)/1.5)

        k = np.ones((simparams.nx, simparams.ny))

        from_Custom = FromCustom(C, D, k)
        C, D, k, alphax, alphay, betax, betay = from_Custom(params)

    """

    def __init__(self, C_init, D_init, k_init):
        # Attention les entrés prennent des valeurs de pixels dans [0, 1]
        self.C_init = C_init
        # Attention les entrés prennent des valeurs de pixels dans [0, 1]
        self.D_init = D_init
        # Attention les entrés prennent des valeurs de pixels dans [-1, 1]
        self.k_init = k_init

    def __call__(self, params):
        # Initialisation de la concentration en O2
        C = params.initial_concentration * self.C_init
        params.C_ini = C
        print("Added initial C to params : params.C_ini")
        # Initialisation du coefficient de diffusion :
        D = params.D * self.D_init
        params.D_mat = D
        print("Added Diff mat to params : params.D_mat")
        dDx = np.zeros((params.nx, params.ny))
        # Attention claculer la dérivé de D
        dDx[1:-1, 1:-1] = D[2:, 1:-1] - D[1:-1, 1:-1]
        dDy = np.zeros((params.nx, params.ny))
        # Attention claculer la dérivé de D
        dDy[1:-1, 1:-1] = D[1:-1, 2:] - D[1:-1, 1:-1]
        # Initialisation de l'absobtion/production en O2
        k = params.k * self.k_init
        params.k_mat = k
        print("Added k mat to params : params.k_mat")
        # Initialisation des préfacteur de l'équation de diffusion discrète
        alphax = D * params.dt / (params.dx)**2
        alphay = D * params.dt / (params.dy)**2
        betax = dDx * params.dt / (params.dx)**2
        betay = dDy * params.dt / (params.dy)**2
        return C, D, k, alphax, alphay, betax, betay


def run_simulation(params, init_method, C_0_cst=True, save_last_only=False, C0=None):
    r"""
    Runs the simulation of the 2D diffusion equation:

    .. math::
        \frac{\partial C(\mathbf{r}, t)}{\partial t} = \nabla \cdot \big(D(\mathbf{r}) \cdot \nabla C(\mathbf{r}, t)\big) - k(\mathbf{r}) \cdot C(\mathbf{r}, t)

    Parameters
    ----------
    params : object
        Contains the simulation parameters, including time step (`dt`), grid resolution (`dx`, `dy`), 
        and number of time steps (`nt`).
    init_method : callable
        A function that initializes the state variables, returning `C`, `D`, `k`, `alphax`, `alphay`, 
        `betax`, and `betay`.
    C_0_cst : bool, optional
        If True, maintains the initial constant value of `C` throughout the simulation (default: True).
    save_last_only : bool, optional
        If True, only saves the final state of the simulation (default: False).
    C0 : 2D np.ndarray, optional
        An optional initial concentration array to override the default initialization.

    Returns
    -------
    list of 2D np.ndarray
        A list containing the concentration fields at each saved time step. If `save_last_only` is True, 
        only the final state is saved.

    Raises
    ------
    ValueError
        If the stability criterion is not respected.

    Notes
    -----
    The function uses an explicit Euler method to solve the diffusion equation. The stability criterion is checked
    before the simulation begins using the `crit_stab` function. The total amount of substance in the system is 
    printed before and after the simulation.

    Examples
    --------
    To run a simulation with default parameters:

    .. code-block:: python
    
        from OxyGenie.diffusion import *
        
        params = {
            "D": 1e-5, "k": 8, "Lx": 0.05, "Ly": 0.05, "T": 0.5, 
            "nt": 2500, "nx": 256 * 2, "ny": 256 * 2,
            "initial_concentration": 100.0, "step": 10,
        }
        simparams = SimulationParams(**params)
        
        C = np.zeros((simparams.nx,simparams.ny))
        C[120:130, 120:130] = 1
        X, Y = np.meshgrid(np.linspace(-1, 1, simparams.ny), np.linspace(-1, 1, simparams.nx))
        D = np.exp(-((X)**2+(Y-1)**2)/1.5)
        k = np.ones((simparams.nx, simparams.ny))
        
        results = run_simulation(simparams, FromCustom(C, D, k), C_0_cst=True)
        
        Simu_plot.simple(simparams, results[-1])

    """

    C, D, k, alphax, alphay, betax, betay = init_method(params)

    if C_0_cst:
        C_i = C.copy()
    else:
        C_i = None

    if C0 is not None:
        C = C0

    dt = params.dt
    crit_stab(alphax, alphay)
    cax = alphax[1:-1, 1:-1]
    cay = alphay[1:-1, 1:-1]
    cbx = betax[1:-1, 1:-1]
    cby = betay[1:-1, 1:-1]
    L_result = [C.copy()]
    print(
        f"\n Quantité de matière totale initiale : {N_tot(L_result[0], params.dx, params.dy):e} mol")
    for i in tqdm(range(params.nt)):
        C = compute_step_euler(C, D, k, cax, cay, cbx, cby, dt)
        if C_i is not None:
            C[C_i != 0] = C_i[C_i != 0]
        if not save_last_only:
            if i % params.step == 0:
                L_result.append(C)
    if save_last_only:
        L_result.append(C)
    print(
        f"\n Quantité de matière totale finale : {N_tot(L_result[-1], params.dx, params.dy):e} mol")
    return L_result
