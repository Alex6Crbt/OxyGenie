import numpy as np


class SimulationParams:
    def __init__(self, **kwargs):
        # Initialisation des paramètres de base fournis par l'utilisateur
        self.D = kwargs.get("D", 1e-5)
        self.k = kwargs.get("k", 0)
        self._Lx = kwargs.get("Lx", 0.01)  # Toujours donné
        self._Ly = kwargs.get("Ly", 0.01)  # Toujours donné
        self._T = kwargs.get("T", 0.01)  # Toujours donné
        self.initial_concentration = kwargs.get("initial_concentration", 100.0)
        self.speed = kwargs.get("speed", 1)
        self.step = kwargs.get("step", 1)

        # Paramètres dérivés
        self._nt = kwargs.get("nt", 10000)
        self._nx = kwargs.get("nx", 100)
        self._ny = kwargs.get("ny", 100)
        self._dt = kwargs.get("dt", None)
        self._dx = kwargs.get("dx", None)
        self._dy = kwargs.get("dy", None)

        # Calcul des paramètres dérivés s'ils ne sont pas fournis
        if self._dt is None and self._nt is not None:
            self._dt = self.T / (self._nt - 1)

        if self._dx is None and self._nx is not None:
            self._dx = self.Lx / (self._nx - 1)

        if self._dy is None and self._ny is not None:
            self._dy = self.Ly / (self._ny - 1)

        if self._dt is not None:
            self._nt = int(np.floor(self.T / self._dt) + 1)

        if self._dx is not None:
            self._nx = int(np.floor(self.Lx / self._dx) + 1)

        if self._dy is not None:
            self._ny = int(np.floor(self.Ly / self._dy) + 1)

    # Getter et Setter pour _nt
    @property
    def nt(self):
        return self._nt

    @nt.setter
    def nt(self, value):
        if value is not None and self._nt != value:  # Seule modification si la valeur change
            self._nt = value
            self._dt = self.T / (self._nt - 1)  # Recalcule dt

    # Getter et Setter pour _dt
    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        if value is not None and self._dt != value:  # Seule modification si la valeur change
            self._dt = value
            self._nt = int(np.floor(self.T / self._dt) + 1)  # Recalcule nt

    # Getter et Setter pour _nx
    @property
    def nx(self):
        return self._nx

    @nx.setter
    def nx(self, value):
        if value is not None and self._nx != value:  # Seule modification si la valeur change
            self._nx = value
            self._dx = self.Lx / (self._nx - 1)  # Recalcule dx

    # Getter et Setter pour _dx
    @property
    def dx(self):
        return self._dx

    @dx.setter
    def dx(self, value):
        if value is not None and self._dx != value:  # Seule modification si la valeur change
            self._dx = value
            self._nx = int(np.floor(self.Lx / self._dx) + 1)  # Recalcule nx

    # Getter et Setter pour _ny
    @property
    def ny(self):
        return self._ny

    @ny.setter
    def ny(self, value):
        if value is not None and self._ny != value:  # Seule modification si la valeur change
            self._ny = value
            self._dy = self.Ly / (self._ny - 1)  # Recalcule dy

    # Getter et Setter pour _dy
    @property
    def dy(self):
        return self._dy

    @dy.setter
    def dy(self, value):
        if value is not None and self._dy != value:  # Seule modification si la valeur change
            self._dy = value
            self._ny = int(np.floor(self.Ly / self._dy) + 1)  # Recalcule ny

    # Getter et Setter pour _T
    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        if value is not None and self._T != value:  # Seule modification si la valeur change
            self._T = value
            self._nt = int(np.floor(self._T / self._dt) + 1)  # Recalcule nt

    # Getter et Setter pour _Lx
    @property
    def Lx(self):
        return self._Lx

    @Lx.setter
    def Lx(self, value):
        if value is not None and self._Lx != value:  # Seule modification si la valeur change
            self._Lx = value
            self._nx = int(np.floor(self._Lx / self._dx) + 1)  # Recalcule nx

    # Getter et Setter pour _Ly
    @property
    def Ly(self):
        return self._Ly

    @Ly.setter
    def Ly(self, value):
        if value is not None and self._Ly != value:  # Seule modification si la valeur change
            self._Ly = value
            self._ny = int(np.floor(self._Ly / self._dy) + 1)  # Recalcule ny

    def __repr__(self):
        return f"SimulationParams(D={self.D}, k={self.k},\n" \
               f"T={self.T}, Lx={self.Lx}, Ly={self.Ly},\n"\
               f"nt={self._nt}, nx={self._nx}, ny={self._ny},\n"\
               f"dt={self._dt}, dx={self._dx}, dy={self._dy})"


def crit_stab(alphax, alphay):
    stab_coef = np.max(alphax) + np.max(alphay)
    if stab_coef >= 0.5:
        raise ValueError(
            f"Critère de stabilité non respecté (>0.5): {stab_coef:.2e}")
    else:
        print(f"Critère de stabilité respecté (<0.5): {stab_coef:.2e}")
