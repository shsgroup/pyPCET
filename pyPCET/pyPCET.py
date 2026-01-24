import numpy as np
import numpy.linalg as la
try:
    from scipy.integrate import simps
except ImportError:
    from scipy.integrate import simpson as simps
from .functions import *
from .units import *
from .FGH_1D import FGH_1D


class pyPCET(object):
    """
    This class set up the calculation for nonadiabatic PCET rate constant
    In this implementation, we assumed that the vibronic coupling V_uv = V_el * S_uv

    The input parameters are:
        ReacProtonPot (2D array or function): proton potential of the reactant state
        ProdProtonPot (2D array or function): proton potential of the product state
        DeltaG (float): reaction free energy of the PCET process in eV. This should be the free energy difference between electronic states, i.e., ZPEs should not be included
        Lambda (float): reorganization energy of the PCET reaction in eV
        Vel (float): electronic coupling between reactant and product states in eV, default = 0.0434 eV = 1 kcal/mol
        NStates (int): number of proton vibrational states to be calculated, default = 10
        NGridPot (int): number of grid points used for FGH calculation, default = 256
        Smooth (string): method to smooth the proton potential if given as 2Darray, possible choices are 'fit_poly6', 'fit_poly8', 'bspline', default = 'bspline' 

    The program will automatically determine the ranges of proton position to perform subsequent calculations. 
    Users could fine tune these ranges by parseing additional inputs 'rmin', 'rmax'. 
    """
    def __init__(self, ReacProtonPot, ProdProtonPot, DeltaG, Lambda, Vel=0.0434, NStates=10, NGridPot=256, Smooth='bspline', **kwargs):
        """
        *** Initialization ***
        The input of the proton potential can be either a 2D array or a callable function. 

        If these inputs are 2D arrays, a fitting will be performed to create a callable function for subsequent calculations. 
        By default, the proton potentials will be fitted to an 6th-order polynomial. 
        The 2D array should have shape (N, 2), the first row is the proton position in Angstrom, the second row is the potential energy in eV. 

        If these inputs are functions, they must only take one argument, which is the proton position in Angstrom. 
        The unit of the returned proton potentials should be eV. 
        """
        if callable(ReacProtonPot):
            self.ReacProtonPot = ReacProtonPot
        elif isarray(ReacProtonPot) and len(ReacProtonPot) == 2:
            r = ReacProtonPot[0]
            pot = ReacProtonPot[1]
            rmin1 = np.min(r)
            rmax1 = np.max(r)
            if Smooth == 'fit_poly6':
                self.ReacProtonPot = fit_poly6(r, pot)
            elif Smooth == 'fit_poly8':
                self.ReacProtonPot = fit_poly8(r, pot)
            elif Smooth == 'bspline':
                self.ReacProtonPot = bspline(r, pot)
            else:
                raise ValueError("'Smooth' must be set to one of the followings: 'fit_poly6', 'fit_poly8', or 'bspline'")
        else:
            raise TypeError("'ReacProtonPot' must be a 2D array with shape (N, 2) or a callable function")

        if callable(ProdProtonPot):
            self.ProdProtonPot = ProdProtonPot
        elif isarray(ProdProtonPot) and len(ProdProtonPot) == 2:
            r = ProdProtonPot[0]
            pot = ProdProtonPot[1]
            rmin2 = np.min(r)
            rmax2 = np.max(r)
            if Smooth == 'fit_poly6':
                self.ProdProtonPot = fit_poly6(r, pot)
            elif Smooth == 'fit_poly8':
                self.ProdProtonPot = fit_poly8(r, pot)
            elif Smooth == 'bspline':
                self.ProdProtonPot = bspline(r, pot)
            else:
                raise ValueError("'Smooth' must be set to one of the followings: 'fit_poly6', 'fit_poly8', or 'bspline'")
        else:
            raise TypeError("'ProdProtonPot' must be a 2D array with shape (N, 2) or a callable function")

        # determine the range of proton position for subsequent calculations
        if 'rmin' in kwargs.keys():
            rmin = kwargs['rmin']
        elif 'rmin1' in locals() and 'rmin2' in locals():
            rmin = np.min([rmin1, rmin2])
        else:
            rmin = -0.8
        
        if 'rmax' in kwargs.keys():
            rmax = kwargs['rmax']
        elif 'rmax1' in locals() and 'rmax2' in locals():
            rmax = np.max([rmax1, rmax2])
        else:
            rmax = 0.8
        self.rp = np.linspace(rmin, rmax, NGridPot)

        self.DeltaG = DeltaG
        self.Lambda = Lambda
        self.Vel = Vel
        self.NStates = NStates

        # Create the matrices and vectors used for calculation
        # Pu: Boltzmann distribution of proton states on the reactant side
        # Suv: overlap matrix of proton vibrational wave functions associated with the reactant and product states
        # dGuv: reaction free energy between vibronic states u and v
        # kuv: contribution of the (u,v) pair to the total PCET rate constant 
        # total rate constants is \sum_u\sum_v kuv 
        self.Pu = np.zeros(NStates)
        self.Suv = np.zeros((NStates, NStates))
        self.dGuv = np.zeros((NStates, NStates))
        self.kuv = np.zeros((NStates, NStates))


    def calc_proton_vibrational_states(self, mass=massH):
        """
        This function calculates the proton vibrational states (energies and wave functions)
        for proton moving in ReacProtonPot and ProdProtonPot respectively. 
        The FGH_1D code implemented by Maxim Secor is used
        """
        # calculate proton potentials on a grid, self.rp
        # the FGH code requires atomic units, so the units are converted
        rp_in_Bohr = self.rp*A2Bohr
        ngrid = len(rp_in_Bohr)
        sgrid = rp_in_Bohr[-1] - rp_in_Bohr[0]
        dx = sgrid/(ngrid-1)
        E_reac_in_Ha = self.ReacProtonPot(self.rp)*eV2Ha
        E_prod_in_Ha = self.ProdProtonPot(self.rp)*eV2Ha

        # record the mass used in this calculation
        # If a calculation is repeated for the same proton potential and the same mass, 
        # we will used the stored results from the previous calculation to save time
        self.MassUsedPreviously = mass

        # calculate the proton vibrational energies and wave fucntions for the reactant
        eigvals_reac, eigvecs_reac = FGH_1D(ngrid, sgrid, E_reac_in_Ha, mass)
        self.ReacProtonEnergyLevels = eigvals_reac[:self.NStates]*Ha2eV

        # the output wave functions are normalized such that \sum_i \Psi_i^2 = 1 where i is the index of grid points
        # the correct normalization is that \int \Psi^2 dr = 1
        # the normalized wave functions has unit of A^-1/2
        unnormalized_wfcs_reac = np.transpose(eigvecs_reac)[:self.NStates]
        normalized_wfcs_reac = np.array([wfci/np.sqrt(simps(wfci*wfci, self.rp)) for wfci in unnormalized_wfcs_reac])
        self.ReacProtonWaveFunctions = normalized_wfcs_reac

        # calculate the proton vibrational energies and wave fucntions for the product
        eigvals_prod, eigvecs_prod = FGH_1D(ngrid, sgrid, E_prod_in_Ha, mass)
        self.ProdProtonEnergyLevels = eigvals_prod[:self.NStates]*Ha2eV

        unnormalized_wfcs_prod = np.transpose(eigvecs_prod)[:self.NStates]
        normalized_wfcs_prod = np.array([wfci/np.sqrt(simps(wfci*wfci, self.rp)) for wfci in unnormalized_wfcs_prod])
        self.ProdProtonWaveFunctions = normalized_wfcs_prod

    def calc_reactant_state_distributions(self, T=298):
        Boltzmann_factors = np.exp(-self.ReacProtonEnergyLevels/kB/T)
        partition_func = np.sum(Boltzmann_factors)
        self.Pu = Boltzmann_factors/partition_func
        return self.Pu

    def calc_proton_overlap_matrix(self):
        for u in range(self.NStates):
            for v in range(self.NStates):
                self.Suv[u,v] = simps(self.ReacProtonWaveFunctions[u]*self.ProdProtonWaveFunctions[v], self.rp)
        return self.Suv

    def calc_reaction_free_energy_matrix(self):
        for u in range(self.NStates):
            for v in range(self.NStates):
                self.dGuv[u,v] = self.DeltaG + (self.ProdProtonEnergyLevels[v] - self.ProdProtonEnergyLevels[0]) - (self.ReacProtonEnergyLevels[u] - self.ReacProtonEnergyLevels[0])
        return self.dGuv

    def calc_kinetic_contribution_matrix(self, T=298):
        k0 = 2*np.pi/hbar*self.Vel*self.Vel
        self.Iuv = 1/np.sqrt(4*np.pi*self.Lambda*kB*T) * np.exp(-(self.dGuv + self.Lambda)**2/(4*self.Lambda*kB*T))
        self.kuv = k0*np.matmul(np.diag(self.Pu), self.Suv*self.Suv*self.Iuv)
        return self.kuv

    def calculate(self, mass=massH, T=298, reuse_saved_proton_states=False):
        # In certain occasions, we want to calculate the rate constant multiple times using the same proton potentials but different DeltaG, Lambda, or Vel parameters. 
        # Since the proton vibrational states and the overlap between the proton vibrational wave functions only depend on the proton potentials and the mass of the particle, 
        # we can reuse the stored proton vibrational states from the previous calculation for the new calculation, which will significantly speed up the calculation. 
        # This can be done by setting reuse_saved_proton_states=True. However, this feature should be used with caution. 
        if not (hasattr(self, 'ReacProtonEnergyLevels') and hasattr(self, 'ProdProtonEnergyLevels') and hasattr(self, 'ReacProtonWaveFunctions') and hasattr(self, 'ProdProtonWaveFunctions') and hasattr(self, 'MassUsedPreviously')):
            # No stored proton states found, new calculations are needed
            reuse_saved_proton_states = False
        elif self.MassUsedPreviously != mass:
            # The mass of the particle has been changed, new calculations are needed
            reuse_saved_proton_states = False

        if not reuse_saved_proton_states:
            self.calc_proton_vibrational_states(mass)
            self.calc_proton_overlap_matrix()

        self.calc_reactant_state_distributions(T)
        self.calc_reaction_free_energy_matrix()
        self.calc_kinetic_contribution_matrix(T)

        self.k_tot = np.sum(self.kuv)
        return self.k_tot
        
    def set_parameters(self, **kwargs):
        """
        reset DeltaG, Lambda, and Vel parameres, this is useful for electrochemical PCET, 
        where integration over electronic states with different energy in the electrode is needed
        """
        if 'DeltaG' in kwargs.keys():
            self.DeltaG = kwargs['DeltaG']
        if 'Lambda' in kwargs.keys():
            self.Lambda = kwargs['Lambda']
        if 'Vel' in kwargs.keys():
            self.Vel = kwargs['Vel']

    def get_reactant_proton_states(self):
        """
        returns proton vibrational energy levels and wave functions of the reactant state
        """
        return self.ReacProtonEnergyLevels, self.ReacProtonWaveFunctions

    def get_product_proton_states(self):
        """
        returns proton vibrational energy levels and wave functions of the product state
        """
        return self.ProdProtonEnergyLevels, self.ProdProtonWaveFunctions

    def get_reactant_state_distributions(self):
        return self.Pu

    def get_proton_overlap_matrix(self):
        return self.Suv

    def get_reaction_free_energy_matrix(self):
        return self.dGuv

    def get_activation_free_energy_matrix(self):
        return (self.dGuv + self.Lambda)**2/(4*self.Lambda)

    def get_kinetic_contribution_matrix(self):
        return self.kuv

    def get_total_rate_constant(self):
        return self.k_tot

