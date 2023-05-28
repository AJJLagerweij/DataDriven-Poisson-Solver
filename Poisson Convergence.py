r"""
Comparing minimization cost function and error to exact solution for validation.

The problem solved is the generic Poisson equation:

.. math::
    1 \nabla^2 u = 0 \quad 0 \leq x \leq 1000\\

    u(0) = 0\\

    u(1) = 0

Now the objective of this script is to compare the cost function to the quality of the solution. Whith quality of the
solution we mean:

.. math::
     e = \sum_{d=0}^D H^0(u_d^\text{sim}-u^\text{exact})

Bram van der Heijden
Mechanics of Composites for Energy and Mobility
KAUST
2023
"""

# Importing required modules.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import partial

# Importing my own scripts.
from configuration import Configuration
from problem import Hat
from test import Laplace_Dirichlet_Dirichlet
from patch import PatchDatabase
from constitutive import LinearMaterial

# Setup basic plotting properties.
plt.close('all')


# The right-hand side skeleton equation.
def rhs_hats(hats, x):
    gx = np.zeros_like(x)  # Initialize rhs values

    # For each hat in hats we set the appropriate values.
    for hat in hats:
        a, b, value = hat
        index = (a <= x) & (x <= b)
        gx[index] = value
    return gx


if __name__ == "__main__":
    # Problem definition.
    problem_length = 1000.  # Length of the problem in mm.
    problem_h = 200.  # Width of the hat function in mm.
    problem_rhs = 0.2  # Right hand side heating in W / mm.
    problem_a = 0.  # Left boundary value in degreeC.
    problem_b = -5.  # Right boundary value in degreeC.
    domain_num = 2  # Amount subdomains.
    domain_length = 525  # Length of the subdomains in mm.
    problem = Hat(problem_length, problem_h, problem_rhs, problem_a, problem_b, domain_length, domain_num)
    # problem.plot()

    # Material definition, required for the test, and verification of the exact solution.
    material = LinearMaterial(1500)  # Constant conductivity in W mm / degC, this value will not change the result.

    # Perform test according to the following test matrix.
    specimen_length = 1000  # Specimen length in mm.
    specimen_dx = 0.1  # mm discretization step size (measurement spacial resolution)
    rhs = partial(rhs_hats, [(400, 600, problem_rhs)])  # rhs in test setup.
    test = Laplace_Dirichlet_Dirichlet(specimen_length, specimen_dx, 0., 0., rhs, material)

    # Create empty database and add test to it.
    database = PatchDatabase()
    database.add_test(test)
    database.mirror()
    test.plot()

    # Either create a configurations-database from patch admissibility or from loading previous simulation results.
    x = np.linspace(0, problem_length, 1001)  # Spatial discretization in mm.

    # Bad initial guess, and verifying how J0Omega converges.
    configuration = Configuration(problem, database)  # From patch admissibility.
    configuration.rbd = np.array([[0, -1.2], [1.199, 0]])  # Create bad initial guess.
    configuration.optimize(x, material=material, verbose=False, order='Omega0')
    intermediate_J0Omega = np.array(configuration._intermediate_results)
    intermediate_J0Omega = pd.DataFrame({'cost': intermediate_J0Omega[:, 0], 'error': intermediate_J0Omega[:, 1],
                                         'rot1': intermediate_J0Omega[:, 3]/domain_length,
                                         'rot2': intermediate_J0Omega[:, 4]/domain_length})
    intermediate_J0Omega.to_csv("Intermediate_J0Omega.csv")

    # Bad initial guess, and verifying how J1Omega converges.
    configuration = Configuration(problem, database)  # From patch admissibility.
    configuration.rbd = np.array([[0, -1.2], [1.199, 0]])  # Create bad initial guess.
    configuration.optimize(x, material=material, verbose=False, order='Omega1')
    intermediate_J1Omega = np.array(configuration._intermediate_results)
    intermediate_J1Omega = pd.DataFrame({'cost': intermediate_J1Omega[:, 0], 'error': intermediate_J1Omega[:, 1],
                                         'rot1': intermediate_J1Omega[:, 3]/domain_length,
                                         'rot2': intermediate_J1Omega[:, 4]/domain_length})
    intermediate_J1Omega.to_csv("Intermediate_J1Omega.csv")

    # Bad initial guess, and verifying how J0Gamma converges.
    configuration = Configuration(problem, database)  # From patch admissibility.
    configuration.rbd = np.array([[0, -1.2], [1.199, 0]])  # Create bad initial guess.
    configuration.optimize(x, material=material, verbose=False, order='Gamma0')
    intermediate_J0Gamma = np.array(configuration._intermediate_results)
    intermediate_J0Gamma = pd.DataFrame({'cost': intermediate_J0Gamma[:, 0], 'error': intermediate_J0Gamma[:, 1],
                                         'rot1': intermediate_J0Gamma[:, 3]/domain_length,
                                         'rot2': intermediate_J0Gamma[:, 4]/domain_length})
    intermediate_J0Gamma.to_csv("Intermediate_J0Gamma.csv")

    # Bad initial guess, and verifying how J1Gamma converges.
    configuration = Configuration(problem, database)  # From patch admissibility.
    configuration.rbd = np.array([[0, -1.2], [1.199, 0]])  # Create bad initial guess.
    configuration.optimize(x, material=material, verbose=False, order='Gamma1')
    intermediate_J1Gamma = np.array(configuration._intermediate_results)
    intermediate_J1Gamma = pd.DataFrame({'cost': intermediate_J1Gamma[:, 0], 'error': intermediate_J1Gamma[:, 1],
                                         'rot1': intermediate_J1Gamma[:, 3]/domain_length,
                                         'rot2': intermediate_J1Gamma[:, 4]/domain_length})
    intermediate_J1Gamma.to_csv("Intermediate_J1Gamma.csv")

    # Bad initial guess, and verifying how J1Omega_weighted converges.
    configuration = Configuration(problem, database)  # From patch admissibility.
    configuration.rbd = np.array([[0, -1.2], [1.199, 0]])  # Create bad initial guess.
    configuration.optimize(x, material=material, verbose=False, order='Omega1_weights')
    intermediate_J1Omega_w = np.array(configuration._intermediate_results)
    intermediate_J1Omega_w = pd.DataFrame({'cost': intermediate_J1Omega_w[:, 0], 'error': intermediate_J1Omega_w[:, 1],
                                           'rot1': intermediate_J1Omega_w[:, 3]/domain_length,
                                           'rot2': intermediate_J1Omega_w[:, 4]/domain_length})
    intermediate_J1Omega_w.to_csv("Intermediate_J1Omega_weights.csv")

    # Plot convergence of Cost vs Error.
    fig = plt.figure()
    plt.loglog(intermediate_J0Omega['error'], intermediate_J0Omega['cost'], 's:', label=' $J^\Omega_0$', color='C1')
    plt.loglog(intermediate_J1Omega['error'], intermediate_J1Omega['cost'], 's:', label=' $J^\Omega_1$', color='C2')
    plt.loglog(intermediate_J0Gamma['error'], intermediate_J0Gamma['cost'], 's:', label=' $J^\Gamma_0$', color='C3')
    plt.loglog(intermediate_J1Gamma['error'], intermediate_J1Gamma['cost'], 's:', label=' $J^\Gamma_1$', color='C4')
    plt.loglog(intermediate_J1Omega_w['error'], intermediate_J1Omega_w['cost'], 's:', label=' $J^\Omega_w$', color='C5')
    plt.legend(loc='upper left', frameon=True)
    plt.xlabel('Error to Exact')
    plt.ylabel('Cost')

    # Create surface plots comparing cost vs error to exact.
    num = 11
    lim = 4
    rotation_domain1 = np.linspace(-lim, 0, num=num)
    rotation_domain2 = np.linspace(0, lim, num=num)
    costJ0Omega = np.zeros((num, num))
    costJ1Omega = np.zeros((num, num))
    costJ0Gamma = np.zeros((num, num))
    costJ1Gamma = np.zeros((num, num))
    costJ1Omega_w = np.zeros((num, num))
    error = np.zeros((num, num))
    monotonic = np.full((num, num), True)

    for i, rot1 in enumerate(rotation_domain1):
        for j, rot2 in enumerate(rotation_domain2):
            configuration = Configuration(problem, database)
            configuration.rbd = np.array([[0, rot1], [rot2, 0]])
            costJ0Omega[i, j] = configuration.error(x, order='Omega0')
            costJ1Omega[i, j] = configuration.error(x, order='Omega1')
            costJ0Gamma[i, j] = configuration.error(x, order='Gamma0')
            costJ1Gamma[i, j] = configuration.error(x, order='Gamma1')
            costJ1Omega_w[i, j] = configuration.error(x, order='Omega1_weights')
            error[i, j] = configuration.compare_to_exact(x, material)

    rotation_domain1 = rotation_domain1 / domain_length
    rotation_domain2 = rotation_domain2 / domain_length

    # Plot the error surfaces.
    fig, axs = plt.subplots(2, 3, layout="constrained")

    # Cost J^Omega_0
    costH0Omega_surface = axs[0, 0].contourf(rotation_domain1, rotation_domain2, costJ0Omega)
    axs[0, 0].set_ylabel("Tip displacement domain 1")
    axs[0, 0].set_xlabel("Tip displacement domain 2")
    fig.colorbar(costH0Omega_surface, ax=axs[0, 0], label="$J^\Omega_0$")
    # axs[0, 0].plot(intermediate_J0Omega['rot1'], intermediate_J0Omega['rot2'], 's:', color='C1')

    # Cost J^Omega_1
    costH1Omega_surface = axs[0, 1].contourf(rotation_domain1, rotation_domain2, costJ1Omega)
    axs[0, 1].set_ylabel("Tip displacement domain 1")
    axs[0, 1].set_xlabel("Tip displacement domain 2")
    fig.colorbar(costH1Omega_surface, ax=axs[0, 1], label="$J^\Omega_1$")
    # axs[0, 1].plot(intermediate_J1Omega['rot1'], intermediate_J1Omega['rot2'], 's:', color='C2')

    # Cost J^Gamma_0
    costH0Gamma_surface = axs[0, 2].contourf(rotation_domain1, rotation_domain2, costJ0Gamma)
    axs[0, 2].set_ylabel("Tip displacement domain 1")
    axs[0, 2].set_xlabel("Tip displacement domain 2")
    fig.colorbar(costH0Gamma_surface, ax=axs[0, 2], label="$J^\Gamma_0")
    # axs[0, 2].plot(intermediate_J0Gamma['rot1'], intermediate_J0Gamma['rot2'], 's:', color='C3')

    # Cost J^Gamma_1
    costH1Gamma_surface = axs[1, 0].contourf(rotation_domain1, rotation_domain2, costJ1Gamma)
    axs[1, 0].set_ylabel("Tip displacement domain 1")
    axs[1, 0].set_xlabel("Tip displacement domain 2")
    fig.colorbar(costH1Gamma_surface, ax=axs[1, 0], label="$J^\Gamma_1$")
    # axs[1, 0].plot(intermediate_J1Gamma['rot1'], intermediate_J1Gamma['rot2'], 's:', color='C4')

    # Cost J^Omega_1_weighted
    costH1Omega_w_surface = axs[1, 1].contourf(rotation_domain1, rotation_domain2, costJ1Omega_w)
    axs[1, 1].set_ylabel("Tip displacement domain 1")
    axs[1, 1].set_xlabel("Tip displacement domain 2")
    fig.colorbar(costH1Omega_w_surface, ax=axs[1, 1], label="$J^\Omega_w$")
    # axs[1, 1].plot(intermediate_J1Omega_w['rot1'], intermediate_J1Omega_w['rot2'], 's:', color='C5')

    # Error surface.
    error_surface = axs[1, 2].contourf(rotation_domain1, rotation_domain2, error)
    axs[1, 2].set_ylabel("Tip displacement domain 1")
    axs[1, 2].set_xlabel("Tip displacement domain 2")
    fig.colorbar(error_surface, ax=axs[1, 2], label="Error to Exact")

    # Store the surface plots.
    grid_rot1, grid_rot2 = np.meshgrid(rotation_domain1, rotation_domain2)
    error_surfaces = pd.DataFrame({'rot1': grid_rot1.flatten(), 'rot2': grid_rot2.flatten(),
                                   'cost0Omega': costJ0Omega.flatten(), 'cost1Omega': costJ1Omega.flatten(),
                                   'cost0Gamma': costJ0Gamma.flatten(), 'cost1Gamma': costJ1Gamma.flatten(),
                                   'cost1Omega_weights': costJ1Omega_w.flatten(), 'error': error.flatten()})
    error_surfaces.to_csv("Error_surfaces_Poisson.csv")
