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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Importing my own scripts.
from configuration import Configuration
from problem import Homogeneous
from test import Laplace_Dirichlet_Dirichlet
from patch import PatchDatabase
from constitutive import LinearMaterial

# Setup basic plotting properties.
plt.close('all')


if __name__ == "__main__":
    # Problem definition.
    problem_length = 1.  # Length of the problem in mm.
    problem_a = 0.  # Left boundary value in degreeC.
    problem_b = 0.  # Right boundary value in degreeC.
    domain_num = 2  # Amount subdomains.
    domain_length = 0.525  # Length of the subdomains in mm.
    problem = Homogeneous(problem_length, problem_a, problem_b, domain_length, domain_num)
    # problem.plot()

    # Material definition, required for the test, and verification of the exact solution.
    material = LinearMaterial(1)  # Constant conductivity in W mm / degC

    # Perform test according to the following test matrix.
    specimen_length = 1.  # Specimen length in mm.
    specimen_dx = 0.001  # mm discretization step size (measurement spacial resolution)
    rhs = lambda x: 0.*x  # rhs in test setup.
    test = Laplace_Dirichlet_Dirichlet(specimen_length, specimen_dx, 0., 0., rhs, material)

    # Create empty database and add test to it.
    database = PatchDatabase()
    database.add_test(test)
    database.mirror()
    # test.plot()

    # Either create a configurations-database from patch admissibility or from loading previous simulation results.
    x = np.linspace(0, problem_length, 1001)  # Spatial discretization in mm.

    # Two configurations for J0.
    configuration = Configuration(problem, database)
    configuration.rbd = np.array([[0, -0.6], [0.6, 0]])  # Small cost, far away from the solution.
    configuration.plot(x, material=material)
    cost1 = configuration.error(x, order='Omega0')
    error1 = configuration.compare_to_exact(x, material)
    ud1 = configuration.domain_primal(x)

    configuration.rbd = np.array([[0, -0.5], [-0.5, 0]])  # Large cost, closer to the solution.
    configuration.plot(x, material=material)
    cost2 = configuration.error(x, order='Omega0')
    error2 = configuration.compare_to_exact(x, material)
    ud2 = configuration.domain_primal(x)

    J0Omega_configurations = {'x': x}
    for d in range(2):
        J0Omega_configurations[f'u{d}_1'] = ud1[d]
        J0Omega_configurations[f'u{d}_2'] = ud2[d]
    J0Omega_configurations = pd.DataFrame(J0Omega_configurations)
    J0Omega_configurations.to_csv('J0_configurations.csv')
    J0Omega_summary = pd.DataFrame({'cost_1': [cost1], 'error_1': [error1], 'cost_2': [cost2], 'error_2': [error2]})
    J0Omega_summary.to_csv('J0_summary.csv')

    # Two configurations for J1.
    configuration = Configuration(problem, database)
    configuration.rbd = np.array([[0, -0.5], [0.5, 0]])  # Small cost, far away from the solution.
    configuration.plot(x, material=material)
    cost1 = configuration.error(x, order='Omega1')
    error1 = configuration.compare_to_exact(x, material)
    ud1 = configuration.domain_primal(x)

    configuration.rbd = np.array([[0, -0.6], [-0.6, 0]])  # Large cost, closer to the solution.
    configuration.plot(x, material=material)
    cost2 = configuration.error(x, order='Omega1')
    error2 = configuration.compare_to_exact(x, material)
    ud2 = configuration.domain_primal(x)

    J1Omega_configurations = {'x': x}
    for d in range(2):
        J1Omega_configurations[f'u{d}_1'] = ud1[d]
        J1Omega_configurations[f'u{d}_2'] = ud2[d]
    J0Omega_configurations = pd.DataFrame(J1Omega_configurations)
    J0Omega_configurations.to_csv('J1_configurations.csv')
    J1Omega_summary = pd.DataFrame({'cost_1': [cost1], 'error_1': [error1], 'cost_2': [cost2], 'error_2': [error2]})
    J1Omega_summary.to_csv('J1_summary.csv')

    # Bad initial guess, and verifying how J0Omega converges.
    configuration = Configuration(problem, database)  # From patch admissibility.
    configuration.rbd = np.array([[0, -1.2], [1.199, 0]])  # Create bad initial guess.
    configuration.optimize(x, material=material, verbose=False, order='Omega0')
    intermediate_J0Omega = np.array(configuration._intermediate_results)
    intermediate_J0Omega = pd.DataFrame({'cost': intermediate_J0Omega[:, 0], 'error': intermediate_J0Omega[:, 1],
                                         'rot1': intermediate_J0Omega[:, 3], 'rot2': intermediate_J0Omega[:, 4]})
    intermediate_J0Omega.to_csv("Intermediate_J0Omega.csv")

    # Bad initial guess, and verifying how J1Omega converges.
    configuration = Configuration(problem, database)  # From patch admissibility.
    configuration.rbd = np.array([[0, -1.2], [1.199, 0]])  # Create bad initial guess.
    configuration.optimize(x, material=material, verbose=False, order='Omega1')
    intermediate_J1Omega = np.array(configuration._intermediate_results)
    intermediate_J1Omega = pd.DataFrame({'cost': intermediate_J1Omega[:, 0], 'error': intermediate_J1Omega[:, 1],
                                         'rot1': intermediate_J1Omega[:, 3], 'rot2': intermediate_J1Omega[:, 4]})
    intermediate_J1Omega.to_csv("Intermediate_J1Omega.csv")

    # Bad initial guess, and verifying how J0Gamma converges.
    configuration = Configuration(problem, database)  # From patch admissibility.
    configuration.rbd = np.array([[0, -1.2], [1.199, 0]])  # Create bad initial guess.
    configuration.optimize(x, material=material, verbose=False, order='Gamma0')
    intermediate_J0Gamma = np.array(configuration._intermediate_results)
    intermediate_J0Gamma = pd.DataFrame({'cost': intermediate_J0Gamma[:, 0], 'error': intermediate_J0Gamma[:, 1],
                                         'rot1': intermediate_J0Gamma[:, 3], 'rot2': intermediate_J0Gamma[:, 4]})
    intermediate_J0Gamma.to_csv("Intermediate_J0Gamma.csv")

    # Bad initial guess, and verifying how J1Gamma converges.
    configuration = Configuration(problem, database)  # From patch admissibility.
    configuration.rbd = np.array([[0, -1.2], [1.199, 0]])  # Create bad initial guess.
    configuration.optimize(x, material=material, verbose=False, order='Gamma1')
    intermediate_J1Gamma = np.array(configuration._intermediate_results)
    intermediate_J1Gamma = pd.DataFrame({'cost': intermediate_J1Gamma[:, 0], 'error': intermediate_J1Gamma[:, 1],
                                         'rot1': intermediate_J1Gamma[:, 3], 'rot2': intermediate_J1Gamma[:, 4]})
    intermediate_J1Gamma.to_csv("Intermediate_J1Gamma.csv")

    # Bad initial guess, and verifying how J1Omega_weighted converges.
    configuration = Configuration(problem, database)  # From patch admissibility.
    configuration.rbd = np.array([[0, -1.2], [1.199, 0]])  # Create bad initial guess.
    configuration.optimize(x, material=material, verbose=False, order='Omega1_weights')
    intermediate_J1Omega_w = np.array(configuration._intermediate_results)
    intermediate_J1Omega_w = pd.DataFrame({'cost': intermediate_J1Omega_w[:, 0], 'error': intermediate_J1Omega_w[:, 1],
                                           'rot1': intermediate_J1Omega_w[:, 3], 'rot2': intermediate_J1Omega_w[:, 4]})
    intermediate_J1Omega_w.to_csv("Intermediate_J1Omega_weights.csv")

    # Plot convergence of Cost vs Error.
    fig = plt.figure()
    plt.loglog(intermediate_J0Omega['error'], intermediate_J0Omega['cost'], 's:', label=' $J^\Omega_0$', color='C1')
    plt.loglog(intermediate_J1Omega['error'], intermediate_J1Omega['cost'], 's:', label=' $J^\Omega_1$', color='C2')
    plt.loglog(intermediate_J0Gamma['error'], intermediate_J0Gamma['cost'], 's:', label=' $J^\Gamma_0$', color='C3')
    plt.loglog(intermediate_J1Gamma['error'], intermediate_J1Gamma['cost'], 's:', label=' $J^\Gamma_1$', color='C4')
    plt.loglog(intermediate_J1Omega_w['error'], intermediate_J1Omega_w['cost'], 's:', label=' $J^\Omega_{(4,1)}$', color='C5')
    plt.legend(loc='upper left', frameon=True)
    plt.xlabel('Error to Exact')
    plt.ylabel('Cost')

    # Create surface plots comparing cost vs error to exact.
    num = 41
    lim = 1.25
    rotation_domain1 = np.linspace(-lim, lim, num=num)
    rotation_domain2 = np.linspace(-lim, lim, num=num)
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

    # Plot the error surfaces.
    fig, axs = plt.subplots(2, 3, layout="constrained")

    # Cost J^Omega_0
    costH0Omega_surface = axs[0, 0].contourf(rotation_domain1, rotation_domain2, costJ0Omega)
    axs[0, 0].set_ylabel("Rotation domain 1")
    axs[0, 0].set_xlabel("Rotation domain 2")
    fig.colorbar(costH0Omega_surface, ax=axs[0, 0], label="$J^\Omega_0$")
    axs[0, 0].plot(intermediate_J0Omega['rot1'], intermediate_J0Omega['rot2'], 's:', color='C1')

    # Cost J^Omega_1
    costH1Omega_surface = axs[0, 1].contourf(rotation_domain1, rotation_domain2, costJ1Omega)
    axs[0, 1].set_ylabel("Rotation domain 1")
    axs[0, 1].set_xlabel("Rotation domain 2")
    fig.colorbar(costH1Omega_surface, ax=axs[0, 1], label="$J^\Omega_1$")
    axs[0, 1].plot(intermediate_J1Omega['rot1'], intermediate_J1Omega['rot2'], 's:', color='C2')

    # Cost J^Gamma_0
    costH0Gamma_surface = axs[0, 2].contourf(rotation_domain1, rotation_domain2, costJ0Gamma)
    axs[0, 2].set_ylabel("Rotation domain 1")
    axs[0, 2].set_xlabel("Rotation domain 2")
    fig.colorbar(costH0Gamma_surface, ax=axs[0, 2], label="$J^\Gamma_0")
    axs[0, 2].plot(intermediate_J0Gamma['rot1'], intermediate_J0Gamma['rot2'], 's:', color='C3')

    # Cost J^Gamma_1
    costH1Gamma_surface = axs[1, 0].contourf(rotation_domain1, rotation_domain2, costJ1Gamma)
    axs[1, 0].set_ylabel("Rotation domain 1")
    axs[1, 0].set_xlabel("Rotation domain 2")
    fig.colorbar(costH1Gamma_surface, ax=axs[1, 0], label="$J^\Gamma_1$")
    axs[1, 0].plot(intermediate_J1Gamma['rot1'], intermediate_J1Gamma['rot2'], 's:', color='C4')

    # Cost J^Omega_1_weighted
    costH1Omega_w_surface = axs[1, 1].contourf(rotation_domain1, rotation_domain2, costJ1Omega_w)
    axs[1, 1].set_ylabel("Rotation domain 1")
    axs[1, 1].set_xlabel("Rotation domain 2")
    fig.colorbar(costH1Omega_w_surface, ax=axs[1, 1], label="$J^\Omega_(4,1)$")
    axs[1, 1].plot(intermediate_J1Omega_w['rot1'], intermediate_J1Omega_w['rot2'], 's:', color='C5')

    # Error surface.
    error_surface = axs[1, 2].contourf(rotation_domain1, rotation_domain2, error)
    axs[1, 2].set_ylabel("Rotation domain 1")
    axs[1, 2].set_xlabel("Rotation domain 2")
    fig.colorbar(error_surface, ax=axs[1, 2], label="Error to Exact")
    axs[1, 2].plot(intermediate_J0Omega['rot1'], intermediate_J0Omega['rot2'], 's:', color='C1')
    axs[1, 2].plot(intermediate_J1Omega['rot1'], intermediate_J1Omega['rot2'], 's:', color='C2')
    axs[1, 2].plot(intermediate_J0Gamma['rot1'], intermediate_J0Gamma['rot2'], 's:', color='C3')
    axs[1, 2].plot(intermediate_J1Gamma['rot1'], intermediate_J1Gamma['rot2'], 's:', color='C4')
    axs[1, 2].plot(intermediate_J1Omega_w['rot1'], intermediate_J1Omega_w['rot2'], 's:', color='C5')

    # Store the surface plots.
    grid_rot1, grid_rot2 = np.meshgrid(rotation_domain1, rotation_domain2)
    error_surfaces = pd.DataFrame({'rot1': grid_rot1.flatten(), 'rot2': grid_rot2.flatten(),
                                   'cost0Omega': costJ0Omega.flatten(), 'cost1Omega': costJ1Omega.flatten(),
                                   'cost0Gamma': costJ0Gamma.flatten(), 'cost1Gamma': costJ1Gamma.flatten(),
                                   'cost1Omega_weights': costJ1Omega_w.flatten(), 'error': error.flatten()})
    error_surfaces.to_csv("Error_surfaces.csv")
