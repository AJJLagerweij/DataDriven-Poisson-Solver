r"""
The Laplace example.

The problem solved is the non-linear Laplace equation:

.. math::
    \div( K [\nabla u]) = 0 \quad 0 \leq x \leq 1000\\

    u(0) = 0\\

    u(1000) = 0

which in this case has a linear conductivity :math:`K=1\nabla u`, and could be solved with the linear non-homogeneous
solver available in the ODE solver branch of this git. Nevertheless, here it is assumed that the conductivity is not a
constant, and that it is not known that the actual material is linear.

Our dataset does not contain the solution to this problem. However, it contains the solution for a similar particular
solution. With Frankenstein's algorithm, we cut the solution in the database in parts and reassemble it. According to
appendix A of the dissertation, the following solutions for the domain should be considered:

.. math::
    u_d(x) = u_{p_d}(x-t_{d}) + \bar{u}_d

where :math:`b_d` is an unknown constant. And :math:`u_{p_d}(x)` is the displacement of a patch that satisfies:

.. math::
    g_{p_d}(x-t_{d}) = \tilde{g}(x) \quad \forall x\in\mathcal{D}_d

where :math:`t_d` is the translation of the patch to global coordinates. The unknowns in this problem are: the patch
for each domain (discrete), the coordinate translation :math:`t_d`, and rigid temperature addition :math:`\bar{u}_d`.
In contrast to the ODE solver one might not find the exact solution when minimizing the cost function, hence from all
configurations the best configuration needs to be selected. The best configuration is the one with the lowest cost.

Bram van der Heijden
Mechanics of Composites for Energy and Mobility
KAUST
2023
"""

# Importing required modules.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import pandas as pd

# Importing my own scripts.
from configuration import ConfigurationDatabase
from problem import Homogeneous
from test import Laplace_Dirichlet_Dirichlet
from patch import PatchDatabase
from constitutive import LinearMaterial

# Setup basic plotting properties.
plt.close('all')


def export_results(configurations, domain_length, problem_length, material):
    results = pd.DataFrame(columns=['rot1', 'rot2', 'J0Omega', 'J1Omega', 'J0Gmega', 'J1Gmega', 'J1Omega_w', 'error'])

    for num, configuration in enumerate(configurations.database['configuration']):
        results = pd.concat([results, configuration_details(configuration, domain_length, problem_length, material)])

    return results


def configuration_details(configuration, domain_length, problem_length, material):
    # Calculate slope domain 1 and domain two.
    x = np.array([domain_length, problem_length - domain_length])
    ud = configuration.domain_primal(x)
    slope1 = ud[0, 0]/domain_length
    slope2 = -ud[1, 1]/domain_length

    x = np.linspace(0, problem_length, 1001)
    J0Omega = configuration.error(x, order='Omega0')
    J1Omega = configuration.error(x, order='Omega1')
    J0Gamma = configuration.error(x, order='Gamma0')
    J1Gamma = configuration.error(x, order='Gamma1')
    J1Omega_w = configuration.error(x, order='Omega1_weights')
    error = configuration.compare_to_exact(x, material)
    results = pd.DataFrame({'rot1': [slope1], 'rot2': [slope2], 'J0Omega': [J0Omega], 'J1Omega': [J1Omega],
                            'J0Gamma': [J0Gamma], 'J1Gamma': [J1Gamma], 'J1Omega_w': [J1Omega_w], 'error': [error]})
    return results


if __name__ == "__main__":
    # Whether to use parallel optimization or not.
    parallel = False

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
    b_list = [0.55, 0.9523, 1.1429, 2.2857]

    # Create patch database by looping over all tests.
    database = PatchDatabase()
    for b in b_list:
        test = Laplace_Dirichlet_Dirichlet(specimen_length, specimen_dx, 0., b, rhs, material)
        database.add_test(test)
    database.mirror()
    database.plot()

    # Either create a configurations-database from patch admissibility or from loading previous simulation results.
    x = np.linspace(0, problem_length, 1001)  # Spatial discretization in mm.

    # Either create a configurations-database from patch admissibility or from loading previous simulation results.
    configurations = ConfigurationDatabase.create_from_problem_patches(problem, database)  # From patch admissibility.
    configurations.optimize(x, order='Omega0', parallel=parallel)
    configurations.compare_to_exact(x, material, parallel=parallel)
    configurations.sort('error')
    # configurations.plot(x, material, max_images=5)

    # plot and export results.
    results = export_results(configurations, domain_length, problem_length, material)
    results.to_csv('Scatter_cost_error.csv')

    # Plot the error pointss.
    fig, axs = plt.subplots(2, 3, layout="constrained", sharex=True, sharey=True)

    # Cost J^Omega_0
    costH0Omega_points = axs[0, 0].scatter(results.rot1, results.rot2, c=results.J0Gamma, lw=0)
    axs[0, 0].set_ylabel("Slope domain 1")
    axs[0, 0].set_xlabel("Slope domain 2")
    fig.colorbar(costH0Omega_points, ax=axs[0, 0], label="$J^\Omega_0$")

    # Cost J^Omega_1
    costH1Omega_points = axs[0, 1].scatter(results.rot1, results.rot2, c=results.J1Omega, lw=0)
    axs[0, 1].set_ylabel("Slope domain 1")
    axs[0, 1].set_xlabel("Slope domain 2")
    fig.colorbar(costH1Omega_points, ax=axs[0, 1], label="$J^\Omega_1$")

    # Cost J^Gamma_0
    costH0Gamma_points = axs[0, 2].scatter(results.rot1, results.rot2, c=results.J0Gamma, lw=0)
    axs[0, 2].set_ylabel("Slope domain 1")
    axs[0, 2].set_xlabel("Slope domain 2")
    fig.colorbar(costH0Gamma_points, ax=axs[0, 2], label="$J^\Gamma_0")

    # Cost J^Gamma_1
    costH1Gamma_points = axs[1, 0].scatter(results.rot1, results.rot2, c=results.J1Gamma, lw=0)
    axs[1, 0].set_ylabel("Slope domain 1")
    axs[1, 0].set_xlabel("Slope domain 2")
    fig.colorbar(costH1Gamma_points, ax=axs[1, 0], label="$J^\Gamma_1$")

    # Cost J^Omega_1_weighted
    costH1Omega_w_points = axs[1, 1].scatter(results.rot1, results.rot2, c=results.J1Omega_w, lw=0)
    axs[1, 1].set_ylabel("Slope domain 1")
    axs[1, 1].set_xlabel("Slope domain 2")
    fig.colorbar(costH1Omega_w_points, ax=axs[1, 1], label="$J^\Omega_(4,1)$")

    # Error points.
    error_points = axs[1, 2].scatter(results.rot1, results.rot2, c=results.error, lw=0)
    axs[1, 2].set_ylabel("Slope domain 1")
    axs[1, 2].set_xlabel("Slope domain 2")
    fig.colorbar(error_points, ax=axs[1, 2], label="$Error to exact$")

    # Look at the convergence, compare distance to exact solution and overlapping error.
    plt.figure()
    plt.xlabel("Cost $J$")
    plt.ylabel("Distance to Exact Solution $e$")
    plt.scatter(results.error, results.J0Omega, label='$J^\Omega_0$')
    plt.scatter(results.error, results.J1Omega, label='$J^\Omega_1$')
    plt.scatter(results.error, results.J0Gamma, label='$J^\Gamma_0$')
    plt.scatter(results.error, results.J1Gamma, label='$J^\Gamma_1$')
    plt.scatter(results.error, results.J1Omega_w, label='$J^\Omega_{4,1}$')
    plt.legend()
    plt.show()
