r"""
The beam example.

The problem solved is the generic Poisson equation:

.. math::
    \nabla^2 u = g(x) \quad 0 \leq x \leq 1\\

    u(0) = 0\\

    u(1) = -0.05

In this example the particular problem consists of a hat function.

.. math::
    g(x) =
    \begin{cases}
        0 & 0 \leq x < 0.4
        1 & 0.4 \leq x \leq 0.6
        0 & 0.6 < x \leq 1
    \end{cases}

Our dataset does not contain the solution to this problem. However, it contains the solution for a similar particular
solution. With Frankenstein's algorithm, we cut the solution in the database in parts and reassemble it such that it
solves for the particular problem, that is a patch that satisfies :math:`\nabla^2 u_{p_d}(x) = g(x) \qquad \forall
x\in\Omega_d` only within that subdomain. This however does not solve the problem entirely, as the homogeneous
solution is still to be found. We know that a homogeneous solution of Poisson problems can be found in the following
functional space:

.. math::
    h_d(x) = a_d x + b_d

where :math:`a_d` and :math:`b_d` are unknown constants.

Now the solution space of each subdomain can be written as:

.. math::
    u_d(x) = u_{p_d}(x) + a_d x + b_d

Where a boundary conditions constrain certain :math:`a_d` and `b_d` variables.

Now the full solution con be found by finding the other variables through a minimization of the overlapping domain
decomposition cost function.

.. math::
    u^{exact} = \text{argmin}_{a_d, b_d} \sum_{i=1}^D \sum_{j>i}^D \int_{\Omega_i\cap\Omega_j} \| u_i - u_j \|^2 dx

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
from problem import Hat, Homogeneous
from test import Laplace_Dirichlet_Dirichlet
from patch import PatchDatabase
from constitutive import LinearMaterial

# Setup basic plotting properties.
plt.close('all')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['backend'] = 'Qt5agg'


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
    problem_length = 1.
    problem_h = 0.2
    domain_num = 4
    a = 0.
    b = -0.05
    # domain_length = 0.525
    domain_length = 0.2875  # Length of the subdomains
    # domain_length = 0.16875
    problem = Hat(problem_length, problem_h, a, b, domain_length, domain_num)

    # Material definition.
    material = Softening(250, 1)

    # Create empty database.
    database = PatchDatabase()

    # Perform test according to the following test matrix.
    specimen_length = [1]  # specimen length.
    rhs_list = [
                # partial(rhs_hats, [(0.40, 0.60, 1.00)]),  # Exactly the problem, and thus also the exact solution.
                partial(rhs_hats, [(0.00, 0.60, 1.00)]),  # Smallest, small, medium, large and largest database.
                partial(rhs_hats, [(0.00, 0.40, 1.00)]),  # Smallest, small, medium, large and largest database.
                partial(rhs_hats, [(0.40, 1.00, 1.00)]),  # Smallest, small, medium, large and largest database.
                partial(rhs_hats, [(0.60, 1.00, 1.00)]),  # Smallest, small, medium, large and largest database.
                partial(rhs_hats, [(0.30, 0.50, 1.00)]),  # Small, medium, large and largest database.
                partial(rhs_hats, [(0.50, 0.70, 1.00)]),  # Small, medium, large and largest database.
                # partial(rhs_hats, [(0.30, 0.70, 0.50)]),  # Medium, large and largest database.
                # partial(rhs_hats, [(0.45, 0.55, 2.00)]),  # Medium, large and largest database.
                # partial(rhs_hats, [(0.00, 0.40, 1.00), (0.80, 1.00, 1.00)]),  # Large and largest database.
                # partial(rhs_hats, [(0.00, 0.40, 1.00), (0.80, 1.00, 0.50)]),  # Large and largest database.
                # partial(rhs_hats, [(0.00, 0.20, 1.00), (0.40, 0.60, 1.00), (0.80, 1.00, 1.00)]),  # Largest database.
                # partial(rhs_hats, [(0.00, 0.20, 0.50), (0.40, 0.60, 1.00), (0.80, 1.00, 0.50)]),  # Largest database.
                ]  # Potential rhs equations

    # Perform the testing and add the result to the database.
    x = np.linspace(0, 1, 101)
    specimen_dx = x[1]  # mm discretization step size (measurement spacial resolution)
    rhs = partial(rhs_hats, [(0.00, 0.025,  -1.0), (0.05, 0.25, -2.0), (0.70, 0.95,  1.0)])  # Test contains the particular parts.
    test = Laplace_Dirichlet_Dirichlet(specimen_length, specimen_dx, 0, 0, rhs, material)

    # Add the test to the database.
    database.add_test(test)
    database.mirror()
    print("\nNumber of patches", database.num_patches())
    database.plot()

    # Either create a configurations-database from patch admissibility or from loading previous simulation results.
    name = f'Hat-Simulation d {domain_num} p {database.num_patches()}'
    configurations = ConfigurationDatabase.create_from_problem_patches(problem, database)  # From patch admissibility.
    # configurations = ConfigurationDatabase.create_from_load(f'{name}.pkl.gz')  # Load previous simulation results.

    # Configurations are evaluated over at the following locations.
    x = np.linspace(0, 1, 101)

    # Perform calculations on the database.
    print(f'{configurations.num_configurations()} are in this database')
    configurations.optimize(x, parallel=parallel)
    configurations.compare_to_exact(x, material, parallel=parallel)
    configurations.save(f'{name}.pkl.gz')

    # Get the best configuration in DD-error.
    configurations.sort('error')
    config = configurations.database.iloc[0, 0]
    config.plot(x, material=material)

    # Get the best configuration in distance to the exact solution.
    configurations.sort('error_to_exact')
    config = configurations.database.iloc[0, 0]
    config.plot(x, material=material)

    # Compare the two error norms.
    configurations.sort('error')
    configurations.database.plot.scatter('error', 'error_to_exact')
    plt.show()
