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
from configuration import Configuration
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
    problem_length = 1.  # Length of the problem.
    problem_h = 0.2  # Width of the hat function.
    problem_a = 0.  # Left boundary value.
    problem_b = 0.  # Right boundary value.
    domain_num = 2  # Amount subdomains.
    domain_length = 0.525  # Length of the subdomains.
    problem = Homogeneous(problem_length, problem_a, problem_b, domain_length, domain_num)

    # Locations for the error and error computations and plots.
    x = np.linspace(0, 1, 1001)

    # Material definition, required for the test, and verification of the exact solution.
    material = LinearMaterial(1)

    # Create empty database.
    database = PatchDatabase()

    # Perform test according to the following test matrix.
    specimen_length = 1  # specimen length.
    specimen_dx = x[1]  # mm discretization step size (measurement spacial resolution)
    # rhs = partial(rhs_hats, [(0.00, 0.025,  -1.0), (0.05, 0.2, -2.0), (0.75, 0.95,  1.0)])  # Test contains the particular parts.
    rhs = partial(rhs_hats, [])
    test = Laplace_Dirichlet_Dirichlet(specimen_length, specimen_dx, 0, 0.05, rhs, material)

    # Add the test to the database.
    database.add_test(test)
    database.mirror()
    print("\nNumber of patches", database.num_patches())
    test.plot()

    # Either create a configurations-database from patch admissibility or from loading previous simulation results.
    configuration = Configuration(problem, database)  # From patch admissibility.
    configuration.plot(x, material=material)
    configuration.optimize(x, material=material, verbose=True)
    configuration.plot(x, material=material)

    # Look at the convergence, compare distance to exact solution and overlapping error.
    configuration._error_comparison = np.array(configuration._error_comparison)
    plt.figure()
    plt.xlabel("Overlapping Error $\mathcal{E}$")
    plt.ylabel("Distance to Exact Solution $e$")
    plt.loglog(configuration._error_comparison[:, 0], configuration._error_comparison[:, 1], 'o-', linewidth=0.25)
    plt.show()
