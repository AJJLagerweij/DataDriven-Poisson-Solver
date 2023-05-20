r"""
The beam example.

The problem solved is the generic Poisson equation:

.. math::
    1000 \nabla^2 u = g(x) \quad 0 \leq x \leq 1000\\

    u(0) = 0\\

    u(1) = -5

In this example the particular problem consists of a hat function.

.. math::
    g(x) =
    \begin{cases}
        0 & 0 \leq x < 400
        1 & 400 \leq x \leq 600
        0 & 600 < x \leq 1000
    \end{cases}

Our dataset does not contain the solution to this problem. However, it contains the solution for a similar particular
solution. With Frankenstein's algorithm, we cut the solution in the database in parts and reassemble it such that it
solves for the particular problem, that is a patch that satisfies :math:`k \nabla^2 u_{p_d}(x) = g(x) \qquad \forall
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
    problem_length = 1000.  # Length of the problem in mm.
    problem_h = 200.  # Width of the hat function in mm.
    problem_rhs = 0.2  # Right hand side heating in W / mm.
    problem_a = 0.00  # Left boundary value in degreeC.
    problem_b = -5.  # Right boundary value in degreeC.
    domain_num = 4  # 16 # Amount subdomains.
    domain_length = 287.5  # 109.375 # Length of the subdomains in mm.
    problem = Hat(problem_length, problem_h, problem_rhs, problem_a, problem_b, domain_length, domain_num)

    # Locations for the error and error computations and plots.
    x = np.linspace(0, problem_length, 10001)  # Location in mm.
    problem.plot()

    # Material definition, required for the test, and verification of the exact solution.
    material = LinearMaterial(1000)  # Constant conductivity in W mm / degC

    # Perform test according to the following test matrix.
    specimen_length = 750.  # Specimen length in mm.
    specimen_dx = x[1]  # mm discretization step size (measurement spacial resolution)
    rhs = partial(rhs_hats, [(0, 25,  -1.0e-1), (50, 200, -3.0e-1), (500, 700,  2.0e-1)])  # rhs in test setup.
    test = Laplace_Dirichlet_Dirichlet(specimen_length, specimen_dx, 0., 5., rhs, material)

    # Create empty database and add test to it.
    database = PatchDatabase()
    database.add_test(test)
    database.mirror()
    test.plot()

    # Either create a configurations-database from patch admissibility or from loading previous simulation results.
    configuration = Configuration(problem, database)  # From patch admissibility.
    configuration.plot(x, material=material)  # Configuration without optimization.
    configuration.optimize(x, material=material, verbose=True)  # Minimizing the cost function. Material used for error to exact.
    configuration.plot(x, material=material)
