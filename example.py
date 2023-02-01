r"""
The beam example.

The problem solved is the generic Poisson equation:

.. math::
    \nabla^2 u = g(x) \quad 0 \leq x \leq 1\\

    u(0) = 0\\

    u(1) = 0

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
    u^{exact} = \text{argmin}_{a_d, b_d} \sum_{a=1}^D \sum_{b>a}^D \int_{\Omega_a\cap\Omega_b} \| u_a - u_b \|^2 dx

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

# Importing my own scripts.
from configuration import ConfigurationDatabase
from problem import Hat
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
    # Run settings.
    parallel = False

    # Problem definition.
    problem_length = 1.
    problem_h = 0.2
    domain_num = 4
    domain_length = 0.2875  # Length of the subdomains
    problem = Hat(problem_length, problem_h, domain_length, domain_num)

    # Locations for the error and error computations and plots.
    x = np.linspace(0, 1, 1001)

    # Material definition.
    material = LinearMaterial(1)

    # Create empty database.
    database = PatchDatabase()

    # Perform test according to the following test matrix.
    specimen_length = [1]  # specimen length.
    rhs_list = [
                # partial(rhs_hats, [(0.4, 0.6, 1.0)]),  # Exact solution
                partial(rhs_hats, [(0.30, 0.50,  1.0), (0.7, 1.0, -1.0)]),  # Test that contains the particular parts.
                ]  # Potential rhs equations

    # Perform the testing and add the result to the database.
    specimen_dx = x[1]  # mm discretization step size (measurement spacial resolution)
    for length in specimen_length:
        for rhs in rhs_list:
            test = Laplace_Dirichlet_Dirichlet(specimen_length, specimen_dx, 0, 0, rhs, material)
            database.add_test(test)
            test.plot()

    # Plot the resulting database, if required one can rotate or mirror here.
    print("\nNumber of patches", database.num_patches())

    # Either create a configurations-database from patch admissibility or from loading previous simulation results.
    name = f'Hat-Simulation #d {domain_num} #p {database.num_patches()} overlap 0.05'
    configurations = ConfigurationDatabase.create_from_problem_patches(problem, database)  # From patch admissibility.
    # configurations = ConfigurationDatabase.create_from_load(f'{name}.pkl.gz')  # Load previous simulation results.

    # Perform calculations on the database.
    print(f'Number of configurations {configurations.num_configurations()}')

    # There is only a single configruation, hence we just optimize that one.
    config = configurations.database.iloc[0, 0]
    config.plot(x, material=material)
    config.optimize(x)
    config.plot(x, material=material)
    plt.show()
