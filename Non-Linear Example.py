r"""
The potentially non-linear Poisson example.

The problem solved is the non-linear Poisson equation:

.. math::
    \div( K [\nabla u]) = \tilde{g}(x) \quad 0 \leq x \leq 1000\\

    u(0) = 0\\

    u(1000) = 0

which in this case has a linear conductivity :math:`K=1\nabla u`, and could be solved with the linear non-homogeneous
solver available in the ODE solver branch of this git. Nevertheless, here it is assumed that the conductivity is not a
constant, and that it is not known that the actual material is linear.

Heating is applied as a hat function.

.. math::
    g(x) =
    \begin{cases}
        0 & 0 \leq x < 400
        1 & 400 \leq x \leq 600
        0 & 600 < x \leq 1000
    \end{cases}

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

Bram Lagerweij
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
from configuration import ConfigurationDatabase
from problem import Hat
from test import Laplace_Dirichlet_Dirichlet
from patch import PatchDatabase
from constitutive import LinearMaterial, Hardening, Softening

# Setup basic plotting properties.
plt.close('all')


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
    parallel = True

    # Problem definition.
    problem_length = 1.  # Length of the problem in mm.
    problem_h = 0.2  # Width of the hat function in mm.
    problem_rhs = 1.  # Right hand side heating in W / mm.
    problem_a = 0.00  # Left boundary value in degreeC.
    problem_b = -0.05  # Right boundary value in degreeC.
    # domain_num = 2  # Amount subdomains.
    # domain_length = 0.525  # Length of the subdomains in mm.
    # domain_num = 4
    # domain_length = 0.2875
    domain_num = 8
    domain_length = 0.16875
    problem = Hat(problem_length, problem_h, problem_rhs, problem_a, problem_b, domain_length, domain_num)

    # Material definition, required for the test, and verification of the exact solution.
    material = Softening(250, 1)  # Hardening relation between temperature gradient and heat flux.

    # Create empty database.
    database = PatchDatabase()

    # Perform test according to the following test matrix.
    specimen_length = [1.]  # Specimen length in mm.
    rhs_list = [
                # partial(rhs_hats, [(0.4, 0.6, 1.)]),  # Exactly the problem, and thus also the exact solution.
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
    specimen_dx = 0.001  # mm discretization step size (measurement spacial resolution)
    for length in specimen_length:
        for rhs in rhs_list:
            test = Laplace_Dirichlet_Dirichlet(length, specimen_dx, 0, 0, rhs, material)
            database.add_test(test)

    # Plot the resulting database, if required one can rotate or mirror here.
    print("\nNumber of patches", database.num_patches())
    database.plot()

    # Either create a configurations-database from patch admissibility or from loading previous simulation results.
    name = f'Convergence d {domain_num} p {database.num_patches()}'
    configurations = ConfigurationDatabase.create_from_problem_patches(problem, database)  # From patch admissibility.
    # configurations = ConfigurationDatabase.create_from_load(f'{name}.pkl.gz')  # Load previous simulation results.

    # Configurations are evaluated over at the following locations.
    x = np.linspace(0, problem_length, 1001)

    # Perform calculations on the database.
    print(f'{configurations.num_configurations()} are in this database')
    configurations.optimize(x, order='Omega1', parallel=parallel)
    configurations.compare_to_exact(x, material, parallel=parallel)
    configurations.database.plot.scatter('cost', 'error_to_exact')
    configurations.save(f'{name}.pkl.gz')

    # Get the best configuration in DD-cost.
    configurations.sort('cost')
    config = configurations.database.iloc[0, 0]
    config.plot(x, material=material)

    # # Exporting the exact solution.
    # exact_x, exact_u, exact_rhs = problem.exact(x, material)
    # exact = {'x': exact_x, 'u': exact_u, 'rhs': exact_rhs}
    # exact = pd.DataFrame(exact)
    # exact.to_csv(f'ExactSolution.csv')

    # Exporting the solution for plotting.
    x = np.linspace(0, problem_length, 1001)
    print('e= ', config.cost(x))
    u = config.domain_primal(x)
    rhs = config.domain_rhs(x)
    collection = {'x': x}
    for d in range(domain_num):
        collection[f"u{d + 1}"] = u[d]
        collection[f"rhs{d + 1}"] = rhs[d]
    data = pd.DataFrame(collection)
    data.to_csv(f"{name}.csv")

    plt.show()
