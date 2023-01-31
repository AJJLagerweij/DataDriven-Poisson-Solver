r"""
The beam example.

The problem solved for is a 1000 mm long simply supported beam. The supports are located as 0 and 250 mm. A load of
1 N is applied at the end of the beam. It is discretized into 3 domains, each 400 mm long with 100 mm overlap.

The database consists of patches obtained from a several simply supported beams. The patches were harvested are 400mm
long and had a 100 mm overlap. The exact solution is not in the database.

Bram Lagerweij
Mechanics of Composites for Energy and Mobility
KAUST
2021
"""

# Importing required modules.
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


# The right-hand side skeleton equation.
def rhs(x_a, x_b, x):
    r"""
    A generic hat function.
    
    .. math::
        g(x) =
        \begin{cases}
            0 & 0 \leq x < x_a \\
            1 & x_a \leq x x_b \\
            0 & x_b < x \leq L
        \end{cases}
    
    Parameters
    ----------
    x : array
        The location for which the exact solution needs to be found.
    x_a : float
        Lower :math:`x` coordinate for which the rhs has a step change from zero up to 1.
    x_b : float
        Upper :math:`x` coordinate for which the rhs has a step change from 1 back to 0.

    Returns
    -------
    array
        The right side equation for the requested values of :math:`x`.
    """
    gx = np.zeros_like(x)  # Initialize rhs to be zero.
    index = np.where((x_a <= x) & (x <= x_b))  # Find where the load is applied.
    gx[index] = 1  # Set the value to the load at these values.

    return gx


if __name__ == "__main__":
    # Run settings.
    parallel = True

    # Problem definition.
    problem_length = 1.
    problem_h = 0.2
    domain_num = 4
    domain_length = 0.30  # Length of the subdomains
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
                # partial(rhs, 0.60, 1.00),
                # partial(rhs, 0.40, 1.00),
                # partial(rhs, 0.00, 0.40),
                # partial(rhs, 0.00, 0.60),
                partial(rhs, 0.30, 0.50),
                partial(rhs, 0.50, 0.70),
                partial(rhs, 0.45, 0.55),
                partial(rhs, 0.35, 0.65)
                ]  # Potential rhs equations

    # Perform the testing and add the result to the database.
    specimen_dx = x[1]  # mm discretization step size (measurement spacial resolution)
    for length in specimen_length:
        for rhs in rhs_list:
            test = Laplace_Dirichlet_Dirichlet(specimen_length, specimen_dx, 0, 0, rhs, material)
            database.add_test(test)

    # Plot the resulting database, if required one can rotate or mirror here.
    print("\nNumber of patches", database.num_patches())
    database.plot()

    # Either create a configurations-database from patch admissibility or from loading previous simulation results.
    name = f'Hat-Simulation #d {domain_num} #p {database.num_patches()}'
    configurations = ConfigurationDatabase.create_from_problem_patches(problem, database)  # From patch admissibility.
    # configurations = ConfigurationDatabase.create_from_load(f'{name}.pkl.gz')  # Load previous simulation results.

    # Perform calculations on the database.
    print(f'{configurations.num_configurations()} are in this database')
    # configurations.optimize(x, parallel=parallel)
    # configurations.compare_to_exact(x, material)
    # configurations.save(f'{name}.pkl.gz')

    # # Obtain the best configuration.
    # configurations.sort('error')
    config = configurations.database.iloc[0, 0]
    config.plot(x, material=material)
    #
    # configurations.sort('error_to_exact')
    # config = configurations.database.iloc[0, 0]
    # config.plot(x, material=material)
