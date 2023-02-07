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
from constitutive import LinearMaterial, Hardening

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
    parallel = True

    # Problem definition.
    problem_length = 1.
    problem_h = 0.2
    domain_num = 4
    domain_length = 0.30  # Length of the subdomains
    problem = Hat(problem_length, problem_h, domain_length, domain_num)

    # Material definition.
    material = LinearMaterial(1)

    # Create empty database.
    database = PatchDatabase()

    # Perform test according to the following test matrix.
    specimen_length = [1]  # specimen length.
    rhs_list = [
                # partial(rhs_hats, [(0.40, 0.60, 1.00)]),  # Exactly the problem, and thus also the exact solution.
                partial(rhs_hats, [(0.00, 0.60, 1.00)]),  # Small, mid and large database.
                partial(rhs_hats, [(0.00, 0.40, 1.00)]),  # Small, mid and large database.
                partial(rhs_hats, [(0.40, 1.00, 1.00)]),  # Small, mid and large database.
                partial(rhs_hats, [(0.60, 1.00, 1.00)]),  # Small, mid and large database.
                partial(rhs_hats, [(0.30, 0.50, 1.00)]),  # Mid and large database.
                partial(rhs_hats, [(0.50, 0.70, 1.00)]),  # Mid and large database.
                partial(rhs_hats, [(0.30, 0.70, 0.50)]),  # Large database.
                partial(rhs_hats, [(0.45, 0.55, 2.00)]),  # Large database.
                ]  # Potential rhs equations

    # Perform the testing and add the result to the database.
    x = np.linspace(0, 1, 10001)
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
