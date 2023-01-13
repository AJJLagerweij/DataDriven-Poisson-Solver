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

# Importing my own scripts.
from configuration import ConfigurationDatabase
import problem
import test
from patch import PatchDatabase
from constitutive import Softening, LinearMaterial

# Setup basic plotting properties.
plt.close('all')
plt.rcParams['svg.fonttype'] = 'none'

if __name__ == "__main__":
    # Run settings.
    parallel = False

    # Problem definition.
    problem_length = 1000  # mm
    problem_load = -1  # kN
    domain_num = 3  # Number of domains
    domain_length = 1000./domain_num  # Length of the subdomains in mm.
    problem = ExampleLoadControlled(problem_length, problem_load, domain_length, domain_num)
    problem.continuity = -1

    # Locations for the error and error computations and plots.
    x = np.linspace(0, 1000, 10001)

    # Material definition.
    material_var_lin = 7000.0  # Linear material equation constant.
    material_var_nonlin = 10e-6  # Non-linear material equation constant.
    material = LinearMaterial(3e4 * material_var_lin)
    # material = Softening(material_var_lin, material_var_nonlin)

    # Create empty database.
    database = PatchDatabase()

    # The test matrix, consider the Simply Supported beam of the following specifications.
    specimen_length = [1000]  # mm specimen length.
    specimen_load = [-1]  # kN

    # Perform the testing and add the result to the database.
    specimen_dx = 1  # mm discretization step size (DIC resolution)
    for load in specimen_load:
        for length in specimen_length:
            test = Example(length, specimen_dx, load, material)
            database.add_test(test)

    # Plot the resulting database, if required one can rotate or mirror here.
    print("\nNumber of patches", database.num_patches())

    # Create a set of all configurations based upon these admissible configurations.
    configurations = ConfigurationDatabase.create_from_problem_patches(problem, database)

    # Read database.
    # configurations = ConfigurationDatabase.create_from_load(f'Exact Non-overlapping and TPE {domain_num} domains {database.num_patches()} patches.pkl.gz')

    # Perform the optimization, and compute errors.
    configuration = configurations.database.iloc[0, 0]
    configuration.optimize(x, verbose=True)
    # configuration.equilibrium(remove=False, parallel=parallel)
    configuration.compare_to_exact(x, material)
    # configurations.error_alternative(x, parallel=parallel)
    configuration.potential_energy(x)
    # configurations.save(f'Exact Non-overlapping and TPE {domain_num} domains {database.num_patches()} patches.pkl.gz')

    # Plot best error solution.
    # configurations.sort('error')

    configuration.plot(x, material, title=r"$\min \lambda \Pi^p$ s.t. $\sum (u_a - u_b)^2 + (du_a - du_b)^2=0$")

    # # Plot best energy solution.
    # configurations.sort('tpe')
    # configuration = configurations.database.iloc[0, 0]
    # configuration.plot(x, material, title=r"min $\Pi^p$")

    # # Plot best energy solution.
    # configurations.sort('error_to_exact')
    # configuration = configurations.database.iloc[0, 0]
    # configuration.plot(x, material, title=r"min Error to Exact")

    # # Scatter plots vs tpe.
    # configurations.database.plot.scatter(x='error_to_exact', y='error', title=r"Old error $\sum \int (u_a-u_b)^2$ vs TPE")
    # configurations.database.plot.scatter(x='error_to_exact', y='L2M', title=r"$L^2(\Delta u)$ vs TPE")
    # configurations.database.plot.scatter(x='tpe', y='H1', title=r"$H^1(\Delta u)$ vs TPE")
    # configurations.database.plot.scatter(x='tpe', y='H2', title=r"$H^2(\Delta u)$ vs TPE")

    plt.show()

    # Scatter plot error vs energy.
    # configurations.database['e2'] = np.sqrt(configurations.database['error'])/250
    # data = configurations.database[configurations.database['equilibrium'] is True]
    # configurations.database.plot.scatter('error', 'tpe')
