# ToDo: Add support for constant additions to the primal. This also relaxes the Dirichlet BC.
r"""
The configuration is a proposed solution for the problem.

A :py:class:`~configuration.Configuration` includes the selection of :py:class:`~patch.Patch` from
:py:class:`~patch.PatchDatabase` for each of the domains described by :py:class:`~problem.Problem`. It contains the
coordinate transformation as free variables, the optimal value of which will be found through an optimization process.

The :py:class:`~configuration.ConfigurationDatabase` contains a collection with all configurations that are considered.
It will also allow for the access to the optimization and other methods of the configurations in a parallel manner, as
the those are independent.

Bram Lagerweij |br|
Mechanics of Composites for Energy and Mobility |br|
KAUST |br|
2021
"""

# Import external modules
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from itertools import product
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from functools import partial

# Import my own scripts.
from helperfunctions import _m


class Configuration(object):
    r"""
    A configuration is a proposed solution to the problem.

    A configuration consists of a selection of patches and the assignment of coordinate translations. These 
    coordinate transformations can be fixed by an optimization method, and the quality of the solution will be 
    represented by an error function.

    Parameters
    ----------
    problem : Problem
        The problem that prescribes the boundary conditions and the subdomains.
    patches : tuple
        All the patch objects that are to be used in each of the subdomains of the problem.
    translation_bounds : list
        The freedom in translation for the patch in each domain, for each domain this is a tuple with `(t_min, t_max)`.

    Attributes
    ----------
    problem : Problem
        The problem that prescribes the boundary conditions and the subdomains.
    patches : tuple
        All the patch objects that are to be used in each of the subdomains of the problem.
    translation : array
        The coordinate translation for the patch in each domain, needs to stay within given bounds.
    rbd : array
        The primal degrees of freedom :math:`\bar{u}_d` are additions to the primal field that will not influence the
        right hand side. For non-linear Poisson equations this includes a constant addition per domain
         :math:`u_d = u_{p_p} + \bar{u}_d`,
    """

    def __init__(self, problem, patches, translation_bounds):
        r"""
        A configuration is created from an admissible patch & domain combination.
        """
        self.problem = problem
        self.patches = patches

        # Initialize hidden attributes.
        self._rbd = np.zeros(self.problem.num_domains)
        self._translation = np.zeros(problem.num_domains)

        # Set the rigid body motions.
        self.rbd = np.zeros_like(self._rbd)

        # Set the translation and its bounds.
        self._translation_bounds = np.array(translation_bounds)
        self.translation = np.mean(translation_bounds, axis=1)

        # Determine what parameters are free.
        self._free_rbd = np.full_like(self._rbd, True, dtype=bool)
        self._free_translation = np.full_like(self.translation, True, dtype=bool)
        for d, domain in enumerate(self.problem.subdomains):
            # Find which primal freedom is free.
            if len(domain.u_bc) != 0:
                self._free_rbd[d] = False

            # Find free translations.
            if self._translation_bounds[d][0] == self._translation_bounds[d][1]:
                self._free_translation[d] = False

        # Make some attributes semi-immutable to not alter them by accident.
        # From here on a slice can be obtained, but a slice cannot be changed.
        self._rbd.setflags(write=False)
        self.rbd.setflags(write=False)
        self._free_rbd.setflags(write=False)
        self._translation_bounds.setflags(write=False)
        self._translation.setflags(write=False)
        self.translation.setflags(write=False)
        self._free_translation.setflags(write=False)

    @property
    def rbd(self):
        rbd = np.copy(self._rbd)
        rbd.setflags(write=False)
        return rbd

    @rbd.setter
    def rbd(self, rbd):
        """
        Set the rigid body displacement such that is satisfies the kinematic boundary conditions.
        Not all choices of rigid body displacement variables are admissible, and which ones are depends on the boundary
        conditions defined by the problem, the patches that are selected and the translation considered. As a result the
        change in a rdb value has to be validated and corrected such that it will match the boundary conditions.
        Parameters
        ----------
        rbd : array
            The proposed rigid body displacement values.
        """
        # Create a writable copy of the rbd magnitudes.
        rbd_set = self.rbd.copy()
        rbd_set.setflags(write=True)

        for d, domain in enumerate(self.problem.subdomains):
            if len(domain.u_bc) == 0:  # No kinematic constraints.
                rbd_set[d] = rbd[d]  # No checks just set rbd.

            elif len(domain.u_bc) == 1:  # With a kinematic constraint.
                # Get properties.
                patch = self.patches[d]
                t = self._translation[d]
                constraint = domain.u_bc[0]
                u_no_rbd = InterpolatedUnivariateSpline(patch.x + t, patch.u, k=1)  # Interpolated displacement field.

                # Ensure that the boundary condition is satisfied.
                u_at_constraint = u_no_rbd(constraint.x)
                rbd_set[d] = -u_at_constraint

            else:  # There cannot be more than one kinematic constraint.
                ValueError("There cannot be more than one dirichlet boundary conditions in each domain.")

        # Set the rigid body displacements.
        self._rbd = rbd_set
        self._rbd.setflags(write=False)

    @property
    def translation(self):
        translation = np.copy(self._translation)
        translation.setflags(write=False)
        return translation

    @translation.setter
    def translation(self, translation):
        """
        Set the translations, ensure that the bounds are satisfied.

        Parameters
        ----------
        translation : array
            The translation that is to be set.
        """
        # Clip the translation values that are beyond the limits.
        self._translation = np.clip(translation, self._translation_bounds[:, 0], self._translation_bounds[:, 1])
        self._translation.setflags(write=False)

        # Changing the coordinates transformation might affect the rigid body motions.
        self.rbd = self.rbd

    def domain_primal(self, x):
        r"""
        Calculate the primal field :math:`u_d` of the domains.

        The primal field of each domain depends on the patch selected for and the coordinate translation.
        Because the primal field is only known at discrete location these are interpolated to the locations of
        interest :math:`x`. In the end the equation for the domain primal is:

        .. math:: u_d(x) = \mathcal{I}(x, \xi_{d,p}+t_d, u_{d,p})

        where :math:`\xi_{d,p}` and :math:`u_{d,p}` are the coordinates and primal of patch selected for a
        domain. Then :math:`\mathcal{I}(x_i, x_p, f_p)` is a interpolation function to find the magnitude of field
        :math:`f_p` known at coordinates :math:`x_p` at coordinates :math:`x_i`.

        .. note:: The primal field of a domain outside a domain is set to `np.NaN`, as it is technically not
            defined at those locations, that is :math:`u_d(x)=` `np.NaN` :math:`\quad \forall x\notin\Omega_d`.

        Parameters
        ----------
        x : array
            The coordinates where the domain primal has to be calculated.

        Returns
        -------
        array
            The primal field of each domain :math:`u_d(x)`.
        """
        u_domains = np.full((self.problem.num_domains, len(x)), np.NaN, dtype=float)

        # Loop over all domains and get the primal field at x in problem coordinates.
        for d, domain in enumerate(self.problem.subdomains):
            index = np.where((domain.domain[0] <= x) & (x <= domain.domain[1]))[0]
            u_d = InterpolatedUnivariateSpline(self.patches[d].x + self.translation[d], self.patches[d].u, k=3)
            u_domains[d, index] = u_d(x[index]) + self.rbd[d]  # Add primal freedom.
        return u_domains

    def domain_rhs(self, x):
        r"""
        Calculate the right hand side field :math:`g_d` that was prescribed to the domains.

        The primal field of each domain depends on the patch selected for and the coordinate translation.
        Because the primal field is only known at discrete location these are interpolated to the locations of
        interest :math:`x`. In the end the equation for the domain primal is:

        .. math:: g_d(x) = \mathcal{I}(x, \xi_{d,p}+t_d, g_{d,p})

        where :math:`\xi_{d,p}` and :math:`g_{d,p}` are the coordinates and righ hand side of patch selected for a
        domain. Then :math:`\mathcal{I}(x_i, x_p, f_p)` is a interpolation function to find the magnitude of field
        :math:`f_p` known at coordinates :math:`x_p` at coordinates :math:`x_i`.

        .. note:: The right hand side field of a domain outside its own subdomain is set to `np.NaN`,
            as it is technically not defined at those locations, that is :math:`u_d(x)=` `np.NaN` :math:`\quad \forall
            x\notin\Omega_d`.

        Parameters
        ----------
        x : array
            The coordinates where the domain primal has to be calculated.

        Returns
        -------
        array
            The primal field of each domain :math:`u_d(x)`.
        """
        rhs_domains = np.full((self.problem.num_domains, len(x)), np.NaN, dtype=float)

        # Loop over all domains and get the primal field at x in problem coordinates.
        for d, domain in enumerate(self.problem.subdomains):
            index = np.where((domain.domain[0] <= x) & (x <= domain.domain[1]))[0]
            u_d = InterpolatedUnivariateSpline(self.patches[d].x + self.translation[d], self.patches[d].rhs, k=3)
            rhs_domains[d, index] = u_d(x[index])
        return rhs_domains

    def error(self, x):
        r"""
        Calculate the least square domain decomposition error.

        The error represents how wel the primal fields in the overlapping areas match. The lower the error the
        more the primal fields in the overlapping areas agree with each other.

        .. math:: \mathcal{E} = \sum_{a=1}^D \sum_{b>a}^D \int_{\Omega_a\cap\Omega_b} \| u_a - u_b \|^2 dx

        Parameters
        ----------
        x : array
            The locations where the error relation is evaluated.

        Returns
        -------
        float
            The data-driven error.
        """
        # Get fields for current state of configuration.
        ud = self.domain_primal(x)

        # Calculate the error of all domains and add those.
        error = 0

        # Compute missmatch norms on the overlap, no weighting functions used.
        for a in range(len(self.problem.subdomains)):
            for b in range(len(self.problem.subdomains)):
                # Overlap between domain a and b, to ensure not to count everything twice, we do b > a.
                if b > a:
                    # Find the locations of the overlap.
                    overlap_end = self.problem.subdomains[a].domain[1]
                    overlap_start = self.problem.subdomains[b].domain[0]

                    # Check whether these are actually overlapping.
                    if overlap_start < overlap_end:
                        # Find the sample points on the overlapping region.
                        index = np.where((x >= overlap_start) & (x <= overlap_end))

                        # Verify that the subdomains are actually overlapping, and that there are enough sample points.
                        if index[0].shape[0] > 3:
                            u_gap = InterpolatedUnivariateSpline(x[index], (ud[a, index] - ud[b, index])**2, k=3)
                            error += u_gap.integral(overlap_start, overlap_end)

                        else:
                            raise ValueError("Insufficient sample points in overlap.")

        return error

    def _objective_function(self, params, x):
        """
        The objective function that will be minimized.

        The optimization will only act on the free parameters, this function will pack and unpack these correctly into
        the attributes.

        Parameters
        ----------
        params : array
            The input parameters consisting of the translations.
        x : array
            Locations where the objective function has to be evaluated.

        Returns
        -------
        float
            Magnitude of the minimization function.
        """
        # Unpack the rbd freedom.
        rbd = np.copy(self.rbd)
        rbd[self._free_rbd] = params[:np.count_nonzero(self._free_rbd)]
        self.rbd = rbd

        # Unpack the translation parameters.
        translation = np.copy(self.translation)
        translation[self._free_translation] = params[np.count_nonzero(self._free_rbd):]
        self.translation = translation

        # Calculate error norm.
        error = self.error(x)
        return error

    def optimize(self, x, verbose=False):
        r"""
        Find the optimal coordinate translations for each domain.

        A bounded optimization is used, which tries to find the best coordinate transformations such that the error
        is minimized. However, some translations are fixed (these will not be exposed to the optimization) and all
        free translations have bounds, these bounds will be satisfied.

        Parameters
        ----------
        x : array
            Locations where the objective function needs to be analyzed.
        verbose : bool, optional
            Printing the progress of the optimization at every iteration, `False` by default.

        Returns
        -------
        scipy.optimize.OptimizeResult
            The optimization result represented as a `OptimizeResult` object.
        """
        # Get and initialize the free parameters, that is the free translations.
        params_initial = np.hstack((self.rbd[self._free_rbd], self.translation[self._free_translation]))
        lb = np.hstack((np.array([-np.inf] * np.count_nonzero(self._free_rbd)),
                        np.array(self._translation_bounds)[self._free_translation, 0]))
        ub = np.hstack((np.array([np.inf] * np.count_nonzero(self._free_rbd)),
                        np.array(self._translation_bounds)[self._free_translation, 1]))
        bounds = Bounds(lb, ub)

        # Sequential Least Squares Programming (The best optimization approach for this problem)
        options = {'ftol': 1e-25, 'maxiter': 20000, 'disp': verbose, 'iprint': 2}
        result = minimize(self._objective_function, params_initial, args=x, bounds=bounds, method='SLSQP',
                          tol=0, jac='3-point', options=options)

        # Ten ensure that we set the final state of the configuration to the optimal one.
        self._objective_function(result.x, x)
        return result

    def plot(self, x, material=None, title=None, path=None):
        r"""
        Plotting the state of this configuration.

        Parameters
        ----------
        x : array
            The locations where the state is sampled.
        material : Constitutive, optional
            The constitutive equation for the material considered, if provided it is used to calculate the exact
            solution using Euler-Bernoulli beam theory.
        title : str, optional
            The title of the plot, if any is specified.
        path : str, optional
            The path to which a .png and a .svg need to be saved, disabled by default.

        Returns
        -------
        matplotlib.Axis
            The axis of the plot.
        """
        # Get fields for current state of configuration.
        ud = self.domain_primal(x)
        gd = self.domain_rhs(x)

        # Create figures and axes.
        fig, axis = plt.subplots(2, 1, sharex='col', figsize=(10, 6),
                                 gridspec_kw={'height_ratios': [3, 2], 'hspace': 0})
        plt.suptitle(title)
        ax_u = axis[0]  # primal field
        ax_g = axis[1]  # Internal load axis for moment

        # Calculate the value of the cost function, and format it in a string for display purposes.
        result = _m(rf"$\mathcal{{E}}={self.error(x):4.2e}$")

        # Get the reference solution and plot it.
        if material is not None:
            x_exact, u_exact, rhs_exact = self.problem.exact(x, material)
            ax_u.plot(x_exact, u_exact, color='grey', label=_m(r"$u^{exact}$"))
            ax_g.plot(x_exact, rhs_exact, color='grey', label=_m(r"$g(x)^{exact}$"))

        # Plot domain primal, moment and bending energy density fields.
        for d, domain in enumerate(self.problem.subdomains):
            # Get domain information.
            index = np.where((domain.domain[0] <= x) & (x <= domain.domain[1]))[0]
            ax_u.plot(x[index], ud[d, index], label=_m(rf"$u_{d}$"))
            ax_g.plot(x[index], gd[d, index], label=_m(rf"$g_{d}$"))

        # Fix axis and add legend.
        ax_u.annotate(result, xy=(0.02, 0.02), xycoords='axes fraction', ha='left', va='bottom')
        ax_u.legend(loc=1, frameon=False)
        ax_g.legend(loc=1, frameon=False)

        # Add axis labels.
        ax_u.set_ylabel(_m(r"Primal field $u$"))
        ax_g.set_ylabel(_m(r"Right hand side $g$"))
        ax_g.set_xlabel(_m(r"Location $x$"))

        # Save the plot as image.
        if path is not None:
            fig.savefig(path + title + '.png')
            fig.savefig(path + title + '.svg')
        return ax_u, ax_g

    def compare_to_exact(self, x, material):
        r"""
        Compare this configuration to the exact solution.

        This is measured as the Euclidean norm :math:`L^2` between the domain primal and the reference primal.

        .. math:: e = \sum_{d=1}^D \sqrt{\int_{\Omega_d} \| u_d - u^{\text{exact}} \|^2 dx}

        Parameters
        ----------
        x : array
            The locations where the bending energy is evaluated.
        material : Constitutive
            The constitutive response of this material.

        Returns
        -------
        error : float
            The :math:`L^2` between the domain primals and the reference solution primal.
        """
        # Get fields for the exact solution and the current state of configuration.
        x_exact, u_exact, rhs_exact = self.problem.exact(x, material)
        ud = self.domain_primal(x_exact)

        # Compare the primal fields:
        error = 0
        for d in range(len(self.problem.subdomains)):
            # Find the domain filled with our subdomain d.
            start = self.problem.subdomains[d].domain[0]
            end = self.problem.subdomains[d].domain[1]
            index = np.where((x_exact >= start) & (x_exact <= end))

            # Compute the primal mismatch between the domain and reference solution in this domain.
            u_gap = (ud[d, index] - u_exact[index]) ** 2
            u_gap_spline = InterpolatedUnivariateSpline(x_exact[index], u_gap, k=3)

            # Compute the error.
            error += np.sqrt(u_gap_spline.integral(start, end))

        return error


class ConfigurationDatabase(object):
    r"""
    This contains the collection of all configurations that are considered.

    To manage all different configurations this class can be used. It allows for the creation of all configurations and
    contains function calls that apply on all the configuration objects simultaneously.

    .. note:: Creates an empty database by default, please use the class-methods :py:meth:`create_from_problem_patches`
        or :py:meth:`create_from_load` to create a filled ConfigurationDatabase.

    Parameters
    ----------
    database : DataFrame, optional
        The DataFrame containing all configurations, `None` by default.

    Attributes
    ----------
    database : DataFrame
        A pandas DataFrame containing all the different configurations.
    """

    def __init__(self, database=None):
        r"""
        The database creates configurations from all admissible patch & domain combinations.
        """
        if database is None:
            self.database = pd.DataFrame()
        else:
            self.database = database

    @classmethod
    def create_from_problem_patches(cls, problem, patch_database):
        r"""
        Creating a ConfigurationDatabase from a problem formulation and patch database.

        For each subdomain in the problem the admissible patches in the database will be found and configurations will
        be created from all possible combinations of admissible patches.

        Parameters
        ----------
        problem : Problem
            The problem formulation that describes the problem in question.
        patch_database : PatchDatabase
            The patch database.

        Returns
        -------
        ConfigurationDatabase
            A database with all admissible configurations.
        """

        # Get the admissible patch - domain combinations and their translation bounds.
        admissibility, translations = problem.domain_patch_admissibility(patch_database)
        combinations = list(product(*admissibility))

        database = []
        for combination in combinations:
            patches = tuple([patch_database.database[patch] for patch in combination])

            # Loop over all admissible translation ranges of this patch domain configuration.
            translation_bounds_list = []
            for domain, patch in enumerate(combination):
                translation_bounds_list.append(translations[domain][patch])

            # Create list with potential translation bound combinations.
            translation_bounds_combinations = list(product(*translation_bounds_list))
            for translation_bounds in translation_bounds_combinations:
                configuration = Configuration(problem, patches, translation_bounds)
                database.append([configuration, combination, translation_bounds])

        # Create dataframe.
        index = ['configuration', 'patch', 'translation bounds']
        database = pd.DataFrame(database, columns=index)
        return cls(database=database)

    @classmethod
    def create_from_load(cls, filename, path=''):
        """
        Creating a ConfigurationDatabase from a previously saved simulation.

        .. warning::
            Loading pickled data received from untrusted sources can be unsafe. See:
            https://docs.python.org/3/library/pickle.html

        Parameters
        ----------
        filename : str
            The name of the file to be loaded, should include the file extension.
        path : str, optional
            Path to the file that is to be loaded, working directory be default.

        Returns
        -------
        ConfigurationDatabase
            Returns a database with all the configurations that were previously stored.
        """
        database = pd.read_pickle(path + filename)
        return cls(database=database)

    def num_configurations(self):
        r"""
        The number of configurations in the database.

        Returns
        -------
        int
            The number of configurations.
        """
        return len(self.database)

    def optimize(self, x, parallel=False):
        r"""
        Apply the optimization function to all configurations considered.

        The optimization method is applied to all configurations and the results, the error and the configuration state,
        are added as an column in the database.

        .. warning:: Changes the object in place, the database will be extended with a column on the optimization
            results.

        Parameters
        ----------
        x : array
            Locations where the objective function needs to be analyzed.
        parallel : bool, optional
            `True` if the computation should be performed in parallel, `False` if it should be in series.

        See Also
        --------
        Configuration.optimize : The function that is called for all configurations.
        """

        # Create partial optimization function that contains all fixed information.
        def function(config):
            """The wrapper around the :py:meth:`Configuration.optimize` function."""
            result = config.optimize(x)
            out = pd.Series(
                [config, result.fun, config.translation, result.success])
            return out

        # Apply the optimization function to all configurations.
        if self.num_configurations() == 0:
            raise ValueError("There are no configurations in this database.")
        else:
            print(f"Minimizing the error equation for {self.num_configurations()} configurations.")
            if parallel:
                # Check whether parallel run was initialized before.
                if hasattr(self.database.configuration, 'parallel_apply') is False:
                    pandarallel.initialize()

                # The parallel run.
                self.database[['configuration', 'error', 'translation',
                               'success']] = self.database.configuration.parallel_apply(function)
            else:
                print("    Slow computation because execution is in series.")
                self.database[['configuration', 'error', 'translation',
                               'success']] = self.database.configuration.apply(function)

    def error(self, x, parallel=False):
        r"""
        Apply the error calculation function to all configurations considered and sort based upon it.

        .. warning:: Changes the object in place, a column with the error magnitude will be added to the database.

        Parameters
        ----------
        x : array
            Locations where the objective function needs to be analyzed.
        parallel : bool, optional
            `True` if the computation should be performed in parallel, `False` if it should be in series.

        See Also
        --------
        Configuration.error : The function that is called for all configurations.
        """

        # Create partial optimization function that contains all fixed information.
        def function(config):
            """The wrapper around the :py:meth:`Configuration.error` function."""
            error = config.error(x)
            return error

        # Apply the optimization function to all configurations.
        if self.num_configurations() == 0:
            raise ValueError("There are no configurations in this database.")
        else:
            print(f"Computing the DD Error for {self.num_configurations()} configurations.")
            if parallel:
                # Check whether parallel run was initialized before.
                if hasattr(self.database.configuration, 'parallel_apply') is False:
                    pandarallel.initialize()

                # The parallel run.
                self.database['error'] = self.database.configuration.parallel_apply(function)
            else:
                print("    Slow computation because execution is in series.")
                self.database['error'] = self.database.configuration.apply(function)

    def compare_to_exact(self, x, material, parallel=False):
        r"""
        Apply te compare to exact calculation to all configurations in the database.

        Parameters
        ----------
        x : array
            Locations where the objective function needs to be analyzed.
        material : Constitutive
            The constitutive equation of the material in question.
        parallel : bool, optional
            `True` if the computation should be performed in parallel, `False` if it should be in series.

        See Also
        --------
        Configuration.compare_to_exact : The function that is called for all configurations.
        """

        # Create partial optimization function that contains all fixed information.
        def function(config):
            """The wrapper around the :py:meth:`Configuration.potential_energy` function."""
            error_to_exact = config.compare_to_exact(x, material)
            return error_to_exact

        # Apply the optimization function to all configurations.
        if self.num_configurations() == 0:
            raise ValueError("There are no configurations in this database.")
        else:
            print(f"Computing error with respect to reference solution for {self.num_configurations()} configurations.")
            if parallel:
                # Check whether parallel run was initialized before.
                if hasattr(self.database.configuration, 'parallel_apply') is False:
                    pandarallel.initialize()

                # The parallel run.
                self.database['error_to_exact'] = self.database.configuration.parallel_apply(function)
            else:
                print("    Slow computation because execution is in series.")
                self.database['error_to_exact'] = self.database.configuration.apply(function)

    def sort(self, key):
        """
        Sort the configuration database according to the given column.

        Parameters
        ----------
        key : str
            The column for which the database needs to be sorted.
        """
        self.database = self.database.sort_values(key, ascending=True)

    def plot(self, x, material, max_images=5, title=None, path=None):
        r"""
        Plot the first 'max_images` configurations, based upon the current order in `self.database`.

        Parameters
        ----------
        x : array
            The locations where the state is sampled.
        material : Constitutive
            The constitutive equation for the material considered, if provided it is used to calculate the exat solution
            using Euler-Bernoulli beam theory.
        max_images : int, optional
            The maximum number of configurations that will be plotted, defaults to 5.
        title : str, optional
            The title on top of the image, if any is specified.
        path : str, optional
            The path to which a .png and a .svg need to be saved, disabled by default.

        See Also
        --------
        Configuration.plot : The function that is called for all configurations.
        """
        for i, configuration in enumerate(self.database.configuration.head(max_images)):
            title_i = f'Configuration {self.database.index[i]}'
            if title is not None:
                title_i = title + f' {i} ({title_i})'
            configuration.plot(x, material, title=title_i, path=path)
        plt.show()

    def save(self, filename, path='', compression="infer"):
        r"""
        Saving the DataFrame into a pickle.

        This will store the entirety of the configurations database as a pickle. The benefit is that this allows for the
        computation of optimal solutions on remote machines, then transfer the data to your machine to analyze the
        results. The optimization results (the errors, and optimal coordinate translation) are then
        included and post-processing can be performed without the need for expensive computations.

        Parameters
        ----------
        filename : str
            The name of the pickled file, do include the preferred file extension.
        path : str, optional
            The path to where the file should be stored, saves to working directory by default.
        compression : str, optional
            Whether and with what format the object should be compressed, `infer` from filename extension by default.
        """
        self.database.to_pickle(path + filename, compression=compression)
