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
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import minimize
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
    patch_database : PatchDatabase
            The patch database.

    Attributes
    ----------
    problem : Problem
        The problem that prescribes the boundary conditions and the subdomains.
    patches : tuple
        All the patch objects that are to be used in each of the subdomains of the problem.
    rbd : array
        The degree of freedoms of the homogeneous solutions. The linear homogeneous solutions contain two degrees of
        freedom per domain.
    """

    def __init__(self, problem, patch_database):
        r"""
        A configuration is created from an admissible patch & domain combination.
        """
        # For each domain we want to find and admissible patch.
        admissibility, translations = problem.domain_patch_admissibility(patch_database)

        # Assign these to the domains.
        try:
            pd = [patches_in_d[0] for patches_in_d in admissibility]
        except IndexError:
            for d in range(len(admissibility)):
                print(f'Domain {d} has {len(admissibility[d])} patches that satisfy RHS requirements.')

            raise IndexError("No patch that satisfies the RHS requirements for a domain.")

        # Initialize the problem.
        self.problem = problem
        self._translation = np.array([translations_d[pd[d]][0][1] for d, translations_d in enumerate(translations)])
        self.patches = tuple(patch_database.database[patch] for patch in pd)

        # Initialize the rigid body motions.
        self._rbd = np.zeros((problem.num_domains, 2))  # Private rigid body motion object.
        self.rbd = np.zeros((problem.num_domains, 2))

        # Determine what parameters are free.
        self._free_rbd = np.full_like(self.rbd, True, dtype=bool)
        for d, domain in enumerate(self.problem.subdomains):
            # Find free rigid body motions.
            if len(domain.u_bc) == 1:
                # If constraint is at domain end, the fixed rbd is related to domain end. Otherwise it is domain start.
                if domain.u_bc[0].x == domain.domain[1]:
                    self._free_rbd[d, 1] = False
                else:
                    self._free_rbd[d, 0] = False
            if len(domain.u_bc) == 2:
                self._free_rbd[d, 0] = False
                self._free_rbd[d, 1] = False

        # Make some attributes semi-immutable to not alter them by accident.
        # From here on a slice can be obtained, but a slice cannot be changed.
        self._translation.setflags(write=False)
        self._rbd.setflags(write=False)
        self.rbd.setflags(write=False)
        self._free_rbd.setflags(write=False)

        # Empty optimization variables.
        self._intermediate_results = []

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
            if len(domain.u_bc) == 0:  # No kinematic constraints → No checks just set rbd.
                rbd_set[d] = rbd[d]

            else:  # A condition exists.
                # Get the current domain settings
                patch = self.patches[d]
                t = self._translation[d]
                u_no_rbd = InterpolatedUnivariateSpline(patch.x + t, patch.u, k=1)  # Interpolated displacement field.
                d_start = domain.domain[0]
                d_end = domain.domain[1]
                length = d_end - d_start

                if len(domain.u_bc) == 1:  # Only one constraint exists, this fixes a single degree of freedom.
                    # Get constraint properties.
                    constraint = domain.u_bc[0]

                    # Get patch properties at the constraints.
                    u_at_constraint = u_no_rbd(constraint.x)

                    # Assumed that u2 is free, and u1 gets fixed by choosing u2.
                    if constraint.x != d_end:
                        u2 = rbd[d, 1]
                        u1 = (constraint.magnitude - u_at_constraint - u2 * (constraint.x - d_start) / length) * \
                             length / (constraint.x - d_end)

                    else:  # u2 gets fixed by the constraint and u1 remains free.
                        u1 = rbd[d, 0]
                        u2 = constraint.magnitude - u_at_constraint

                    # Set the rbd values.
                    rbd_set[d] = np.array([u1, u2])

                if len(domain.u_bc) == 2:  # Two constraints exist, both degrees of freedom are fixed.
                    # Get constraint properties.
                    constraint1 = domain.u_bc[0]
                    constraint2 = domain.u_bc[1]

                    # Get patch properties at the constraints.
                    u_at_constraint1 = u_no_rbd(constraint1.x)
                    u_at_constraint2 = u_no_rbd(constraint2.x)

                    if constraint1.x == d_start and constraint2.x == d_end:
                        u1 = constraint1.magnitude - u_at_constraint1
                        u2 = constraint2.magnitude - u_at_constraint2
                    else:  # At least a single constraint does not intersect.
                        if constraint1.x == d_start:
                            u1 = constraint1.magnitude - u_at_constraint1
                            u2 = (constraint2.magnitude - u_at_constraint2 - u1 * (constraint2.x - d_end) / length)*\
                                 length / (constraint2.x - d_start)
                        elif constraint2.x == d_end:  # End constraint.
                            u2 = constraint2.magnitude - u_at_constraint2
                            u1 = (constraint1.magnitude - u_at_constraint1 - u2 * (constraint1.x - d_start) / length)*\
                                 length / (constraint1.x - d_end)
                        else:  # No constraints intersect.
                            raise NotImplementedError("Having two Dirichet constraints while not at boundary is not "
                                                      "implemented.")

                    # Set the rbd values.
                    rbd_set[d] = np.array([u1, u2])

                else:
                    ValueError("There cannot be more than two dirichlet boundary conditions in one domain.")

        # Set the resulting rigid body motions.
        self._rbd = rbd_set
        self._rbd.setflags(write=False)

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
            d_start = domain.domain[0]
            d_end = domain.domain[1]
            index = np.where((domain.domain[0] <= x) & (x <= domain.domain[1]))[0]
            u_d = InterpolatedUnivariateSpline(self.patches[d].x + self._translation[d], self.patches[d].u, k=3)
            u_domains[d, index] = u_d(x[index]) + \
                                  self.rbd[d, 0] * (x[index] - d_end) / (d_end - d_start) + \
                                  self.rbd[d, 1] * (x[index] - d_start) / (d_end - d_start)
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
            u_d = InterpolatedUnivariateSpline(self.patches[d].x + self._translation[d], self.patches[d].rhs, k=3)
            rhs_domains[d, index] = u_d(x[index])
        return rhs_domains

    def error(self, x, order='Omega1'):
        r"""
        Calculate the least square domain decomposition error.

        The error represents how wel the primal fields in the overlapping areas match. The lower the error the
        more the primal fields in the overlapping areas agree with each other.

        .. math:: \mathcal{E} = \sum_{a=1}^D \sum_{b>a}^D \int_{\Omega_a\cap\Omega_b} \| u_a - u_b \|^2 dx

        Parameters
        ----------
        x : array
            The locations where the error relation is evaluated.
        order : string, optional
            Defines the type of norm in the cost function. Defaults to Omega1.

        Returns
        -------
        float
            The data-driven error.
        """
        # Test type flag.
        if order is None:
            order = 'Omega1'
        if order not in ('Omega0', 'Omega1', 'Gamma0', 'Gamma1', 'Omega1_weights'):
            raise ValueError("Order flag type must be 'Omega0', 'Omega1', 'Gamma0', 'Gamma1' or 'Omega1_weights'")

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
                            u_gap = InterpolatedUnivariateSpline(x[index], (ud[a, index] - ud[b, index]), k=3)
                            du_gap = u_gap.derivative()
                            if order == 'Omega0':
                                local_error = u_gap(x[index])**2
                                local_error = InterpolatedUnivariateSpline(x[index], local_error, k=3)
                                error += 0.5 * local_error.integral(overlap_start, overlap_end)
                            if order == 'Omega1':
                                local_error = u_gap(x[index]) ** 2 + du_gap(x[index])**2
                                local_error = InterpolatedUnivariateSpline(x[index], local_error, k=3)
                                error += 0.5 * local_error.integral(overlap_start, overlap_end)
                            if order == 'Gamma0':
                                local_error = u_gap(x[index]) ** 2
                                local_error = InterpolatedUnivariateSpline(x[index], local_error, k=3)
                                error += 0.5 * (local_error(overlap_start) + local_error(overlap_end))
                            if order == 'Gamma1':
                                local_error = u_gap(x[index]) ** 2 + du_gap(x[index])**2
                                local_error = InterpolatedUnivariateSpline(x[index], local_error, k=3)
                                error += 0.5 * (local_error(overlap_start) + local_error(overlap_end))
                            if order == 'Omega1_weights':
                                o = (overlap_end-overlap_start)
                                L = self.problem._length
                                A = 1 / (o * L**2)
                                B = (3*L**2 - o**2) / (12*o*L**2)
                                local_error = A*u_gap(x[index]) ** 2 + B*du_gap(x[index]) ** 2
                                local_error = InterpolatedUnivariateSpline(x[index], local_error, k=3)
                                error += 0.5 * local_error.integral(overlap_start, overlap_end)

                        else:
                            raise ValueError("Insufficient sample points in overlap.")

        return error

    def _objective_function(self, params, x, order=None):
        """
        The objective function that will be minimized.

        The optimization will only act on the free parameters, this function will pack and unpack these correctly into
        the attributes.

        Parameters
        ----------
        params : array
            The input parameters consisting of the free rigid body motions.
        x : array
            Locations where the objective function has to be evaluated.
        order : string, optional
            Defines the type of norm in the cost function. Defaults to `None`.

        Returns
        -------
        float
            Magnitude of the minimization function.
        """
        # Unpack the rbd parameters.
        rbd = np.copy(self.rbd)
        rbd[self._free_rbd] = params
        self.rbd = rbd

        # Calculate error norm.
        error = self.error(x, order=order)
        return error

    def _store_intermediate(self, x, material, params, order=None):
        """
        Store intermediate results of the optimization.

        Parameters
        ----------
        x : array
            Locations where the objective function needs to be analyzed.
        material : Constitutive, optional
            The constitutive equation for the material considered, if provided it is only used to calculate the distance
            between the current solution and the exact solution.
        params : array
            The input parameters consisting of the free rigid body motions.
        order : string, optional
            Defines the type of norm in the cost function. Defaults to `None`.

        Returns
        -------

        """
        # Unpack the rbd parameters.
        rbd = np.copy(self.rbd)
        rbd[self._free_rbd] = params
        self.rbd = rbd

        # Calculate error norm, and the distance to the exact solution for each subdomain.
        error = self.error(x, order=order)
        ed = self.compare_to_exact(x, material)
        self._intermediate_results.append(np.hstack(([error, ed], rbd.flatten())))

    def optimize(self, x, verbose=False, material=None, order=None):
        r"""
        Find the optimal rigid body displacements for each domain.

        A bounded optimization is used, which tries to find the best rigid body displacements such that the error
        is minimized.

        Parameters
        ----------
        x : array
            Locations where the objective function needs to be analyzed.
        verbose : bool, optional
            Printing the progress of the optimization at every iteration, `False` by default.
        material : Constitutive, optional
            The constitutive equation for the material considered, if provided it is only used to calculate the distance
            between the current solution and the exact solution.
        order : string, optional
            Defines the type of norm in the cost function. Defaults to `None`.

        Returns
        -------
        scipy.optimize.OptimizeResult
            The optimization result represented as a `OptimizeResult` object.
        """
        # Get and initialize the free parameters, that is all free rbd.
        params_initial = self.rbd[self._free_rbd]

        # If material is provided it is used to verify the way that we converge to the exact solution.
        callback = None  # Default callback.
        if material is not None:
            self._store_intermediate(x, material, params_initial, order=order)
            callback = partial(self._store_intermediate, x, material, order=order)

        # Sequential Least Squares Programming (The best optimization approach for this problem)
        options = {'ftol': 1e-30, 'disp': verbose, 'iprint': 2}
        result = minimize(self._objective_function, params_initial, args=(x, order), method='SLSQP', jac='3-point',
                          tol=0, options=options, callback=callback)

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
        result = _m(rf"$J^\Omega_0={self.error(x, order='Omega0'):4.2e}$" + "\n" +
                    rf"$J^\Omega_1={self.error(x, order='Omega1'):4.2e}$" + "\n" +
                    rf"$J^\Gamma_1={self.error(x, order='Gamma1'):4.2e}$" + "\n" +
                    rf"$e={self.compare_to_exact(x, material):4.2e}$")

        # Get the reference solution and plot it.
        if material is not None:
            x_exact, u_exact, rhs_exact = self.problem.exact(x, material)
            ax_u.plot(x_exact, u_exact, linestyle='dotted', color='grey', label=_m(r"$u^{exact}$"))
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
        ax_u.set_ylabel(_m(r"Primal field $u(x)$"))
        ax_g.set_ylabel(_m(r"Right hand side $g(x)$"))
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

        .. math:: e = \sum_{d=1}^D \int_{\Omega_d} \| u_d - u^{\text{exact}} \|^2 dx

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
        ed = np.zeros(self.problem.num_domains)
        for d in range(len(self.problem.subdomains)):
            # Find the domain filled with our subdomain d.
            start = self.problem.subdomains[d].domain[0]
            end = self.problem.subdomains[d].domain[1]
            index = np.where((x_exact >= start) & (x_exact <= end))

            # Compute the primal mismatch between the domain and reference solution in this domain.
            u_gap = (ud[d, index] - u_exact[index]) ** 2
            u_gap_spline = InterpolatedUnivariateSpline(x_exact[index], u_gap, k=3)

            # Compute the error.
            ed[d] = (u_gap_spline.integral(start, end))

        error = np.sum(ed)
        return error
