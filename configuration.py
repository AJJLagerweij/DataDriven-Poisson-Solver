r"""
The configuration is a proposed solution for the problem.

A :py:class:`~configuration.Configuration` includes the selection of :py:class:`~patch.Patch` from
:py:class:`~patch.PatchDatabase` for each of the domains described by :py:class:`~problem.Problem`. It contains the not
yet fixed rigid body motion displacement variables and the coordinate transformation as free variables, the optimal
value of which will be found through an optimization process.

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

    A configuration consists of a selection of patches and the assignment of RBD and coordinate translations. These
    variable RBD and coordinate transformations can be fixed by an optimization method, and the quality of the solution
    will be represented by an error function.

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
    rbd : array
        The rigid body displacement variables, for each domain a rigid translation and rotation.
    translation : array
        The coordinate translation for the patch in each domain, needs to stay within given bounds.
    """

    def __init__(self, problem, patches, translation_bounds):
        r"""
        A configuration is created from an admissible patch & domain combination.
        """
        self.problem = problem
        self.patches = patches
        self._rbd = np.zeros((problem.num_domains, 2))  # Private rigid body motion object.
        self._translation_bounds = np.array(translation_bounds)
        self._translation = np.zeros(problem.num_domains)
        self.translation = np.mean(translation_bounds, axis=1)
        self.rbd = np.zeros((problem.num_domains, 2))

        # Determine what parameters are free.
        self._free_rbd = np.full_like(self.rbd, True, dtype=bool)
        self._free_translation = np.full_like(self.translation, True, dtype=bool)
        for d, domain in enumerate(self.problem.subdomains):
            # Find free rigid body motions.
            if len(domain.u_bc) >= 1:
                self._free_rbd[d, 0] = False
            if len(domain.u_bc) == 2 or len(domain.du_bc) == 1:
                self._free_rbd[d, 1] = False

            # Find free translations.
            if self._translation_bounds[d][0] == self._translation_bounds[d][1]:
                self._free_translation[d] = False

        # Make some attributes semi-immutable to not alter them by accident.
        # From here on a slice can be obtained, but a slice cannot be changed.
        self._translation_bounds.setflags(write=False)
        self._translation.setflags(write=False)
        self.translation.setflags(write=False)
        self._rbd.setflags(write=False)
        self.rbd.setflags(write=False)
        self._free_rbd.setflags(write=False)
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
            # No kinematic constraints → No checks just setting self._rbd[d].
            if len(domain.u_bc) + len(domain.du_bc) == 0:
                rbd_set[d] = rbd[d]
            else:  # Kinematic constraints do exist.
                # Get the current domain settings
                patch = self.patches[d]
                t = self.translation[d]
                u_no_rbd = InterpolatedUnivariateSpline(patch.x + t, patch.u, k=1)  # Interpolated displacement field.

                # If two constraints exist on displacement, we fix translation and rotation.
                if len(domain.u_bc) == 2:
                    constraint1 = domain.u_bc[0]
                    constraint2 = domain.u_bc[1]
                    domain_middle = (domain.domain[0] + domain.domain[1]) / 2
                    u_at_constraint1 = u_no_rbd(constraint1.x)
                    u_at_constraint2 = u_no_rbd(constraint2.x)
                    rbd_du = (((constraint2.magnitude - u_at_constraint2) - (constraint1.magnitude - u_at_constraint1))
                              / (constraint2.x - constraint1.x))
                    rbd_u = constraint1.magnitude - (u_at_constraint1 + rbd_du * (constraint1.x - domain_middle))
                else:
                    # If there is a slope constraint we change the slope.
                    if len(domain.du_bc) == 1:
                        constraint = domain.du_bc[0]
                        du_at_constraint = u_no_rbd.derivative(1)(constraint.x)
                        rbd_du = constraint.magnitude - du_at_constraint
                    else:
                        rbd_du = rbd[d, 1]

                    # If there is a single displacement constraint, rotation is already set.
                    if len(domain.u_bc) == 1:
                        constraint = domain.u_bc[0]
                        domain_middle = (domain.domain[0] + domain.domain[1]) / 2
                        u_at_constraint = u_no_rbd(constraint.x) + rbd_du * (constraint.x - domain_middle)
                        rbd_u = constraint.magnitude - u_at_constraint
                    else:
                        rbd_u = rbd[d, 0]

                rbd_set[d, 0] = rbd_u
                rbd_set[d, 1] = rbd_du

        # Set the resulting rigid body motions.
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
        Set the rigid body displacement such that is satisfies the kinematic boundary conditions.

        This function performs two tasks, clipping any translation values outside the translation bounds. And correcting
        the choices of rigid body displacement variables to ensure that the configuration satisfies the kinematic
        boundary after the translation is set.

        Parameters
        ----------
        translation : array
            The translation that is to be set.
        """
        # Clip the translation values that are beyond the limits.
        self._translation = np.clip(translation, self._translation_bounds[:, 0], self._translation_bounds[:, 1])
        self._translation.setflags(write=False)

        # Correct the rigid body displacement variables accordingly.
        self.rbd = self.rbd

    def domain_displacement(self, x):
        r"""
        Calculate the displacement field of the domains :math:`u_d(x)`.

        The displacement field of each domain depends on the patch selected for it, the coordinate translation and the
        rigid body displacement. Because the displacement field is only known at discrete location these are
        interpolated to the locations of interest :math:`x`. In the end the equation for the domain displacement is:

        .. math:: u_d(x) = \mathcal{I}(x, \xi_{d,p}+t_d, u_{d,p}) + \bar{u}_d + \bar{du}_d x

        where :math:`\xi_{d,p}` and :math:`u_{d,p}` are the coordinates and displacement of patch selected for a domain,
        :math:`t_d` the coordinate translation, :math:`\bar{u}` the rbd translation and :math:`\bar{du}` the rbd
        rotation of the domain. Then :math:`\mathcal{I}(x_i, x_p, f_p)` is a interpolation function to find the
        magnitude of field :math:`f_p` known at coordinates :math:`x_p` at coordinates :math:`x_i`.

        .. note:: The displacement field of a domain outside a domain is set to `np.NaN`, as it is technically not
            defined at those locations, that is :math:`u_d(x)=` `np.NaN` :math:`\quad \forall x\notin\Omega_d`.

        Parameters
        ----------
        x : array
            The coordinates where the domain displacement has to be calculated.

        Returns
        -------
        array
            The displacement field of each domain :math:`u_d(x)`.
        """
        u_domains = np.full((self.problem.num_domains, len(x)), np.NaN, dtype=float)

        # Loop over all domains and get the displacement field at x in problem coordinates.
        for d, domain in enumerate(self.problem.subdomains):
            domain_middle = (domain.domain[0] + domain.domain[1]) / 2
            index = np.where((domain.domain[0] <= x) & (x <= domain.domain[1]))[0]
            u_no_rbd = InterpolatedUnivariateSpline(self.patches[d].x + self.translation[d], self.patches[d].u, k=3)
            u_domains[d, index] = u_no_rbd(x[index]) + self.rbd[d, 0] + self.rbd[d, 1] * (x[index] - domain_middle)
        return u_domains

    def global_displacement(self, x):
        r"""
        Calculate the global displacement :math:`\hat{u}`.

        The global displacement is based upon the state of the configuration, that is the selection of patches, rigid
        body displacement and coordinate translations. It sums the domains together weighted using a hat function:

        .. math:: \hat{u}(x) = \sum_{d=1}^D w_d(x) u_d(x)

        .. note:: The spatial derivatives of this function is not just the derivative of the domain displacement field.
            The derivatives of the weighting functions will start to play a role as well as a chain rule. If the weights
            are not smooth enough the derivative does not exist.

        .. warning:: This displacement field does perform a partition of unity superposition. Thus it violates the
            non-linearity assumption. For small errors this should be acceptable as the superposition is of two similar
            fields.

        Parameters
        ----------
        x : array
            The locations :math:`x` for which the global displacement needs to be found.

        Returns
        -------
        array
            The global displacement :math:`\hat{u}` at the requested locations :math:`x`.
        """
        # Get the weights for all domains.
        w = self.problem.weights(x)
        u_domain = self.domain_displacement(x)
        u_global = np.nansum(w * u_domain, axis=0)
        return u_global

    def equilibrium(self):
        r"""
        Verify whether this configuration satisfies the external equilibrium equations.

        In this type of beam problem two equilibrium equations have to be verified, the external shear force equilibrium
        and the external moment equilibrium.

        .. math::
            \mathbin{+\mathord\uparrow} \sum F = \int_\Omega V(x) \, dx = 0 \qquad \text{and} \qquad
            \mathbin{\stackrel{\curvearrowleft}{+}} \sum M  = \int_\Omega V(x)\cdot x + M(x)\, dx = 0

        The calculations first compute the contributions of the static boundary to the equilibrium equations, this can
        be evaluated exactly as the static state at the static boundary is exactly formulated in the problem definition.
        Then on the kinematic boundary the reaction forces and moments are extracted from the external forces that were
        applied to the patches chosen, which are then added to the global equilibrium calculations.

        .. note:: Technically the reaction forces are dependent of the coordinate translation of each domain which are
            unknown before the optimization is ran. But in this case we only have kinematic point constraints. And for
            each patch there are two options, the static fields are constants and hence the translations are free or
            the static fields are fluctuating (linear or with punctures) in which case the translation is fixed. Both
            result in the same, the equilibrium will not change with the optimization as it is either constant with
            translation or the translation itself is a constant. Hence the equilibrium equation can be evaluated before
            the optimization, this will greatly reduce the amount of configurations that have to be optimized for. This
            is NOT applicable to 2-dimensional problem.

        Returns
        -------
        bool
            `True` if the configuration is in external equilibrium, `False` otherwise.
        """
        # Start with no loads.
        M_global = 0
        V_global = 0

        # Calculate the contribution of the static boundaries specified in the problem formulation.
        for constraint in self.problem.M_bc:  # Moment on the moment boundary.
            M, Mx = constraint.global_equilibrium_contribution()
            M_global -= M

        for constraint in self.problem.V_bc:  # Moment and shear due to shear boundary.
            V, Vx = constraint.global_equilibrium_contribution()
            M_global += Vx
            V_global += V

        # We'll add to this the contribution due to the patch selected at the kinematic constraint.
        # But this external loading is only known per domain, some constraints happen to be in an overlapping region.
        # That is the reaction forces will appear twice, once in both domains, but this is solved using the weights of
        # the partition of unity.
        # FIXME: The InterpolatedUnivariateSpline is a bad choice, as the changes in load might be non-smooth.
        for d, domain in enumerate(self.problem.subdomains):
            for constraint in domain.du_bc:  # Get the reaction moment at the slope constraint.
                # Constraint location and the domain weight at that location.
                x = constraint.x
                w = self.problem.weights(np.array([x]))

                # Reaction moment due to slope constraint.
                M = InterpolatedUnivariateSpline(self.patches[d].x + self.translation[d], self.patches[d].M, k=2)
                M_global -= float(w[d] * M(x))

            for constraint in domain.u_bc:  # Get the reaction forces at the displacement constraint.
                # Constraint location and the domain weight at that location.
                x = constraint.x
                w = self.problem.weights(np.array([x]))

                # Reaction force due to displacement constraint.
                V = InterpolatedUnivariateSpline(self.patches[d].x + self.translation[d], self.patches[d].V, k=1)
                M_global += float(w[d] * V(x) * x)
                V_global += float(w[d] * V(x))

        # If both M_global and V_global are zero then the configuration is in equilibrium.
        output = (np.isclose(M_global, 0) and np.isclose(V_global, 0))
        return output

    def error(self, x):
        r"""
        Calculate the data-driven error.

        The error represents how wel the displacement fields in the overlapping areas match. The lower the error the
        more the displacement fields in the overlapping areas agree with each other.

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
        ud = self.domain_displacement(x)

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

                        # Verify that the subdomains are actually overlapping, and that we have more than 3 sample points.
                        if index[0].shape[0] > 3:
                            u_gap = InterpolatedUnivariateSpline(x[index], (ud[a, index] - ud[b, index])**2, k=3)
                            error += u_gap.integral(overlap_start, overlap_end)

                        else:
                            raise ValueError("Insufficient sample points in overlap.")

        return error

    def _error2(self, x):
        """Experimenting with a different error, one without overlapping, but touching domains."""
        # Get the domain displacement field.
        ud = self.domain_displacement(x)

        # Calculate the error.
        error = 0
        # Compute missmatch norms on the overlap, no weighting functions used.
        for a in range(len(self.problem.subdomains)):
            for b in range(len(self.problem.subdomains)):
                # Overlap between domain a and b, to ensure not to count everything twice, we do b > a.
                if b > a:
                    # Verify whether the end of domain a is the start of domain b.
                    start_a = self.problem.subdomains[a].domain[0]
                    end_a = self.problem.subdomains[a].domain[1]
                    start_b = self.problem.subdomains[b].domain[0]
                    end_b = self.problem.subdomains[b].domain[1]

                    if end_a == start_b:
                        x_interface = end_a

                        # Calculate displacement and slope of domains a and b at the x_interface.
                        index_a = np.where((start_a <= x) & (x <= end_a))
                        u_a = InterpolatedUnivariateSpline(x[index_a], ud[a, index_a], k=2)
                        u_a_interface = u_a(x_interface)
                        du_a_interface = u_a.derivative(1)(x_interface)

                        index_b = np.where((start_b <= x) & (x <= end_b))
                        u_b = InterpolatedUnivariateSpline(x[index_b], ud[b, index_b], k=2)
                        u_b_interface = u_b(x_interface)
                        du_b_interface = u_b.derivative(1)(x_interface)

                        # Calculate error.
                        error += (u_a_interface - u_b_interface)**2 + (du_a_interface - du_b_interface)**2

        return error

    def _error2moment(self, x):
        """Experimenting with a different error, one without overlapping, but touching domains."""
        # Get the domain displacement field.
        md = self.domain_moment(x)

        # Calculate the error.
        error = 0
        # Compute missmatch where the domains touch.
        for a in range(len(self.problem.subdomains)):
            for b in range(len(self.problem.subdomains)):
                # Overlap between domain a and b, to ensure not to count everything twice, we do b > a.
                if b > a:
                    # Verify whether the end of domain a is the start of domain b.
                    start_a = self.problem.subdomains[a].domain[0]
                    end_a = self.problem.subdomains[a].domain[1]
                    start_b = self.problem.subdomains[b].domain[0]
                    end_b = self.problem.subdomains[b].domain[1]

                    if end_a == start_b:
                        x_interface = end_a

                        # Calculate displacement and slope of domains a and b at the x_interface.
                        index_a = np.where((start_a <= x) & (x <= end_a))
                        m_a = InterpolatedUnivariateSpline(x[index_a], md[a, index_a], k=2)
                        m_a_interface = m_a(x_interface)
                        dm_a_interface = m_a.derivative(1)(x_interface)

                        index_b = np.where((start_b <= x) & (x <= end_b))
                        m_b = InterpolatedUnivariateSpline(x[index_b], md[b, index_b], k=2)
                        m_b_interface = m_b(x_interface)
                        dm_b_interface = m_b.derivative(1)(x_interface)

                        # Calculate error.
                        error += (m_a_interface - m_b_interface) ** 2 + (dm_a_interface - dm_b_interface) ** 2

        return error

    def error_alternative(self, x):
        r"""
        Calculate alternative error norms for the configuration.

        An error represents how wel the domain decomposition satisfies the kinematic and static requirements.

        Purely Kinematic Norms
        ----------------------
        These norms only consider kinematic matching on the overlap areas. All these norms equal 0 for the exact
        solution. As these norms are also applicable to 2D/3D solids the :math:`\nabla` icon is used to indicate spacial
        derivatives. These are the easiest to quantify as it does not require any static information. After all, internal
        statics cannot be measured directly.

        The Euclidean norm :math:`L^2(u)` (also called the :math:`H^0(u)` norm) in the overlapping region.

        .. math:: L^2(u) = \sum_{a=1}^D \sum_{b>a}^D \sqrt{\int_{\Omega_a\cap\Omega_b} \| u_a - u_b \|^2 dx}

        The first Hilbert norm :math:`H^1(u)` in the overlapping regions.

        .. math:: H^1(u) = \sum_{a=1}^D \sum_{b>a}^D \sqrt{\int_{\Omega_a\cap\Omega_b} \| u_a - u_b \|^2
            + \| \nabla u_a - \nabla u_b\|^2 dx}

        The second Hilbert norm :math:`H^2(u)` in the overlapping regions.

        .. math:: H^2(u) = \sum_{a=1}^D \sum_{b>a}^D \sqrt{\int_{\Omega_a\cap\Omega_b} \| u_a - u_b \|^2
            + \| \nabla u_a - \nabla u_b\|^2 + \| \nabla^2 u_a - \nabla^2 u_b \|^2 dx}

        The third Hilbert norm :math:`H^3(u)` in the overlapping regions.

        .. math:: H^3(u) = \sum_{a=1}^D \sum_{b>a}^D \sqrt{\int_{\Omega_a\cap\Omega_b} \| u_a - u_b \|^2
            + \| \nabla u_a - \nabla u_b\|^2 + \| \nabla^2 u_a - \nabla^2 u_b \|^2
            + \| \nabla^3 u_a - \nabla^3 u_b \|^2 dx}

        Purely Static Norms
        -------------------
        The following norm only depends on the statics. Hence it cannot be used to find the rigid body motion
        components, as these will not affect statics.

        The Euclidean norm of the moment :math:`L^2(M)` in the overlapping region.

        .. math:: L^2(M) = \sum_{a=1}^D \sum_{b>a}^D \sqrt{\int_{\Omega_a\cap\Omega_b} \| M_a - M_b \|^2 dx}

        The Euclidean norm comparing the 'global internal moment' to the domain moments. This global internal moment
        is computed from the external reaction forces, which is only possible if all the external forces are in
        equilibrium with each other. For more details see :py:meth:`self.global_moment`.

        .. math:: L^2(M_d-M^\text{global}) = \sum_{d=1}^{D} \sqrt{\int_{\Omega_d} \| M_d - M^\text{global} \|^2 dx}

        Norms combining patch statics and kinematics in the overlapping region
        -----------------------------------------------------------------------
        These norms require static information, and can thus be related to some energy. The required internal moments
        information is available for each patch, as the reaction forces/moments were measured during testing. This
        information cannot be obtained for 2D or 3D solids, and only applicable to beams, trusses and other 1D
        structures.

        A norm characterizing the strain in energy missmatch on the overlapping region. The three parts of this norm are
        returned separately, the first term as `E_Deltamddu`, the second term as `L2` and the last one as `ML2`.

        .. math:: \mathcal{E}_{\Delta M u''} = \sum_{a=1}^D \sum_{b>a}^D \frac{1}{2}
            \sqrt{\int_{\Omega_a\cap\Omega_b} M_a u''_a - M_b u''_b
            + \lambda_u (u_a - u_b)^2 + \lambda_M (M_a - M_b)^2 dx}

        Parameters
        ----------
        x : array
            The locations where the error relation is evaluated.

        Returns
        -------
        L2u, H1u, H2u, H3u, L2M, E_Deltamddu: float
            The different error norms that could be used to qualify the solution.
        """
        # Get fields for current state of configuration.
        ud = self.domain_displacement(x)
        md = self.domain_moment(x)

        # Compute missmatch norms on the overlap, no weighting functions used.
        L2u, H1u, H2u, H3u, L2M = 0, 0, 0, 0, 0
        for a in range(len(self.problem.subdomains)):
            for b in range(len(self.problem.subdomains)):
                # Overlap between domain a and b, to ensure not to count everything twice, we do b >a.
                if b > a:
                    # Find the locations of the overlap.
                    overlap_end = self.problem.subdomains[a].domain[1]
                    overlap_start = self.problem.subdomains[b].domain[0]
                    index = np.where((x >= overlap_start) & (x <= overlap_end))

                    # Verify that the subdomains are actually overlapping, and that we have more than 3 sample points.
                    if index[0].shape[0] > 3:
                        # Get the mismatch between the two fields.
                        u_gap = InterpolatedUnivariateSpline(x[index], ud[a, index] - ud[b, index], k=3)

                        # Calculate the L2(u) norm.
                        L2u_field = (u_gap(x[index]) ** 2)
                        L2u_spline = InterpolatedUnivariateSpline(x[index], L2u_field, k=3)
                        L2u += np.sqrt(L2u_spline.integral(overlap_start, overlap_end))

                        # Calculate the H1(u) norm.
                        uH1u_field = (u_gap(x[index]) ** 2
                                      + u_gap.derivative(1)(x[index]) ** 2)
                        H1u_spline = InterpolatedUnivariateSpline(x[index], uH1u_field, k=3)
                        H1u += np.sqrt(H1u_spline.integral(overlap_start, overlap_end))

                        # Calculate the H2(u) norm.
                        H2u_field = (u_gap(x[index]) ** 2
                                     + u_gap.derivative(1)(x[index]) ** 2
                                     + u_gap.derivative(2)(x[index]) ** 2)
                        H2u_spline = InterpolatedUnivariateSpline(x[index], H2u_field, k=3)
                        H2u += np.sqrt(H2u_spline.integral(overlap_start, overlap_end))

                        # Calculate the H3(u) norm.
                        H3u_field = (u_gap(x[index]) ** 2
                                     + u_gap.derivative(1)(x[index]) ** 2
                                     + u_gap.derivative(2)(x[index]) ** 2
                                     + u_gap.derivative(3)(x[index]) ** 2)
                        H3u_spline = InterpolatedUnivariateSpline(x[index], H3u_field, k=3)
                        H3u += np.sqrt(H3u_spline.integral(overlap_start, overlap_end))

                        # Calculate the L2(M) mismatch in domain moments.
                        L2M_field = ((md[a, index] - md[b, index]) ** 2)
                        L2M_spline = InterpolatedUnivariateSpline(x[index], L2M_field, k=3)
                        L2M += np.sqrt(L2M_spline.integral(overlap_start, overlap_end))

        # Compute Energy based upon the overlap missmatch.
        E_Deltamddu = 0
        for a in range(len(self.problem.subdomains)):
            for b in range(len(self.problem.subdomains)):
                # Overlap between domain a and b, to ensure not to count everything twice, we do b > a.
                if b > a:
                    # Find the locations of the overlap.
                    overlap_end = self.problem.subdomains[a].domain[1]
                    overlap_start = self.problem.subdomains[b].domain[0]
                    index = np.where((x >= overlap_start) & (x <= overlap_end))

                    # Verify that the subdomains are actually overlapping, and that we have more than 3 sample points.
                    if index[0].shape[0] > 3:
                        # Get the mismatch between the two fields.
                        u_gap = InterpolatedUnivariateSpline(x[index], ud[a][index] - ud[b][index], k=3)

                        # Created moment weighted curvature and shear weighted slope missmach.
                        ddu_a = InterpolatedUnivariateSpline(x[index], ud[a][index], k=3).derivative(2)
                        ddu_b = InterpolatedUnivariateSpline(x[index], ud[b][index], k=3).derivative(2)
                        mddu_gap = ((md[a][index] * ddu_a(x[index]) - md[b][index] * ddu_b(x[index]))**2)
                        mddu_gap_spine = InterpolatedUnivariateSpline(x[index], mddu_gap, k=3)
                        E_Deltamddu += mddu_gap_spine.integral(overlap_start, overlap_end)

        # Compute the mismatch between the global and domain displacement fields.
        M_global = self.global_moment(x)
        L2M_global = 0
        for d in range(len(self.problem.subdomains)):
            # Find the domain filled with our subdomain d.
            start = self.problem.subdomains[d].domain[0]
            end = self.problem.subdomains[d].domain[1]
            index = np.where((x >= start) & (x <= end))

            # Compute the displacement mismatch between the domain and reference solution in this domain.
            M_global_gap = (md[d, index] - M_global[index])**2
            L2M_global_spine = InterpolatedUnivariateSpline(x[index], M_global_gap, k=3)
            L2M_global += np.sqrt(L2M_global_spine.integral(start, end))

        return L2u, H1u, H2u, H3u, L2M, L2M_global, E_Deltamddu

    def domain_moment(self, x):
        r"""
        Calculate the internal moment field of the domains :math:`M_d(x)`.

        The moment field of each domain depends on the patch selected for it and the coordinate translation. Because the
        displacement field is only known at discrete location these are interpolated to the locations of interest
        :math:`x`. In the end the equation for the internal domain moment is:

        .. math:: M_d(x) = \mathcal{I}(x, \xi_{d,p} + t_d, M_{d,p})

        where :math:`\xi_{d,p}` and :math:`u_{d,p}` are the coordinates and displacement of patch selected for the
        domain, :math:`t_d` the coordinate translation. Then :math:`\mathcal{I}(x_i, x_p, f_p)` is a interpolation
        function to find the magnitude of field :math:`f_p` known at coordinates :math:`x_p` at coordinates :math:`x_i`.

        .. note:: The moment field of a domain outside that domain is set to `np.NaN`, as it is technically not
            defined at those locations, that is :math:`M_d(x)=` `np.NaN` :math:`\quad \forall x\notin\Omega_d`.

        Parameters
        ----------
        x : array
            The coordinates where the domain displacement has to be calculated.

        Returns
        -------
        array
            The moment field of each domain :math:`M_d(x)`.
        """
        M_domains = np.full((self.problem.num_domains, len(x)), np.NaN, dtype=float)

        # Loop over all domains and get the displacement field at x in problem coordinates.
        for d, domain in enumerate(self.problem.subdomains):
            index = np.where((domain.domain[0] <= x) & (x <= domain.domain[1]))[0]
            m = InterpolatedUnivariateSpline(self.patches[d].x + self.translation[d], self.patches[d].M_int, k=1)
            M_domains[d, index] = m(x[index])
        return M_domains

    def global_moment(self, x):
        r"""
        Obtain the internal moment :math:`\hat{M}` at location `x`.

        As the external loads and reaction forces are known, the problem is statically determined. Hence, the internal
        moments can be obtained.

        .. note:: This is not generically possible, only if the external loads and reaction forces are in equilibrium.

        .. note:: This is not the internal moment due to the curvature and constitutive equation.

        Returns
        -------
        array
            Internal moment at every location `x`.
        """
        # raise DeprecationWarning("Internal moment is depricated, as the physical meaning is comprimised. The moment"
        #                          " is not related to the moment of the patches selected but the internal moment due to"
        #                          " the reaction forces.")

        # Create the moment object.
        m_global = np.zeros_like(x)

        # Check whether the external loads are in equilibrium.
        if self.equilibrium():
            # Get contribution due to external shears applied t the problem
            for constraint in self.problem.M_bc:  # Moment due to moment boundary.
                M, Mx = constraint.global_internal_contribution(x)
                m_global -= M

            for constraint in self.problem.V_bc:  # Moment due to shear boundary.
                V, Vx = constraint.global_internal_contribution(x)
                m_global += Vx

            # Get the reaction shear from the patches.
            for d, domain in enumerate(self.problem.subdomains):
                for constraint in domain.du_bc:  # Get the reaction moment at the kinematic slope constraint.
                    # Constraint location and the domain weight at that location.
                    w = self.problem.weights(np.array([constraint.x]))

                    # Get the reaction forces at the displacement constraint.
                    F = InterpolatedUnivariateSpline(self.patches[d].x + self.translation[d], self.patches[d].M, k=1)
                    M = float(w[d] * F(constraint.x))
                    m_global[x >= constraint.x] -= M

                for constraint in domain.u_bc:  # Get the reaction moment at the displacement constraint.
                    # Constraint location and the domain weight at that location.
                    w = self.problem.weights(np.array([constraint.x]))

                    # Get the reaction forces at the displacement constraint.
                    # Reaction force due to displacement constraint.
                    F = InterpolatedUnivariateSpline(self.patches[d].x + self.translation[d], self.patches[d].V, k=1)
                    where_contribute = np.where(x >= constraint.x)
                    M = w[d] * F(constraint.x) * (x[where_contribute] - constraint.x)
                    m_global[where_contribute] += M

        else:
            m_global = np.full_like(x, np.NaN)
        return m_global

    def domain_shear(self, x):
        r"""
        Calculate the internal shear field of the domains :math:`V_d(x)`.

        The shear field of each domain depends on the patch selected for it and the coordinate translation. Because the
        displacement field is only known at discrete location these are interpolated to the locations of interest
        :math:`x`. In the end the equation for the internal domain moment is:

        .. math:: V_d(x) = \mathcal{I}(x, \xi_{d,p} + t_d, V_{d,p})

        where :math:`\xi_{d,p}` and :math:`u_{d,p}` are the coordinates and displacement of patch selected for the
        domain, :math:`t_d` the coordinate translation. Then :math:`\mathcal{I}(x_i, x_p, f_p)` is a interpolation
        function to find the magnitude of field :math:`f_p` known at coordinates :math:`x_p` at coordinates :math:`x_i`.

        .. note:: The shear field of a domain outside that domain is set to `np.NaN`, as it is technically not
            defined at those locations, that is :math:`M_d(x)=` `np.NaN` :math:`\quad \forall x\notin\Omega_d`.

        Parameters
        ----------
        x : array
            The coordinates where the domain displacement has to be calculated.

        Returns
        -------
        array
            The shear field of each domain :math:`V_d(x)`.
        """
        V_domains = np.full((self.problem.num_domains, len(x)), np.NaN, dtype=float)

        # Loop over all domains and get the displacement field at x in problem coordinates.
        for d, domain in enumerate(self.problem.subdomains):
            index = np.where((domain.domain[0] <= x) & (x <= domain.domain[1]))[0]
            v = InterpolatedUnivariateSpline(self.patches[d].x + self.translation[d], self.patches[d].V_int, k=2)
            V_domains[d, index] = v(x[index])
        return V_domains

    def global_shear(self, x):
        r"""
        Get the internal shear :math:`\hat{V}` at location `x`.

        As the external loads and reaction forces are known, the problem is statically determined. Hence, the internal
        moments can be obtained.

        .. note:: This is not generically possible, only if the external loads and reaction forces are in equilibrium.

        .. note:: This is not the internal shear due to the curvature and constitutive equation and it's derivative.

        Parameters
        ----------
        x : array
            If provided, the locations in local coordinates where the internal shear is calculated.

        Returns
        -------
        array
            Internal shear at every location `x`.
        """
        # raise DeprecationWarning("Internal shear is depricated, as the physical meaning is comprimised. The shear is "
        #                          "not related to the shear of the patches selected but the internal shear due to the "
        #                          "reaction forces.")

        # Create shear object.
        v_global = np.zeros_like(x)

        # Check whether the external loads are in equilibrium.
        if self.equilibrium():
            # Get contribution due to external shears applied t the problem
            for constraint in self.problem.V_bc:  # Shear due to shear boundary.
                V, Vx = constraint.global_internal_contribution(x)
                v_global += V

            # Get the reaction shear.
            for d, domain in enumerate(self.problem.subdomains):
                for constraint in domain.du_bc:  # Get the reaction shear at the displacement constraint.
                    # Constraint location and the domain weight at that location.
                    w = self.problem.weights(np.array([constraint.x]))

                    # Get the reaction forces at the displacement constraint.
                    F = InterpolatedUnivariateSpline(self.patches[d].x + self.translation[d], self.patches[d].V, k=1)
                    V = float(w[d] * F(constraint.x))
                    v_global[x >= constraint.x] += V

        else:
            v_global = np.full_like(x, np.NaN)

        return v_global

    def potential_energy(self, x):
        r"""
        Total potential energy of the configuration.

        The total potential energy of a beam is the bending energy, it requires the curvature :math:`\kappa_d''` and the
        internal moment :math:`M_d` of each of the subdomains. The curvature can simply be obtained from the second
        derivative of the displacement and is a purely kinematic object. However computing the internal moment is a
        static quantity. It can be obtained for beam problems, but is not generally availble for 2D/3D solids.

        .. math::
            \Pi^p = U - W_f

            \text{where:} U = \int_{\Omega} \frac{1}{2} \kappa(x)M(x) dx =
            \sum_{d=1}^D \int_{\Omega_d}} \frac{1}{2} \kappa_d(x)M_d(x) dx

            \text{where:} W = \sum_{i=1}^{N_V} \bar{V}_i \, \hat{u}(x_i) - \sum_{i=1}^{N_M} \bar{M}_i \, \hat{\theta}(x_i)

        where :math:`N_V` is the number of shear boundaries, each denoted as :math:`\bar{V}_i`, and :math:`N_M` the
        number of moment boundaries, each moment denoted as :math:`\bar{M}_i`.

        .. note:: The TPE requires internal statics for each of the domains to be known.

        .. warning:: Currently this assumes linear elasticity.

        Parameters
        ----------
        x : array
            The locations where the bending energy is evaluated.

        Returns
        -------
        float
            The total potential energy.
        """
        # Get the domain displacement.
        u_domains = self.domain_displacement(x)
        m_domains = self.domain_moment(x)

        # Create a function for the local bending energy and integrate.
        internal_energy = 0
        for d, domain in enumerate(self.problem.subdomains):
            # Obtain the domain.
            start = domain.domain[0]
            end = domain.domain[1]
            index = np.where((start <= x) & (x <= end))

            # Get the displacement second derivative.
            ud = u_domains[d, index]
            ud_xx = InterpolatedUnivariateSpline(x[index], ud, k=3).derivative(2)(x[index])

            # Get the moment.
            md = m_domains[d, index]

            # Get the strain energy through the strain energy density.
            strain_energy_density = 0.5 * md * ud_xx
            strain_energy_d = InterpolatedUnivariateSpline(x[index], strain_energy_density, k=3).integral(start, end)
            internal_energy += strain_energy_d

        # Compute energy due to the boundary conditions.
        boundary_energy = 0

        # Get the global displacement field at locations x for the current configuration.
        u = self.global_displacement(x)
        u_interpolated = InterpolatedUnivariateSpline(x, u, k=4)
        du = u_interpolated.derivative(1)(x)

        # Energy on the moment boundary.
        for constraint in self.problem.M_bc:
            boundary_energy += constraint.energy(x, du)

        # Energy on the prescribed shear boundary.
        for constraint in self.problem.V_bc:
            boundary_energy += constraint.energy(x, u)

        # Compute the total contributions.
        total_potential_energy = internal_energy - boundary_energy
        return total_potential_energy

    def _objective_function(self, params, x):
        """
        The objective function that will be minimized.

        The optimization will only act on the free parameters, this function will pack and unpack these correctly into
        the attributes.

        Parameters
        ----------
        params : array
            The input parameters consisting of both the rbd variables and the translations.
        x : array
            Locations where the objective function has to be evaluated.

        Returns
        -------
        float
            Magnitude of the minimization function.
        """
        # Unpack the rbd parameters.
        rbd = np.copy(self.rbd)
        rbd[self._free_rbd] = params[:np.count_nonzero(self._free_rbd)]
        self.rbd = rbd

        # Unpack the translation parameters.
        translation = np.copy(self.translation)
        translation[self._free_translation] = params[np.count_nonzero(self._free_rbd):]
        self.translation = translation

        # Calculate error norm.
        # error = self.error(x)
        weighing_factor = 1e8
        # error = self._error2(x) + weighing_factor * self._error2moment(x)
        error = self.potential_energy(x)
        return error

    def _c1_continuity_measure(self, x, params):
        """
        The measure that verifies continuity across all the patches.

        If this function equals 0 than the configuration is C1 continuous.

        Parameters
        ----------
        x : array
            Locations where the objective function has to be evaluated.
        params : array
            The input parameters consisting of both the rbd variables and the translations.

        Returns
        -------
        float
            Magnitude of the C1 continuity violation across domain interfaces.
        """
        # Unpack the rbd parameters.
        rbd = np.copy(self.rbd)
        rbd[self._free_rbd] = params[:np.count_nonzero(self._free_rbd)]
        self.rbd = rbd

        # Calculate C1 continuity norm.
        error = self._error2(x)
        return error

    def optimize(self, x, verbose=False):
        r"""
        Find the optimal rigid body displacement and coordinate translations variables.

        A bounded B-BFGS-M optimization is used, which tries to find the best rigid body displacement and coordinate
        transformations such that the error is minimized. However, some of the rbd's and translations are fixed
        (these will not be exposed to the optimization) and all free translations have bounds, these bounds will be
        satisfied.

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
        # Get and initialize the free parameters, that is all rbd and the free translations.
        params_initial = np.hstack((self.rbd[self._free_rbd], self.translation[self._free_translation]))
        lb = np.hstack((np.array([-np.inf] * np.count_nonzero(self._free_rbd)),
                        np.array(self._translation_bounds)[self._free_translation, 0]))
        ub = np.hstack((np.array([np.inf] * np.count_nonzero(self._free_rbd)),
                        np.array(self._translation_bounds)[self._free_translation, 1]))
        bounds = Bounds(lb, ub)

        # Define the non-linear constraint that requires C1 continuity.
        c1_continuity = partial(self._c1_continuity_measure, x)
        constraint = NonlinearConstraint(c1_continuity, 0, 0, jac='3-point')

        # Sequential Least Squares Programming (The best optimization approach for this problem)
        options = {'ftol': 1e-25, 'maxiter': 20000, 'disp': verbose, 'iprint': 2}
        result = minimize(self._objective_function, params_initial, args=x, bounds=bounds, constraints=constraint,
                          method='SLSQP', tol=0, jac='3-point', options=options)

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
        ud = self.domain_displacement(x)
        md = self.domain_moment(x)
        vd = self.domain_shear(x)

        # Create figures and axes.
        fig, axis = plt.subplots(3, 1, sharex='col', figsize=(10, 6),
                                 gridspec_kw={'height_ratios': [4, 2, 2], 'hspace': 0})
        plt.suptitle(title)
        ax_u = axis[0]  # Displacement field
        ax_m = axis[1]  # Internal load axis for moment
        ax_mddu = axis[2]  # Energy error axis

        # Create print statement.
        # L2u = self.error_alternative(x)[0]
        # L2M = self.error_alternative(x)[4]
        # L2Mddu = self.error_alternative(x)[5]

        result = _m(rf"$\mathcal{{E}}_{{u}}={self._error2(x):4.2e}$")
        result += _m("\n" + rf"$\mathcal{{E}}_{{M}}={self._error2moment(x):10.8f}$")
        result += _m("\n" + rf"$\Pi^p={self.potential_energy(x):10.8f}$")
        # result += _m("\n" + rf"$\sqrt{{(M_au''_a - M_u''_b)^2}} = {L2Mddu:10.8f}$")
        # result += _m("\n" + rf"$\frac{{1}}{{2}}\sqrt{{(u_a - u_b)^2}} = {L2u:10.8f}$")
        # result += _m("\n" + rf"$\frac{{1}}{{2}}\sqrt{{(M_a - M_b)^2}} = {L2M:10.8f}$")

        # Get the reference solution and plot it.
        if material is not None:
            x_exact, u_exact, M_exact, V_exact = self.problem.exact(x, material)
            ax_u.plot(x_exact, u_exact, color='grey', label=_m(r"$u^{EB}$"))
            ax_m.plot(x_exact, M_exact, color='grey', label=_m(r"$M^{EB}$"))
            ax_mddu.plot(x_exact, V_exact, color='grey', label=_m(r"$V^{EB}$"))

        # Plot domain displacement, moment and bending energy density fields.
        for d, domain in enumerate(self.problem.subdomains):
            # Get domain information.
            index = np.where((domain.domain[0] <= x) & (x <= domain.domain[1]))[0]
            ax_u.plot(x[index], ud[d, index], label=_m(rf"$u_{d}$"))
            ax_m.plot(x[index], md[d, index], label=_m(rf"$M_{d}$"))
            ax_mddu.plot(x[index], vd[d, index], label=_m(rf"$V'_{d}$"))

        # Plot bending energy density mismatch.
        # mddu_gap = np.zeros_like(x)
        # for a in range(len(self.problem.subdomains)):
        #     for b in range(len(self.problem.subdomains)):
        #         # Overlap between domain a and b, to ensure not to count everything twice, we do b >a.
        #         if b > a:
        #             # Find the locations of the overlap.
        #             overlap_end = self.problem.subdomains[a].domain[1]
        #             overlap_start = self.problem.subdomains[b].domain[0]
        #             index = np.where((x >= overlap_start) & (x <= overlap_end))
        #
        #             # Verify that the subdomains are actually overlapping.
        #             if np.any(index):
        #                 ddu_a = InterpolatedUnivariateSpline(x[index], ud[a][index], k=3).derivative(2)
        #                 ddu_b = InterpolatedUnivariateSpline(x[index], ud[b][index], k=3).derivative(2)
        #                 mddu_gap[index] = np.abs(md[a][index] * ddu_a(x[index]) - md[b][index] * ddu_b(x[index]))
        #
        # ax_mddu.plot(x, mddu_gap, color='red', label=f"Energy Gap")

        # Fix axis and add legend.
        ax_mddu.set_xlim(x[0], x[-1])
        ax_u.annotate(result, xy=(0.02, 0.02), xycoords='axes fraction', ha='left', va='bottom')
        ax_u.legend(loc=1, frameon=False)
        ax_m.legend(loc=1, frameon=False)
        ax_mddu.legend(loc=1, frameon=False)

        # Add axis labels.
        ax_u.set_ylabel(_m(r"Displacement $u$ in mm"))
        ax_m.set_ylabel(_m(r"Moment $M$ in kNmm"))
        ax_mddu.set_ylabel(_m(r"Energy density $e$ in kN"))
        ax_mddu.set_xlabel(_m(r"Location $x$ in mm"))

        # Save the plot as image.
        if path is not None:
            fig.savefig(path + title + '.png')
            fig.savefig(path + title + '.svg')
        return ax_u, ax_m, ax_mddu

    def compare_to_exact(self, x, material):
        r"""
        Compare this configuration to the exact solution.

        This is measured as the Euclidean norm :math:`L^2` between the domain displacements and the reference solution
        displacement.

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
            The :math:`L^2` between the domain displacements and the reference solution displacement.
        """
        # Get fields for the exact solution and the current state of configuration.
        x_ex, u_ex, m_ex, v_ex = self.problem.exact(x, material)
        ud = self.domain_displacement(x_ex)

        # Compare the displacement fields:
        error = 0
        for d in range(len(self.problem.subdomains)):
            # Find the domain filled with our subdomain d.
            start = self.problem.subdomains[d].domain[0]
            end = self.problem.subdomains[d].domain[1]
            index = np.where((x_ex >= start) & (x_ex <= end))

            # Compute the displacement mismatch between the domain and reference solution in this domain.
            u_gap = (ud[d, index] - u_ex[index])**2
            u_gap_spline = InterpolatedUnivariateSpline(x_ex[index], u_gap, k=3)

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

        # Obtain the admissible patch - domain combinations and their translation.
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

    def equilibrium(self, parallel=False, remove=True):
        r"""
        Perform the equilibrium calculation on all configurations.

        For all configurations the equilibrium equation is evaluated, all configurations that are not in global external
        equilibrium are deleted from the database.

        .. warning:: Changes the object in place, the all non external equilibrium configurations will be removed.

        Parameters
        ----------
        parallel : bool, optional
            `True` if the computation should be performed in parallel, `False` if it should be in series.
        remove : bool, optional
            Whether to remove the non-equilibrium solutions.

        See Also
        --------
        Configuration.equilibrium : The function that is called for all configurations.
        """

        def equilibrium_func(config):
            """The wrapper around :py:meth:`Configuration.equilibrium`"""
            equilibrium = config.equilibrium()
            return equilibrium

        # Apply the optimization function to all configurations.
        if self.num_configurations() == 0:
            raise ValueError("There are no configurations in this database.")
        else:
            print(f"Verify the equilibrium of {self.num_configurations()} configurations.")
            if parallel:
                # Check whether parallel run was initialized before.
                if hasattr(self.database.configuration, 'parallel_apply') is False:
                    pandarallel.initialize()

                # The parallel run.
                self.database['equilibrium'] = self.database.configuration.parallel_apply(equilibrium_func)
            else:
                print("    Slow computation because execution is in series.")
                self.database['equilibrium'] = self.database.configuration.apply(equilibrium_func)

            # Remove non equilibrium configurations from database.
            if remove:
                self.database = self.database[self.database.equilibrium]

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
                [config, result.fun, config.rbd[:, 0], config.rbd[:, 1], config.translation, result.success])
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
                self.database[['configuration', 'error', 'rbd', 'du_rbd', 'translation',
                               'success']] = self.database.configuration.parallel_apply(function)
            else:
                print("    Slow computation because execution is in series.")
                self.database[['configuration', 'error', 'rbd', 'du_rbd', 'translation',
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

    def error_alternative(self, x, parallel=False):
        r"""
        Apply the alternative error calculation function to all configurations considered.

        .. warning:: Changes the object in place, a column with the error magnitude will be added to the database.

        Parameters
        ----------
        x : array
            Locations where the objective function needs to be analyzed.
        parallel : bool, optional
            `True` if the computation should be performed in parallel, `False` if it should be in series.

        See Also
        --------
        Configuration.error_alternative : The function that is called for all configurations.
        """

        # Create partial optimization function that contains all fixed information.
        def function(config):
            """The wrapper around the :py:meth:`Configuration.error_alternative` function."""
            L2u, H1u, uH2, H3u, L2M, L2Mglobal, E_Deltamddu = config.error_alternative(x)
            out = pd.Series([L2u, H1u, uH2, H3u, L2M, L2Mglobal, E_Deltamddu])
            return out

        # Apply the optimization function to all configurations.
        if self.num_configurations() == 0:
            raise ValueError("There are no configurations in this database.")
        else:
            print(f"Computing alternative Error norms for {self.num_configurations()} configurations.")
            if parallel:
                # Check whether parallel run was initialized before.
                if hasattr(self.database.configuration, 'parallel_apply') is False:
                    pandarallel.initialize()

                # The parallel run.
                self.database[['L2u', 'H1u', 'uH2', 'H3u', 'L2M', 'L2Mglobal', 'E_Deltamddu']] \
                    = self.database.configuration.parallel_apply(function)
            else:
                print("    Slow computation because execution is in series.")
                self.database[['L2u', 'H1u', 'uH2', 'H3u', 'L2M', 'L2Mglobal', 'E_Deltamddu']] \
                    = self.database.configuration.apply(function)

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

    def potential_energy(self, x, parallel=False):
        r"""
        Apply the energy calculation function to all configurations considered and sort based upon it.

        .. warning:: Currently this assumes linear elasticity.

        Parameters
        ----------
        x : array
            Locations where the objective function needs to be analyzed.
        parallel : bool, optional
            `True` if the computation should be performed in parallel, `False` if it should be in series.

        See Also
        --------
        Configuration.potential_energy : The function that is called for all configurations.
        """

        # Create partial optimization function that contains all fixed information.
        def function(config):
            """The wrapper around the :py:meth:`Configuration.potential_energy` function."""
            tpe = config.potential_energy(x)
            return tpe

        # Apply the optimization function to all configurations.
        if self.num_configurations() == 0:
            raise ValueError("There are no configurations in this database.")
        else:
            print(f"Computing the total potential energy for {self.num_configurations()} configurations.")
            if parallel:
                # Check whether parallel run was initialized before.
                if hasattr(self.database.configuration, 'parallel_apply') is False:
                    pandarallel.initialize()

                # The parallel run.
                self.database['tpe'] = self.database.configuration.parallel_apply(function)
            else:
                print("    Slow computation because execution is in series.")
                self.database['tpe'] = self.database.configuration.apply(function)

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

    def error_vs_tpe(self, x, max_num=50, parallel=False, path=None):
        r"""
        Compare the performance of a Data-Driven error tot the total potential energy.

        Parameters
        ----------
        x : array
            The locations where the state is sampled.
        max_num : int, optional
            Number of configurations in the comparison, 50 by default.
        parallel : bool, optional
            Whether to run the underlaying calculations in parallel or not, `True` by default.
        path : str, optional
            The path to which a .png and a .svg need to be saved, disabled by default.

        """
        # First it is verified that the columns for error and tpe exit in our database.
        if 'error' not in self.database.columns:
            # The computations for the error still have to be performed.
            self.error(x, parallel=parallel)

        if 'tpe' not in self.database.columns:
            # The total potential energy computation still needs to be performed.
            self.potential_energy(x, parallel=parallel)

        if 'equilibrium' not in self.database.columns:
            # The equilibrium has to be evaluated.
            self.equilibrium(parallel=parallel, remove=False)

        # Create subfigures to plot TPE vs DD error for each configuration.
        fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={"hspace": 0.0})

        # Make a series that defines the bar colours, the equilibrium solutions are red, the rest blue.
        database_clip = self.database.head(max_num).copy()
        database_clip['color'] = self.database.head(max_num)['equilibrium']
        database_clip.loc[database_clip['equilibrium'], 'color'] = 'C3'
        database_clip.loc[~database_clip['equilibrium'], 'color'] = 'C0'

        # Plot the error and potential energy for each configuration.
        database_clip.plot(y='tpe', kind='bar', ax=ax[0], color=database_clip['color'])
        database_clip.plot(y='error', kind='bar', label='Data-Driven Error', ax=ax[1], color=database_clip['color'])

        # Remove the legends and append the equilibrium vs non-equilibrium legend.
        ax[0].get_legend().remove()
        ax[1].get_legend().remove()
        legend_elements = [Patch(color='C0', label='Non-Equilibrium'),
                           Patch(color='C3', label='Equilibrium Configuration')]
        ax[1].legend(handles=legend_elements)

        # Set axis labels with titles.
        ax[0].set_ylabel('Potential Energy')
        ax[1].set_ylabel('DD Error')
        plt.show()

        if path is not None:
            fig.savefig(path + 'Comparing TPE to DD-Error.png')
            fig.savefig(path + 'Comparing TPE to DD-Error.svg')

    def save(self, filename, path='', compression="infer"):
        r"""
        Saving the DataFrame into a pickle.

        This will store the entirety of the configurations database as a pickle. The benefit is that this allows for the
        computation of optimal solutions on remote machines, then transfer the data to your machine to analyze the
        results. The optimization results (the errors, optimal rigid body motion and coordinate translation) are then
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
