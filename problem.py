r"""
A problem formulation consists of the definition of a domain, the loading conditions and a discretization.

The classes below can be called to define a variety of beam problems. They result in objects that define the coordinate
system and loading conditions. These loads are specified on the static boundary, and the displacement is constraint at
the kinematic boundary.

.. note:: Any future problem description should be a child of the :py:class:`~problem.Problem` which specifies the
    minimal information that is to be stored in a test object.

Bram Lagerweij |br|
Mechanics of Composites for Energy and Mobility |br|
KAUST |br|
2021
"""

# Import external modules.
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.special import comb
from scipy.optimize import minimize_scalar
from scipy.interpolate import InterpolatedUnivariateSpline

# Importing my own scripts.
from constraint import PointConstraint, LinearConstraint
import test


class Domain(object):
    r"""
    The generic definition of a domain consists of it's geometry and boundary conditions.

    This parent class only contains the generic formulation of the geometry and boundaries.

    Attributes
    ----------
    domain : tuple
        The start (`float`) and end (`float`) of the domain in problem coordinates.
    u_bc : list
        A list of the different boundaries where the displacement :math:`u` is constraint.
    du_bc : list
        A list of the different boundaries where the rotation :math:`\theta` is constraint.
    M_bc : list
        A list with the external moment constraints.
    V_bc : list
        A list with the external shear constraints.
    end : list
        A list defining the beam ends, at each of these ends the internal static state must at least satisfy the
        prescribed loads.
    """

    def __init__(self):
        r"""
        In general the initialization of a domain contains the boundary conditions and geometry.
        """
        self.domain = (np.NaN, np.NaN)
        self.u_bc = []
        self.du_bc = []
        self.M_bc = []
        self.V_bc = []
        self.end = []

    def plot(self):
        r"""
        Plotting the domain and the conditions specified.

        .. warning:: This plotting is not entirely trustworthy for the direction and magnitudes of the loads and
            supports. It only gives an indication. Improvement are to be made.

        Returns
        -------
        axis : matplotlib.Axis
            The axis in which the problem is drawn.
        """
        plt.figure(figsize=(10, 5))
        ax = plt.gca()
        # ax.clip_on = False

        plt.plot([self.domain[0], self.domain[1]], [0, 0], c='k', clip_on=False)

        offset = self.domain[1]/50
        for constraint in self.u_bc:
            marker = "Displacement"
            color = 'C1'
            constraint.plot(ax, marker, color, offset=offset)

        for constraint in self.du_bc:
            marker = "Slope"
            color = 'C1'
            constraint.plot(ax, marker, color, offset=offset)

        for constraint in self.M_bc:
            marker = "Moment"
            color = 'C2'
            constraint.plot(ax, marker, color, offset=-offset)

        for constraint in self.V_bc:
            marker = 'Shear'
            color = 'C3'
            constraint.plot(ax, marker, color, offset=offset)

        # Turn off the axis.
        ax.set_aspect('equal')
        ax.axis('off')


class SubDomain(Domain):
    r"""
    The definition of a subdomain.

    This parent class only contains the generic formulation of the geometry and boundaries. It extents the parent class
    by adding methods to verify the admissibility of boundary conditions.

    Parameters
    ----------
    domain : tuple
        The start (`float`) and end (`float`) of the domain in problem coordinates.
    u_bc : list
        A list of the different boundaries where the displacement :math:`u` is constraint.
    du_bc : list
        A list of the different boundaries where the rotation :math:`\theta` is constraint.
    M_bc : list
        A list with the external moment constraints.
    V_bc : list
        A list with the external shear constraints.
    end : list
        A list containing :py:class:`~constraint.PointConstraint` at the beam ends.

    Attributes
    ----------
    domain : tuple
        The start (`float`) and end (`float`) of the domain in problem coordinates.
    u_bc : list
        A list of the different boundaries where the displacement :math:`u` is constraint.
    du_bc : list
        A list of the different boundaries where the rotation :math:`\theta` is constraint.
    M_bc : list
        A list with the external moment constraints.
    V_bc : list
        A list with the external shear constraints.
    end : list
        A list defining the beam ends, at each of these ends the internal static state must at least satisfy the
        prescribed loads.
    """

    def __init__(self, domain, u_bc, du_bc, M_bc, V_bc, end):
        r"""
        Initializing a SubDomain consists of setting its attributes.
        """
        # Check number of kinematic constraints.
        if len(u_bc) + len(du_bc) > 2 or len(du_bc) > 1:
            raise ValueError(f"This domain has to many kinematic constraints, there should be at most 2 constraints "
                             f"({len(u_bc)+len(du_bc)} were specified) of which at most 1 a slope constraint "
                             f"({len(du_bc)} were specified)")

        super().__init__()
        self.domain = domain
        self.u_bc = u_bc
        self.du_bc = du_bc
        self.M_bc = M_bc
        self.V_bc = V_bc
        self.end = end

    def admissible_coordinate_transformations(self, patch):
        r"""
        Calculate the range of translations :math:`t` that move the patch into this domain.

        Parameters
        ----------
        patch : Patch
            The patch that is moved into the domain.

        Returns
        -------
        list
            A list with a tuple for the max and min translations that satisfy the constraint. This list is empty if the
            patch is to small for the domain.
        """
        # Verify whether the patch is large enough to fill the domain entirely.
        if self.domain[1]-self.domain[0] > patch.x.max()-patch.x.min():
            return []

        # Find the possible translations, anything in the range t is possible.
        translations = [(self.domain[1]-patch.x.max(), self.domain[0]-patch.x.min())]
        return translations

    def admissible_end(self, patch, translations):
        r"""
        Calculate for what translations :math:`t` the patch can satisfy the end constraint.

        Parameters
        ----------
        patch : Patch
            The patch that is considered for the domain
        translations : list
            A list with a tuple for the max and min translations that satisfy the constraint. This list is empty if the
            patch is to small for the domain.

        Returns
        -------
        list
            A list with a tuple for the max and min translations that satisfy the constraint. This list is empty if the
            patch is to small for the domain.
        """
        for constraint in self.end:
            for i, translation in enumerate(translations):
                translation = constraint.satisfy_value_free_translation(patch.x, patch.end, translation)
                translations[i:i+1] = translation  # Remove the previous range and replace it with the newly found ones.
        return translations

    def admissible_moment(self, patch, translations):
        r"""
        Calculate for which translation :math:`t` the patch can satisfy the moment bc of the domain.

        Parameters
        ----------
        patch : Patch
            The patch that is considered for the domain
        translations : list
            A list with a tuple for the max and min translations that satisfy the constraint. This list is empty if the
            patch is to small for the domain.

        Returns
        -------
        list
            A list with a tuple for the max and min translations that satisfy the constraint. This list is empty if the
            patch is to small for the domain.
        """
        # Loop over all constraints and harvest the admissible translations.
        for constraint in self.M_bc:
            for i, translation in enumerate(translations):
                translation = constraint.satisfy_value_free_translation(patch.x, patch.M, translation)
                translations[i:i+1] = translation  # Remove the previous range and replace it with the newly found ones.
        return translations

    def admissible_shear(self, patch, translations):
        """
        Calculate for which translation :math:`t` the patch can satisfy the shear bc of the domain.

        Parameters
        ----------
        patch : Patch
            The patch that is considered for the domain
        translations : list
            A list with a tuple for the max and min translations that satisfy the constraint. This list is empty if the
            patch is to small for the domain.

        Returns
        -------
        list
            A list with a tuple for the max and min translations that satisfy the constraint. This list is empty if the
            patch is to small for the domain.
        """
        # Loop over all constraints and harvest the admissible translations.
        for constraint in self.V_bc:
            for i, translation in enumerate(translations):
                translation = constraint.satisfy_value_free_translation(patch.x, patch.V, translation)
                translations[i:i+1] = translation  # Remove the previous range and replace it with the newly found ones.
        return translations

    def admissible(self, patch):
        """
        Determine the admissibility of a patch for this domain.

        .. warning:: This function only checks the static boundary conditions, which are enforced strongly. Because
            the translations are not necessarily fixed uniquely the kinematic BCs will have to be fixed every time that
            the translation changes, that is it depends on the configuration.

        Parameters
        ----------
        patch : Patch
            The patches that will be applied to this domain.

        Returns
        -------
        list
            A list with tuples for each set of discrete coordinate translations, `(tmin, tmax)` for which this patch
            satisfies the geometric and static constraints. The list is empty if the patch cannot satisfy the
            constraints.
        """
        translations = self.admissible_coordinate_transformations(patch)
        translations = self.admissible_end(patch, translations)
        translations = self.admissible_moment(patch, translations)
        translations = self.admissible_shear(patch, translations)
        return translations


class Problem(Domain, ABC):
    r"""
    The definition of a beam problem setup and the information that will be will specify the test.

    This parent class only contains the specification of this type of class and the general solver. That is the
    geometry, constraints and external loading. These properties are know at any location, not just at discrete points.
    After all, the problem is defined in an exact and continuous manner.

    This class extents the :py:class:`~problem.Domain` class by adding the capability of subdomains. For that it also
    includes a generic way to split any problem into subdomains.

    Attributes
    ----------
    subdomains : list
        The list with subdomains to the problem.
    num_domains : int
        The number of subdomains in question.
    continuity : int
        Specification of the continuity at the start and end of the subdomains.
    """
    def __init__(self):
        r"""
        In general the initialization of such a class contains the boundary conditions and geometry.
        """
        super().__init__()
        self.subdomains = []
        self.num_domains = len(self.subdomains)
        self.continuity = -1

    def split_subdomains(self, domain_length, num_domains):
        r"""
        Split the problem into subdomains.

        The problem is split into subdomains. Each subdomain consists of it's own boundary conditions and inherits is of
        the :py:class:`problem.SubDomains`.

        .. warning::
            This function should always be called before solving the problem. For robustness it should be called by the
            initialization function of the children.

        .. warning::
            This function is changing this instance in place.

        Parameters
        ----------
        domain_length : float
            Length of a subdomain.
        num_domains : int
            Number of subdomains.

        Returns
        -------
        Problem
            Returns this problem instance.
        """
        # Calculate how much the domains must overlap to fit this many domains in the length of the problem.

        if num_domains != 1:
            domain_overlap = (num_domains*domain_length + self.domain[0] - self.domain[1]) / (num_domains - 1)
        else:  # The entire problem is a single domain (used for testing purposes).
            domain_overlap = 0

        # Throw an error if the overlap is smaller equal zero.
        if domain_overlap < 0:
            raise ValueError("The overlapping area for this domain decomposition is negative, this is not possible, "
                             "choose larger or more domains.")

        # Loop over all subdomains and add them to the subdomain list.
        self.subdomains = []
        for domain_i in range(num_domains):
            # Define the domain geometry.
            domain_start = domain_i * (domain_length - domain_overlap)
            domain_end = domain_start + domain_length
            domain = (domain_start, domain_end)
            external_start = domain[0] == self.domain[0]  # The start of the subdomain intersects equals problem start.
            external_end = domain[1] == self.domain[1]  # The end of the subdomain intersects equals problem start.

            # Create the constraint lists for the subdomains.
            u_bc = []
            for constraint in self.u_bc:
                cropped_constraint = constraint.crop(domain[0], domain[1])
                if cropped_constraint is not None:
                    u_bc.append(cropped_constraint)

            du_bc = []
            for constraint in self.du_bc:
                cropped_constraint = constraint.crop(domain[0], domain[1])
                if cropped_constraint is not None:
                    du_bc.append(cropped_constraint)

            M_bc = []
            for constraint in self.M_bc:
                # For internal domains the moment constraint is open ended.
                cropped_constraint = constraint.crop(domain[0], domain[1], closed_start=external_start,
                                                     closed_end=external_end)
                if cropped_constraint is not None:
                    M_bc.append(cropped_constraint)

            V_bc = []
            for constraint in self.V_bc:
                # For internal domains the shear constraint is open ended.
                cropped_constraint = constraint.crop(domain[0], domain[1], closed_start=external_start,
                                                     closed_end=external_end)
                if cropped_constraint is not None:
                    V_bc.append(cropped_constraint)

            end = []
            for constraint in self.end:
                cropped_constraint = constraint.crop(domain[0], domain[1])
                if cropped_constraint is not None:
                    end.append(cropped_constraint)

            self.subdomains.append(SubDomain(domain, u_bc, du_bc, M_bc, V_bc, end))

        self.num_domains = len(self.subdomains)
        return self

    def domain_patch_admissibility(self, database):
        """
        Determine the admissibility of every patch in the database for each domain.

        Parameters
        ----------
        database : PatchDatabase
            The collection of patches that will be applied to the problems in this domain.

        Returns
        -------
        admissible_domain_patches : list
            For each domain the index of the admissible patches.
        domain_patch_translations : list
            For each domain (index 0) for each patch (index 1) a list with tuples for the admissible discrete coordinate
            translations, `(tmin, tmax)` for which this patch satisfies the geometric and static constraints of the
            domain.
        """
        admissible_domain_patches = []
        domain_patch_translations = []
        for d, domain in enumerate(self.subdomains):
            patch_list = []
            translations_list = []
            for p, patch in enumerate(database.database):
                translations = domain.admissible(patch)
                translations_list.append(translations)
                if len(translations) != 0:
                    # At least a single translation range is admissible.
                    patch_list.append(p)

            # Add the admissible patches for this domain to the admissible_domain_patches list.
            admissible_domain_patches.append(patch_list)
            domain_patch_translations.append(translations_list)

        return admissible_domain_patches, domain_patch_translations

    def _smoothstep(self, x, overlap_start, overlap_end):
        r"""
        Calculate the weight of a point in the overlapping region.

        In the overlapping region the weights are depended on the continuity parameter that is set. By default the
        continuity is -1, that is there is a discrete step, the weight is 1 in the interior of the domain, 0.5 on the
        overlapping regions and 0 outside the subdomain. If the continuity is 0, then there is a linear decay in the
        overlapping regions. For values 1, 2, ... there are increasingly smoother transitions between the domains. In
        the end it must be guaranteed that the sum of the domain weight equals unity, and that the value of the weight
        of a domain equals zero outside of the domain.

        .. math:: 1 = \sum_{d=1}^D w_d(x) \quad \forall x\in\Omega \quad \text{and}
            \quad w_d(x) = 0 \quad \forall x\notin \Omega_d

        Returns
        -------
        x : array
            The locations in the overlapping region where the weighting function has to determined.
        overlap_start : float
            The start of the overlapping region.
        overlap_end : float
            The end of the overlapping region.
        """
        x = np.clip((x - overlap_start) / (overlap_end - overlap_start), 0, 1)

        if self.continuity == -1:
            result = 0.5 * np.ones_like(x)
        else:
            result = 0
            for n in range(0, self.continuity + 1):
                result += comb(self.continuity + n, n) * comb(2 * self.continuity + 1, self.continuity - n) * (-x) ** n
            result *= x ** (self.continuity + 1)

        return result

    def weights(self, x):
        r"""
        Calculate partition of unity weights :math:`w(x)` that prescribe how this domain is merged into the others.

        In the overlapping regions the weights specify the contribution of each of the domains, for example for the
        global displacement field:

        .. math:: \hat{u}(x) = \sum_{d=1}^D w_d(x) u_d(x)

        Parameters
        ----------
        x : array
            The locations :math:`x` for which the weights need to be determined.

        Returns
        -------
        array
            The weights :math:`w(x)` at :math:`x`.
        """
        # Initialize the weights array.
        w = np.zeros((self.num_domains, len(x)))

        # Loop over all domains.
        for d, domain in enumerate(self.subdomains):
            domain_start = domain.domain[0]
            domain_end = domain.domain[1]

            if d != 0:  # There is a domain to the left.
                overlap_left_end = self.subdomains[d-1].domain[1]  # End previous domain.
            else:  # The first domain has no patch to the left.
                overlap_left_end = np.NaN

            if d != self.num_domains-1:  # There is a domain to the right.
                overlap_right_start = self.subdomains[d+1].domain[0]  # Start next domain.
            else:  # There is no domain to the right.
                overlap_right_start = np.NaN

            # Anything in the domain will be filled with ones.
            w[d, (domain_start <= x) & (x <= domain_end)] = 1

            # Anything in the left overlap equals smoothstep, the np.where call is empty when np.NaN is involved.
            index = np.where((domain_start <= x) & (x <= overlap_left_end))
            w[d, index] = self._smoothstep(x[index], domain_start, overlap_left_end)

            # Anything in the right overlap equals smoothstep, the np.where call is empty when np.NaN is involved.
            index = np.where((overlap_right_start <= x) & (x <= domain_end))
            w[d, index] = 1 - self._smoothstep(x[index], overlap_right_start, domain_end)

        return w

    @abstractmethod
    def exact(self, x, material):
        r"""
        The exact solution to this problem.

        Parameters
        ----------
        x : array
            The location for which the exact solution needs to be found.
        material : Constitutive
            The constitutive equation of the material considered.

        Returns
        -------
        x : array
            The locations were the exact solution was evaluated.
        u : array
            The beam displacement :math:`u(x)`.
        M : array
            The internal moment of the exact solution.
        V : array
            The internal shear of the exact solution.
        """
        pass

    def potential_energy(self, x, material):
        """
        Calculating the potential energy of the exact solution.

        Parameters
        ----------
        x : array
            Discrete locations where the internal energy will be evaluated.
        material : Constitutive
            The constitutive material that the problem is made of.
        """
        # Calculate internal fields of the problem.
        x, u, moment, shear = self.exact(x, material)

        # Process the results.
        u_interp = InterpolatedUnivariateSpline(x, u, k=2)
        du = u_interp.derivative(1)(x)
        curvature = material.curvature(moment)

        # Create a function for the local bending energy and integrate.
        internal_energy_local = InterpolatedUnivariateSpline(x, 0.5 * curvature * moment, k=4)
        internal_energy = internal_energy_local.integral(self.domain[0], self.domain[1])

        # Compute energy due to the boundary conditions.
        boundary_energy = 0

        # Energy on the moment boundary.
        for constraint in self.M_bc:
            boundary_energy += constraint.energy(x, du)

        # Energy on the prescribed shear boundary.
        for constraint in self.V_bc:
            boundary_energy += constraint.energy(x, u)

        # Compute the total contributions.
        total_potential_energy = internal_energy - boundary_energy
        return total_potential_energy


class ExampleLoadControlled(Problem):
    r"""
    An example problem representing a statically loaded beam. The tip loaded beam has simple supports at :math:`x=0` and
    :math:`x=L/4`.

    Parameters
    ----------
    length : float
        Total length of the beam.
    p_tip : float
        Magnitude of the load applied.
    domain_length : float
        Length of the subdomains.
    num_domains : int
        Number of subdomains.
    continuity : int, optional
        The continuity in the overlapping region transitioning from one domain to the other (-1, 0 or a larger integer).

    Attributes
    ----------
    domain : tuple
        The start (`float`) and end (`float`) of the domain in problem coordinates.
    u_bc : list
        A list of the different boundaries where the displacement :math:`u` is constraint.
    M_bc : list
        A list with the external moment constraints.
    V_bc : list
        A list with the external shear constraints.
    end : list
        A list defining the beam ends, at each of these ends the internal static state must at least satisfy the
        prescribed loads.
    continuity : int
        Specification of the continuity at the start and end of the subdomain.
    """

    def __init__(self, length, p_tip, domain_length, num_domains, continuity=-1):
        r"""
        A beam simply supported at x=0 and x=L/4 subjected to a load of p at the tip.
        """
        super().__init__()

        # Initialize the domain and subdomains and continuity.
        self.domain = (0, length)
        self.continuity = continuity

        # The kinematic displacement constraints are the two simple supports at x=0 and x=L/4
        self.u_bc = [PointConstraint(0, 0),
                     PointConstraint(length / 4, 0)]

        # There is no moment applied anywhere on the beam.
        self.M_bc = [LinearConstraint(0, length, 0, incl_start=True, incl_end=True)]

        # There is a zero shear boundary between the supports and from the right support to the load introduction at the
        # tip, there the load is exactly p. Notice that the constraint is not active at the supports itself.
        self.V_bc = [LinearConstraint(0, length / 4, 0, incl_start=False, incl_end=False),
                     LinearConstraint(length / 4, length, 0, incl_start=False, incl_end=False),
                     PointConstraint(length, p_tip)]

        # The start and end of the beams are at the beginning and end.
        self.end = [PointConstraint(0, True),
                    PointConstraint(length, True)]

        # Call the parent class to split the problem into subdomains.
        self.split_subdomains(domain_length, num_domains)

    def exact(self, x, material):
        r"""
        The exact solution to this problem.

        Parameters
        ----------
        x : array
            The location for which the exact solution needs to be found.
        material : Constitutive
            The constitutive equation of the material considered.

        Returns
        -------
        x : array
            The coordinates where the displacement was computed :math:`x`, might differ a bit from the input.
        u : array
            The beam displacement :math:`u(x)`.
        M : array
            The internal moment of the exact solution.
        V : array
            The internal shear of the exact solution.
        """
        length = x[-1]
        load = self.V_bc[-1].magnitude
        dx = x[1] - x[0]
        exact = test.Example(length, dx, load, material)
        exact_moment = exact.moment(material)
        exact_shear = exact.shear(material)
        return exact.x, exact.u, exact_moment, exact_shear


class ExampleDisplacementControlled(Problem):
    r"""
    An example problem that is displacement controlled and static undetermined. The beam with a simple support at
    :math:`x=0` and :math:`x=L/4` while the tip (:math:`x=L`) is displaced by :math:`u`.

    Parameters
    ----------
    length : float
        Total length of the beam.
    u_tip : float
        Tip displacement.
    domain_length : float
        Length of the subdomains.
    num_domains : int
        Number of subdomains.
    continuity : int, optional
        The continuity in the overlapping region transitioning from one domain to the other (-1, 0 or a larger integer).

    Attributes
    ----------
    domain : tuple
        The start (`float`) and end (`float`) of the domain in problem coordinates.
    u_bc : list
        A list of the different boundaries where the displacement :math:`u` is constraint.
    M_bc : list
        A list with the external moment constraints.
    V_bc : list
        A list with the external shear constraints.
    end : list
        A list defining the beam ends, at each of these ends the internal static state must at least satisfy the
        prescribed loads.
    continuity : int
        Specification of the continuity at the start and end of the subdomain.
    """

    def __init__(self, length, u_tip, domain_length, num_domains, continuity=-1):
        r"""
        A beam simply supported at x=0 and x=L/4 subjected to a tip displacement.
        """
        super().__init__()

        # Initialize the domain and subdomains and continuity.
        self.domain = (0, length)
        self.continuity = continuity

        # The kinematic displacement constraints are the two simple supports at x=0 and x=L/4 and at the displaced tip.
        self.u_bc = [PointConstraint(0, 0),
                     PointConstraint(length / 4, 0),
                     PointConstraint(length, u_tip)]

        # There is no moment applied anywhere on the beam.
        self.M_bc = [LinearConstraint(0, length, 0, incl_start=True, incl_end=True)]

        # There is a zero shear boundary between the supports and from the right support to the tip. At the tip itself
        # There is a displacement constraint and the load is unknown.
        self.V_bc = [LinearConstraint(0, length / 4, 0, incl_start=False, incl_end=False),
                     LinearConstraint(length / 4, length, 0, incl_start=False, incl_end=False)]

        # The start and end of the beams are at the beginning and end.
        self.end = [PointConstraint(0, True),
                    PointConstraint(length, True)]

        # Call the parent class to split the problem into subdomains.
        self.split_subdomains(domain_length, num_domains)

    def exact(self, x, material):
        r"""
        The exact solution to this problem.

        Instead of solving the real 4ch order differential equation, the statically determinate equivalent is evaluated
        and then the required tip load to obtain the given tip displacement is found by a minimization approach.

        Parameters
        ----------
        x : array
            The location for which the exact solution needs to be found.
        material : Constitutive
            The constitutive equation of the material considered.

        Returns
        -------
        x : array
            The coordinates where the displacement was computed :math:`x`, might differ a bit from the input.
        u : array
            The beam displacement :math:`u(x)`.
        M : array
            The internal moment of the exact solution.
        V : array
            The internal shear of the exact solution.
        """
        # Local optimization function.
        def tip_error(u_tip, length, dx, material, load):
            """Local function calculates the error in tip displacement for a given load."""
            exact_load_controlled = test.Example(length, dx, load, material)
            return (exact_load_controlled.u[-1] - u_tip) ** 2

        # Obtain a summary of the loading situation.
        length = x[-1]
        u_tip = self.u_bc[-1].magnitude  # Query the required tip displacement.
        dx = x[1] - x[0]

        # Prefill these known quantities in the problem formulation.
        error = partial(tip_error, u_tip, length, dx, material)

        # Perform minimization of the scalar function.
        result = minimize_scalar(error, method='brent', tol=1e-11)
        load = result.x

        exact = test.Example(length, dx, load, material)
        exact_moment = exact.moment(material)
        exact_shear = exact.shear(material)
        return exact.x, exact.u, exact_moment, exact_shear


class CantileverLoadControlled(Problem):
    r"""
    A cantilever beam problem subjected to a tip load.

    Parameters
    ----------
    length : float
        Total length of the beam.
    load_tip : float
        Magnitude of the load applied.
    domain_length : float
        Length of the subdomains.
    num_domains : int
        Number of subdomains.
    continuity : int, optional
        The continuity in the overlapping region transitioning from one domain to the other (-1, 0 or a larger integer).

    Attributes
    ----------
    domain : tuple
        The start (`float`) and end (`float`) of the domain in problem coordinates.
    u_bc : list
        A list of the different boundaries where the displacement :math:`u` is constraint.
    M_bc : list
        A list with the external moment constraints.
    V_bc : list
        A list with the external shear constraints.
    end : list
        A list defining the beam ends, at each of these ends the internal static state must at least satisfy the
        prescribed loads.
    continuity : int
        Specification of the continuity at the start and end of the subdomain.
    """
    def __init__(self, length, load_tip, domain_length, num_domains, continuity=-1):
        r"""
        A cantilever beam subjected to a load of p at the tip.
        """
        super().__init__()

        # Initialize the domain and subdomains and continuity.
        self.domain = (0, length)
        self.continuity = continuity

        # The kinematic displacement constraints is a displacement and slope constraint at x=0.
        self.u_bc = [PointConstraint(0, 0)]
        self.du_bc = [PointConstraint(0, 0)]

        # There is no moment applied anywhere on the beam.
        self.M_bc = [LinearConstraint(0, length, 0, incl_start=False, incl_end=True)]

        # There is a zero shear boundary between the wall and the load introduction.
        self.V_bc = [LinearConstraint(0, length, 0, incl_start=False, incl_end=False),
                     PointConstraint(length, load_tip)]

        # The start and end of the beams are at the beginning and end.
        self.end = [PointConstraint(0, True),
                    PointConstraint(length, True)]

        # Call the parent class to split the problem into subdomains.
        self.split_subdomains(domain_length, num_domains)

    def exact(self, x, material):
        r"""
        The exact solution to this problem.

        Parameters
        ----------
        x : array
            The location for which the exact solution needs to be found.
        material : Constitutive
            The constitutive equation of the material considered.

        Returns
        -------
        x : array
            The coordinates where the displacement was computed :math:`x`, might differ a bit from the input.
        u : array
            The beam displacement :math:`u(x)`.
        M : array
            The internal moment of the exact solution.
        V : array
            The internal shear of the exact solution.
        """
        length = x[-1]
        load = self.V_bc[-1].magnitude
        dx = x[1] - x[0]
        exact = test.Cantilever(length, dx, load, material)
        exact_moment = exact.moment(material)
        exact_shear = exact.shear(material)
        return exact.x, exact.u, exact_moment, exact_shear


class CantileverDisplacementControlled(Problem):
    r"""
    A cantilever beam problem subjected to a tip load.

    Parameters
    ----------
    length : float
        Total length of the beam.
    load_tip : float
        Magnitude of the load applied.
    domain_length : float
        Length of the subdomains.
    num_domains : int
        Number of subdomains.
    continuity : int, optional
        The continuity in the overlapping region transitioning from one domain to the other (-1, 0 or a larger integer).

    Attributes
    ----------
    domain : tuple
        The start (`float`) and end (`float`) of the domain in problem coordinates.
    u_bc : list
        A list of the different boundaries where the displacement :math:`u` is constraint.
    M_bc : list
        A list with the external moment constraints.
    V_bc : list
        A list with the external shear constraints.
    end : list
        A list defining the beam ends, at each of these ends the internal static state must at least satisfy the
        prescribed loads.
    continuity : int
        Specification of the continuity at the start and end of the subdomain.
    """
    def __init__(self, length, tip_displacement, domain_length, num_domains, continuity=-1):
        r"""
        A cantilever beam subjected to a load of p at the tip.
        """
        super().__init__()

        # Initialize the domain and subdomains and continuity.
        self.domain = (0, length)
        self.continuity = continuity

        # The kinematic displacement constraints is a displacement and slope constraint at x=0.
        self.u_bc = [PointConstraint(0, 0), PointConstraint(length, tip_displacement)]
        self.du_bc = [PointConstraint(0, 0)]

        # There is no moment applied anywhere on the beam.
        self.M_bc = [LinearConstraint(0, length, 0, incl_start=False, incl_end=True)]

        # There is a zero shear boundary between the wall and the load introduction.
        self.V_bc = [LinearConstraint(0, length, 0, incl_start=False, incl_end=True)]

        # The start and end of the beams are at the beginning and end.
        self.end = [PointConstraint(0, True),
                    PointConstraint(length, True)]

        # Call the parent class to split the problem into subdomains.
        self.split_subdomains(domain_length, num_domains)

    def exact(self, x, material):
        r"""
        The exact solution to this problem.

        Parameters
        ----------
        x : array
            The location for which the exact solution needs to be found.
        material : Constitutive
            The constitutive equation of the material considered.

        Returns
        -------
        x : array
            The coordinates where the displacement was computed :math:`x`, might differ a bit from the input.
        u : array
            The beam displacement :math:`u(x)`.
        M : array
            The internal moment of the exact solution.
        V : array
            The internal shear of the exact solution.
        """
        # Local optimization function.
        def tip_error(u_tip, length, dx, material, load):
            """Local function calculates the error in tip displacement for a given load."""
            exact_load_controlled = test.Cantilever(length, dx, load, material)
            return (exact_load_controlled.u[-1] - u_tip) ** 2

        # Obtain a summary of the loading situation.
        length = x[-1]
        u_tip = self.u_bc[-1].magnitude  # Query the required tip displacement.
        dx = x[1] - x[0]

        # Prefill these known quantities in the problem formulation.
        error = partial(tip_error, u_tip, length, dx, material)

        # Perform minimization of the scalar function.
        result = minimize_scalar(error, method='brent', tol=1e-11)
        load = result.x

        exact = test.Cantilever(length, dx, load, material)
        exact_moment = exact.moment(material)
        exact_shear = exact.shear(material)
        return exact.x, exact.u, exact_moment, exact_shear


class SimplySupported(Problem):
    r"""
    A simply supported beam problem subjected to a mid point load.

    Parameters
    ----------
    length : float
        Total length of the beam.
    load_centre : float
        Magnitude of the load applied.
    domain_length : float
        Length of the subdomains.
    num_domains : int
        Number of subdomains.
    continuity : int, optional
        The continuity in the overlapping region transitioning from one domain to the other (-1, 0 or a larger integer).

    Attributes
    ----------
    domain : tuple
        The start (`float`) and end (`float`) of the domain in problem coordinates.
    u_bc : list
        A list of the different boundaries where the displacement :math:`u` is constraint.
    M_bc : list
        A list with the external moment constraints.
    V_bc : list
        A list with the external shear constraints.
    end : list
        A list defining the beam ends, at each of these ends the internal static state must at least satisfy the
        prescribed loads.
    continuity : int
        Specification of the continuity at the start and end of the subdomain.
    """
    def __init__(self, length, load_centre, domain_length, num_domains, continuity=-1):
        r"""
        A cantilever beam subjected to a load of p at the tip.
        """
        super().__init__()

        # Initialize the domain and subdomains and continuity.
        self.domain = (0, length)
        self.continuity = continuity

        # The kinematic displacement constraints is a displacement and slope constraint at x=0.
        self.u_bc = [PointConstraint(0, 0), PointConstraint(length, 0)]
        self.du_bc = []

        # There is no moment applied anywhere on the beam.
        self.M_bc = [LinearConstraint(0, length, 0, incl_start=True, incl_end=True)]

        # There is a zero shear boundary between the wall and the load introduction.
        self.V_bc = [LinearConstraint(0, length/2, 0, incl_start=False, incl_end=False),
                     PointConstraint(length/2, load_centre),
                     LinearConstraint(length/2, length, 0, incl_start=False, incl_end=False)]

        # The start and end of the beams are at the beginning and end.
        self.end = [PointConstraint(0, True),
                    PointConstraint(length, True)]

        # Call the parent class to split the problem into subdomains.
        self.split_subdomains(domain_length, num_domains)

    def exact(self, x, material):
        r"""
        The exact solution to this problem.

        Parameters
        ----------
        x : array
            The location for which the exact solution needs to be found.
        material : Constitutive
            The constitutive equation of the material considered.

        Returns
        -------
        x : array
            The coordinates where the displacement was computed :math:`x`, might differ a bit from the input.
        u : array
            The beam displacement :math:`u(x)`.
        M : array
            The internal moment of the exact solution.
        V : array
            The internal shear of the exact solution.
        """
        length = x[-1]
        load = self.V_bc[-2].magnitude
        dx = x[1] - x[0]
        exact = test.SimplySupported(length, dx, load, material)
        exact_moment = exact.moment(material)
        exact_shear = exact.shear(material)
        return exact.x, exact.u, exact_moment, exact_shear


class Clamped(Problem):
    r"""
    A beam clamped between two walls, subjected to a mid point load.

    Parameters
    ----------
    length : float
        Total length of the beam.
    load_centre : float
        Magnitude of the load applied.
    domain_length : float
        Length of the subdomains.
    num_domains : int
        Number of subdomains.
    continuity : int, optional
        The continuity in the overlapping region transitioning from one domain to the other (-1, 0 or a larger integer).

    Attributes
    ----------
    domain : tuple
        The start (`float`) and end (`float`) of the domain in problem coordinates.
    u_bc : list
        A list of the different boundaries where the displacement :math:`u` is constraint.
    M_bc : list
        A list with the external moment constraints.
    V_bc : list
        A list with the external shear constraints.
    end : list
        A list defining the beam ends, at each of these ends the internal static state must at least satisfy the
        prescribed loads.
    continuity : int
        Specification of the continuity at the start and end of the subdomain.
    """
    def __init__(self, length, load_centre, domain_length, num_domains, continuity=-1):
        r"""
        A cantilever beam subjected to a load of p at the tip.
        """
        super().__init__()

        # Initialize the domain and subdomains and continuity.
        self.domain = (0, length)
        self.continuity = continuity

        # The kinematic displacement constraints is a displacement and slope constraint at x=0.
        self.u_bc = [PointConstraint(0, 0), PointConstraint(length, 0)]
        self.du_bc = [PointConstraint(0, 0), PointConstraint(length, 0)]

        # There is no moment applied anywhere on the beam.
        self.M_bc = [LinearConstraint(0, length, 0, incl_start=False, incl_end=False)]

        # There is a zero shear boundary between the wall and the load introduction.
        self.V_bc = [LinearConstraint(0, length/2, 0, incl_start=False, incl_end=False),
                     PointConstraint(length/2, load_centre),
                     LinearConstraint(length/2, length, 0, incl_start=False, incl_end=False)]

        # The start and end of the beams are at the beginning and end.
        self.end = [PointConstraint(0, True),
                    PointConstraint(length, True)]

        # Call the parent class to split the problem into subdomains.
        self.split_subdomains(domain_length, num_domains)

    def exact(self, x, material):
        r"""
        The exact solution to this problem.

        Parameters
        ----------
        x : array
            The location for which the exact solution needs to be found.
        material : Constitutive
            The constitutive equation of the material considered.

        Returns
        -------
        x : array
            The coordinates where the displacement was computed :math:`x`, might differ a bit from the input.
        u : array
            The beam displacement :math:`u(x)`.
        M : array
            The internal moment of the exact solution.
        V : array
            The internal shear of the exact solution.
        """
        length = x[-1]
        load = self.V_bc[-2].magnitude
        dx = x[1] - x[0]
        exact = test.Clamped(length, dx, load, material)
        exact_moment = exact.moment(material)
        exact_shear = exact.shear(material)
        return exact.x, exact.u, exact_moment, exact_shear


class FourPointBending(Problem):
    r"""
    A 4 point bending beam problem subjected to a load.

    Parameters
    ----------
    length : float
        Total length of the beam.
    load : float
        Magnitude of the load applied at the two load introductions.
    ratio : float
        Fractional location (:math:`0\leq r \leq 0.5) where the loads are applied, :math:`x_1 = r*l` and
        :math:`x_2 = (1-r)*l`.
    domain_length : float
        Length of the subdomains.
    num_domains : int
        Number of subdomains.
    continuity : int, optional
        The continuity in the overlapping region transitioning from one domain to the other (-1, 0 or a larger integer).

    Attributes
    ----------
    domain : tuple
        The start (`float`) and end (`float`) of the domain in problem coordinates.
    u_bc : list
        A list of the different boundaries where the displacement :math:`u` is constraint.
    M_bc : list
        A list with the external moment constraints.
    V_bc : list
        A list with the external shear constraints.
    end : list
        A list defining the beam ends, at each of these ends the internal static state must at least satisfy the
        prescribed loads.
    continuity : int
        Specification of the continuity at the start and end of the subdomain.
    """
    def __init__(self, length, load, ratio, domain_length, num_domains, continuity=-1):
        r"""
        A cantilever beam subjected to a load of p at the tip.
        """
        super().__init__()

        # Initialize the domain and subdomains and continuity.
        self.domain = (0, length)
        self.continuity = continuity

        # Get the location of the load introductions.
        load_x1 = length * ratio
        load_x2 = length * (1 - ratio)

        # The kinematic displacement constraints is a displacement and slope constraint at x=0.
        self.u_bc = [PointConstraint(0, 0), PointConstraint(length, 0)]
        self.du_bc = []

        # There is no moment applied anywhere on the beam.
        self.M_bc = [LinearConstraint(0, length, 0, incl_start=True, incl_end=True)]

        # There is a zero shear boundary between the wall and the load introduction.
        self.V_bc = [LinearConstraint(0, load_x1, 0, incl_start=False, incl_end=False),
                     PointConstraint(load_x1, load),
                     LinearConstraint(load_x1, load_x2, 0, incl_start=False, incl_end=False),
                     PointConstraint(load_x2, load),
                     LinearConstraint(load_x2, length, 0, incl_start=False, incl_end=False)]

        # The start and end of the beams are at the beginning and end.
        self.end = [PointConstraint(0, True),
                    PointConstraint(length, True)]

        # Call the parent class to split the problem into subdomains.
        self.split_subdomains(domain_length, num_domains)

    def exact(self, x, material):
        r"""
        The exact solution to this problem.

        Parameters
        ----------
        x : array
            The location for which the exact solution needs to be found.
        material : Constitutive
            The constitutive equation of the material considered.

        Returns
        -------
        x : array
            The coordinates where the displacement was computed :math:`x`, might differ a bit from the input.
        u : array
            The beam displacement :math:`u(x)`.
        M : array
            The internal moment of the exact solution.
        V : array
            The internal shear of the exact solution.
        """
        length = x[-1]
        load = self.V_bc[1].magnitude
        ratio = self.V_bc[1].x / self.domain[1]
        dx = x[1] - x[0]
        exact = test.FourPointBending(length, dx, load, ratio, material)
        exact_moment = exact.moment(material)
        exact_shear = exact.shear(material)
        return exact.x, exact.u, exact_moment, exact_shear


class ClampedFourPoint(Problem):
    r"""
    A double clamped beam subjected to two symetric loads.

    Parameters
    ----------
    length : float
        Total length of the beam.
    load : float
        Magnitude of the load applied at the two load introductions.
    ratio : float
        Fractional location (:math:`0\leq r \leq 0.5) where the loads are applied, :math:`x_1 = r*l` and
        :math:`x_2 = (1-r)*l`.
    domain_length : float
        Length of the subdomains.
    num_domains : int
        Number of subdomains.
    continuity : int, optional
        The continuity in the overlapping region transitioning from one domain to the other (-1, 0 or a larger integer).

    Attributes
    ----------
    domain : tuple
        The start (`float`) and end (`float`) of the domain in problem coordinates.
    u_bc : list
        A list of the different boundaries where the displacement :math:`u` is constraint.
    M_bc : list
        A list with the external moment constraints.
    V_bc : list
        A list with the external shear constraints.
    end : list
        A list defining the beam ends, at each of these ends the internal static state must at least satisfy the
        prescribed loads.
    continuity : int
        Specification of the continuity at the start and end of the subdomain.
    """
    def __init__(self, length, load, ratio, domain_length, num_domains, continuity=-1):
        r"""
        A cantilever beam subjected to a load of p at the tip.
        """
        super().__init__()

        # Initialize the domain and subdomains and continuity.
        self.domain = (0, length)
        self.continuity = continuity

        # Get the location of the load introductions.
        load_x1 = length * ratio
        load_x2 = length * (1 - ratio)

        # The kinematic displacement constraints is a displacement and slope constraint at x=0.
        self.u_bc = [PointConstraint(0, 0), PointConstraint(length, 0)]
        self.du_bc = [PointConstraint(0, 0), PointConstraint(length, 0)]

        # There is no moment applied anywhere on the beam.
        self.M_bc = [LinearConstraint(0, length, 0, incl_start=False, incl_end=False)]

        # There is a zero shear boundary between the wall and the load introduction.
        self.V_bc = [LinearConstraint(0, load_x1, 0, incl_start=False, incl_end=False),
                     PointConstraint(load_x1, load),
                     LinearConstraint(load_x1, load_x2, 0, incl_start=False, incl_end=False),
                     PointConstraint(load_x2, load),
                     LinearConstraint(load_x2, length, 0, incl_start=False, incl_end=False)]

        # The start and end of the beams are at the beginning and end.
        self.end = [PointConstraint(0, True),
                    PointConstraint(length, True)]

        # Call the parent class to split the problem into subdomains.
        self.split_subdomains(domain_length, num_domains)

    def exact(self, x, material):
        r"""
        The exact solution to this problem.

        Parameters
        ----------
        x : array
            The location for which the exact solution needs to be found.
        material : Constitutive
            The constitutive equation of the material considered.

        Returns
        -------
        x : array
            The coordinates where the displacement was computed :math:`x`, might differ a bit from the input.
        u : array
            The beam displacement :math:`u(x)`.
        M : array
            The internal moment of the exact solution.
        V : array
            The internal shear of the exact solution.
        """
        length = x[-1]
        load = self.V_bc[1].magnitude
        ratio = self.V_bc[1].x / self.domain[1]
        dx = x[1] - x[0]
        exact = test.ClampedFourPoint(length, dx, load, ratio, material)
        exact_moment = exact.moment(material)
        exact_shear = exact.shear(material)
        return exact.x, exact.u, exact_moment, exact_shear
