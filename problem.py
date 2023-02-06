r"""
A problem formulation consists of the definition of a domain, the boundary conditions and discretization.

The classes below can be called to define a variety of problems. They result in objects that define the coordinate
system, boundary conditions and inhomogeneity function.

.. note:: Any future problem description should be a child of the :py:class:`~problem.Problem` which specifies the
    minimal information that is to be stored in a test object.

Bram van der Heijden |br|
Mechanics of Composites for Energy and Mobility |br|
KAUST |br|
2023
"""

# Import external modules.
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.interpolate import InterpolatedUnivariateSpline

# Importing my own scripts.
from constraint import PointConstraint, LinearConstraint
import test


class Domain(ABC):
    r"""
    The generic definition of a domain consists of it's geometry and boundary conditions.

    This parent class only contains the generic formulation of the geometry and boundaries.

    Attributes
    ----------
    domain : tuple
        The start (`float`) and end (`float`) of the domain in problem coordinates.
    u_bc : list
        A list with the boundary conditions at the start and end of the domain, `len(u_bc)=2`.
    rhs : list
        A list with the right-hand side of the differential equation in linear segments.
    """

    def __init__(self):
        r"""
        In general the initialization of a domain contains the boundary conditions and geometry.
        """
        self.domain = (np.NaN, np.NaN)
        self.u_bc = []
        self.rhs = []  # Only piecewise linear rhs equations are supported.

    def plot(self):
        r"""
        Plotting the domain and the conditions specified.

        .. warning:: This plotting is not entirely reliable yet, the part that plots the rhs function is not
            implemented yet.

        Returns
        -------
        axis : matplotlib.Axis
            The axis in which the problem is drawn.
        """
        # Create figure and axis
        plt.figure(figsize=(10, 5))
        axis = plt.gca()

        # Plot the domain.
        plt.plot([self.domain[0], self.domain[1]], [0, 0], c='k', clip_on=False)

        # Plot the boundary conditions.
        for boundary_value in self.u_bc:
            boundary_value.plot(axis, "bvp", 'C1')

        for forcing_term in self.rhs:
            forcing_term.plot(axis, "", 'C2')

        return axis


class SubDomain(Domain):
    r"""
    The definition of a subdomain.

    This parent class only contains the generic formulation of the geometry, boundary conditions and rhs equation.
    This extends the class with methods to verify the admissibility of the proposed solution.

    Parameters
    ----------
    domain : tuple
        The start (`float`) and end (`float`) of the domain in problem coordinates.
    u_bc : list
        A list with the boundary conditions at the start and end of the domain, `len(u_bc)=2`.
    rhs : list
        A list with the right-hand side of the differential equation in linear segments.

    Attributes
    ----------
    domain : tuple
        The start (`float`) and end (`float`) of the domain in problem coordinates.
    u_bc : list
        A list with the boundary conditions at the start and end of the domain, `len(u_bc)=2`.
    rhs : list
        A list with the right-hand side of the differential equation in linear segments.
    """

    def __init__(self, domain, u_bc, rhs):
        r"""
        Initializing a SubDomain consists of setting its attributes.
        """
        super().__init__()
        self.domain = domain
        self.u_bc = u_bc
        self.rhs = rhs

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
            patch is too small for the domain.
        """
        # Verify whether the patch is large enough to fill the domain entirely.
        if self.domain[1]-self.domain[0] > patch.x.max()-patch.x.min():
            return []

        # Find the possible translations, anything in the range t is possible.
        translations = [(self.domain[1]-patch.x.max(), self.domain[0]-patch.x.min())]
        return translations

    def admissible_rhs(self, patch, translations):
        r"""
        Calculate for what translations :math:`t` the patch can satisfy the right hand side equations.

        .. math::
            g_p(\xi) = g(x) \qquad \forall x \in \Omega_d \\
            \text{where:} \xi = x - t

        Parameters
        ----------
        patch : Patch
            The patch that is considered for the domain
        translations : list
            A list with a tuple for the max and min translations that satisfy the constraint. This list is empty if the
            patch is too small for the domain.

        Returns
        -------
        list
            A list with a tuple for the max and min translations that satisfy the constraint. This list is empty if the
            patch is too small for the domain.
        """
        if len(self.rhs) == 0:
            raise ValueError("The right hand side must be defined on the entire problem and subdomain.")

        for constraint in self.rhs:
            i = 0  # Start testing the first constraint range.
            while i < len(translations):
                translation = translations[i]  # Get translation range.
                translation = constraint.satisfy_value_free_translation(patch.x, patch.rhs, translation)
                translations[i:i + 1] = translation
                i += len(translation)  # Depending on how many sub-ranges we add we increase the counter.
        return translations

    def admissible_bc(self, patch, translations):
        r"""
        Calculate for what translations :math:`t` the patch can satisfy the Dirichlet boundary conditions.

        .. math::
            u_p(\xi) = u \qquad \forall x \in \Gamma_d \\
            \text{where:} \xi = x - t

        Parameters
        ----------
        patch : Patch
            The patch that is considered for the domain
        translations : list
            A list with a tuple for the max and min translations that satisfy the constraint. This list is empty if the
            patch is too small for the domain.

        Returns
        -------
        list
            A list with a tuple for the max and min translations that satisfy the constraint. This list is empty if the
            patch is too small for the domain.
        """
        for constraint in self.u_bc:
            i = 0  # Start testing the first constraint range.
            while i < len(translations):
                translation = translations[i]  # Get translation range.
                translation = constraint.satisfy_value_free_translation(patch.x, patch.u, translation)
                translations[i:i + 1] = translation
                i += len(translation)  # Depending on how many sub-ranges we add we increase the counter.
        return translations

    def admissible(self, patch):
        r"""
        Determine the admissibility of a patch for this domain.

        This function checks whether the patch can satisfy the right hand side and boundary condition if any.

        .. math::
            g_d(x) = g_p(\xi + t) \quad \forall x \in \Omega_d
            u(x) = \bar{u} \quad \forall x \in \Omega_d \cap \Gamma

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
        translations = self.admissible_rhs(patch, translations)
        # translations = self.admissible_bc(patch, translations)  # Primal freedom satisfies Dirichlet bc.
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
    u_bc : list
        A list with the boundary conditions at the start and end of the domain, `len(u_bc)=2`.
    rhs : list
        A list with the right-hand side of the differential equation in linear segments.
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

            # Create the constraint lists for the subdomains.
            u_bc = []
            for constraint in self.u_bc:
                cropped_constraint = constraint.crop(domain[0], domain[1])
                if cropped_constraint is not None:
                    u_bc.append(cropped_constraint)

            rhs = []
            for constraint in self.rhs:
                cropped_constraint = constraint.crop(domain[0], domain[1])
                if cropped_constraint is not None:
                    rhs.append(cropped_constraint)

            self.subdomains.append(SubDomain(domain, u_bc, rhs))

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

                # Patch combination can be considered if at least one admissible translation exists.
                if len(translations) != 0:
                    patch_list.append(p)

            # Add the admissible patches for this domain to the admissible_domain_patches list.
            admissible_domain_patches.append(patch_list)
            domain_patch_translations.append(translations_list)

        return admissible_domain_patches, domain_patch_translations

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


class Hat(Problem):
    r"""
    A domain with Dirichlet boundaries of zero, and a right hand side of the equation that is zero except for the
    region of a width h on the middle of the domain.

    .. math::
        rhs = \begin{cases}
                0 & 0 \leq x < \frac{L-h}/2 \\
                1 & \frac{L-h}/2 \leq x \leq \frac{L-2}/2 \\
                0 & \frac{L-2}/2 < x \leq L
            \end{cases}

    Parameters
    ----------
    length : float
        Total length of the beam.
    h : float
        Length over which the right hand side is nonzero.
    domain_length : float
        Length of the subdomains.
    num_domains : int
        Number of subdomains.

    Attributes
    ----------
    domain : tuple
        The start (`float`) and end (`float`) of the domain in problem coordinates.
    u_bc : list
        A list with the boundary conditions at the start and end of the domain, `len(u_bc)=2`.
    rhs : list
        A list with the right-hand side of the differential equation in linear segments.
    """

    def __init__(self, length, h, domain_length, num_domains):
        r"""
        A domain with Dirichlet boundaries of zero, and a right hand side of the equation that is zero except for the
        region of a width h on the middle of the domain.

        .. math::
            rhs = \begin{cases}
                    0 & 0 \leq x < \frac{L-h}/2 \\
                    1 & \frac{L-h}/2 \leq x \leq \frac{L-2}/2 \\
                    0 & \frac{L-2}/2 < x \leq L
                \end{cases}
        """
        super().__init__()

        # Store initialize properties.
        self._length = length
        self._h = h

        # Initialize the domain and subdomains and continuity.
        self.domain = (0, length)

        # The kinematic displacement constraints are the two simple supports at x=0 and x=L/4
        self.u_bc = [PointConstraint(0, 0),
                     PointConstraint(length, 0)]

        self.rhs = [LinearConstraint(0, (length - h)/2, 0, incl_start=True, incl_end=False),
                    LinearConstraint((length - h)/2, (length + h)/2, 1, incl_start=True, incl_end=True),
                    LinearConstraint((length + h)/2, length, 0, incl_start=False, incl_end=True)]

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

        def rhs(x):
            # Initialize f(x) to be zero.
            gx = np.zeros_like(x)

            # Find where the load is applied.
            index = np.where(((self._length - self._h) / 2 <= x) & (x <= (self._length + self._h) / 2))

            # Set the value to the load at these values.
            gx[index] = 1

            return gx

        exact = test.Laplace_Dirichlet_Dirichlet(self._length, x[1] - x[0], 0, 0, rhs, material)

        return exact.x, exact.u, exact.rhs


if __name__ == '__main__':
    # Import is required.
    from constitutive import LinearMaterial

    # Create the actual problem that we want to solve.
    L = 1.  # Length of the domain.
    h = 0.1  # Length of the loaded section.
    material = LinearMaterial(1)  # Linear unity material, to create traditional Poisson problem.

    # Create the problem.
    problem = Hat(L, h, 0.4, 3)

    # Plot the problem.
    problem.plot()

    # Obtain the exact solution.
    x = np.linspace(0, L, 200)
    x_exact, u_exact, rhs_exact = problem.exact(x, material)

    # Plot the solution.
    fig, axis = plt.subplots(2, 1, sharex='col', figsize=(10, 6),
                             gridspec_kw={'height_ratios': [3, 2], 'hspace': 0})
    ax_u = axis[0]  # Primal field
    ax_g = axis[1]  # Applied rhs, the inhomogeneity function.

    # Plot the fields.
    lines = []
    lines += ax_u.plot(x_exact, u_exact, c='C0', label="$u(x)$", )
    lines += ax_g.plot(x_exact, rhs_exact, c='C3', label="Applied rhs $rhs(x)$")

    # Add the labels to axis.
    ax_u.set_ylabel(r"Field $u$")
    ax_g.set_ylabel(r"Right hand side $rhs$")
    ax_g.set_xlabel(r"Location $x$")

    # Create Legend.
    labels = [line.get_label() for line in lines]
    ax_u.legend(lines, labels, frameon=False)
