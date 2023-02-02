r"""
The solutions to simple example problems will be used as in-silico 'test setups'.

The classes below can be called upon to obtain the solution of various poisson problems.

.. note:: Any future class should be a child of the :py:class:`~test.Test` which specifies the minimal information that
    is to be stored in a test object.

Bram van der Heijden |br|
Mechanics of Composites for Energy and Mobility |br|
KAUST |br|
2023
"""

# Import external modules.
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import InterpolatedUnivariateSpline

# Import own modules.
from helperfunctions import _m
import copy


class Test(ABC):
    r"""
    The definition of a beam test setup and the information that will be obtained from it.

    This parent class only contains the specification of this type of class and the general solver. It specifies how
    information from a standard test is stored. Only measurement results that can be from classical test setups will
    be included. That is the geometry, deformation and external loading. These properties are know at discrete
    coordinates, as a point cloud. For simplicity the material is assumed to be uniform, hence it is not an attribute
    of the test.

    .. note:: At the end of a beam the internal statics are exactly known. These statics equal the prescribed loads or
        measured reaction forces. Hence they are special points and they will be used as a constraint as well.

    Attributes
    ----------
    x : array
        Local coordinates.
    u : array
        Displacement at these local coordinates.

        Whether the coordinate intersects with the end of the beam, if so `True`, `False` Otherwise.
    """

    def __init__(self):
        r"""
        In general the initialization of such a class contains the boundary conditions, and geometry. And in our case
        it should also include the sample location where our DIC setup will be able to measure the displacement field.
        """
        self.x = np.array([])
        self.u = np.array([])
        self.rhs = np.array([])

    def __eq__(self, other):
        r"""
        Compare of the content in two two patch, `self == other`

        Parameters
        ----------
        other : Patch
            The patch object that is compared to `self`.

        Returns
        -------
        bool
            `True` if the attributes of other contains the same values as `self`, `False` otherwise.
        """
        # Verify that the other object is also of the patch type.
        if not isinstance(other, Test):
            return TypeError("An Patch object cannot be compared to objects of any other class.")

        # If other is also a Patch then they are equal ones the attributes equal.
        all_tests = (np.all(self.x == other.x)
                     and np.all(self.u == other.u)
                     and np.all(self.rhs == other.rhs))
        return all_tests

    def __copy__(self):
        r"""
        Create a shallow copy of a patch.

        Creates a new patch object (with a different location in memory) but the attributes still point the same
        location as the attributes of the old patch object.

        .. warning:: Because the attributes of `self` and the copy point to the same location altering the content of an
        attribute of `self` will affect the content of the attribute in the copy and vice versa.

        Returns
        -------
        Patch
            A shallow copy of a patch.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo=None):
        r"""
        Create a deep copy of a patch.

        Creates a new patch object (with a different location in memory) in a way that all the attributes get copied and
        point to their new memory location as well. This is more expensive then the :py:fun:`~patches.Patch.__copy__`
        but avoids issues with unexpected behaviour when changing the content of an attribute.

        Returns
        -------
        Patch
            A shallow copy of a patch.
        memo : dict, optional
            A dictionary containing all objects that were copied already.
        """
        if memo is None:
            memo = dict()

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def copy(self):
        r"""
        Return a deep copy of the Patch object.

        Returns
        -------
        Patch
            The copy made share no memory with the old object.
        """
        return self.__deepcopy__()

    def potential(self, material, x=None):
        r"""
        Obtain the potential at location `x`.

        The potential is the field after calculating it through the constitutive equation.

        .. math::
           \psi  = f(u(x)))

        .. note:: This function is not data-driven as it requites the constitutive equation.

        Parameters
        ----------
        material : Constitutive
            Constitutive equation :math:`f(u)`.
        x : array, optional
            If provided, the locations in local coordinates where the internal moment is calculated, will default to
            `self.x` if left empty.

        Returns
        -------
        array
            Internal moment at every location `x`.
        """
        if x is None:
            x = self.x

        # Calculate the curvature of the patch, and interpolate to locations x.
        u_interpolated = InterpolatedUnivariateSpline(x, self.u, k=4)
        potential = material.field_to_potential(u_interpolated)

        return potential

    def mirror(self):
        """
        Creates a mirrored version of the test.

        This is an admissible coordinate transformation and should thus be considered. The object that is returned is a
        copy of `self`, hence `self` remains unchanged.

        .. note:: This type of coordinate transformation will always be discrete, it is either a mirrored version of the
            test or not. Hence this should not be moved into the coordinate transformation optimization.

        Returns
        -------
        Test
            The mirrored copy of the test.
        """
        mirrored_test = self.__deepcopy__()
        mirrored_test.u = mirrored_test.u[::-1]
        mirrored_test.rhs = mirrored_test.rhs[::-1]
        return mirrored_test

    def plot(self, axis=None):
        r"""
        Plot the deformation and internal static state of the test setup.

        Parameters
        ----------
        axis : list, optional
            A list with two axis, one for the displacement and one for the moment plots. If not provided a new figure
            will be created with the required axis.

        Returns
        -------
        ax_u : matplotlib.Axis
            The plot axis regarding the field :math:`u`.
        ax_g : matplotlib.Axis
            The axis of the applied inhomogeneity function :math:`rhs(x)`.
        """
        if isinstance(axis, list):
            ax_u = axis[0]
            ax_g = axis[1]
        else:
            fig, axis = plt.subplots(2, 1, sharex='col', figsize=(10, 6),
                                     gridspec_kw={'height_ratios': [3, 2], 'hspace': 0})
            ax_u = axis[0]  # Primal field
            ax_g = axis[1]  # Applied rhs, the inhomogeneity function.

        # Plot the fields.
        lines = []
        lines += ax_u.plot(self.x, self.u, c='C0', label="Primal Field $u(x)$", )
        lines += ax_g.plot(self.x, self.rhs, c='C3', label="Right hand side $g(x)$")

        # Annotate the loading conditions.
        magnitude = 0.25 * (np.max(self.u) - np.min(self.u))

        # Set max and min locations to the axis.
        ax_u.set_xlim(self.x[0] - 0.02 * self.x[-1], 1.02 * self.x[-1])
        ax_u.set_ylim(np.min(self.u) - magnitude, np.max(self.u) + magnitude)

        # Add the labels to axis.
        ax_u.set_ylabel(_m(r"$u(x)$"))
        ax_g.set_ylabel(_m(r"$g(x)$"))
        ax_g.set_xlabel(_m(r"Location $x$"))

        # Create Legend.
        labels = [line.get_label() for line in lines]
        ax_u.legend(lines, labels, frameon=False)
        return ax_u, ax_g


class Laplace_Dirichlet_Dirichlet(Test):
    r"""
    A Laplace problem with purly Dirichlet boundary conditions.

    .. math::
        \nabla^2 \psi(x) = rhs(x) \qquad \text{for} \qquad 0 \leq x \leq L\\

        \psi(x) = f(u(x)) \\

        u(0) = u_0  \qquad \for x=0 and x=L


    Approximation is made with finite differences, `dx` should be small for accurate results.

    Parameters
    ----------
    length : float
        Total length of the beam.
    dx : float
        Step size of a finite difference approximation.
    u_start : float
        Magnitude of the field at the left Dirichlet boundary condition :math:`u_0`.
    u_end : float
        Magnitude of the field at the right Dirichlet boundary condition :math:`u_L`.
    rhs : callable
        The right-hand side function of the problem :math:`rhs(x)`.
    material : Constitutive
        A constitutive equations specifies the material.
    """

    def __init__(self, length, dx, u_start, u_end, rhs, material):
        r"""
        A cantilever beam is defined by its length, load and material.
        """
        super().__init__()

        # Extract information of the discretization.
        dof = int(length / dx) + 1
        x = np.linspace(0, length, dof, dtype=float)
        dx = x[1] - x[0]

        # Determine RHS of the problem.
        g = rhs(x)
        rhs = g.copy()[1:dof-1]

        # Convert the boundary conditions to the potential space.
        psi_start = material.field_to_potential(u_start)
        psi_end = material.field_to_potential(u_end)

        # Determine matrix of the system.
        shape = (dof-2, dof-2)
        diag = np.array([-1, 2, -1])/(dx**2)
        k = sparse.diags(diag, [-1, 0, 1], shape=shape, format='lil')

        # Boundary condition \psi(0) = psi_start
        rhs[0] += psi_start/(dx**2)

        # Boundary condition psi(L) = psi_end.
        rhs[-1] += psi_end/(dx**2)

        # Convert into csr format.
        k = k.tocsr()

        # Solve the system k psi = rsh.
        psi = spsolve(k, rhs)

        # Convert the solution back into the actual field u.
        u = np.zeros(dof)
        u[1:dof-1] = material.potential_to_field(psi)
        u[0] = u_start
        u[-1] = u_end

        # Define what locations are the end of the beam.
        end = np.full_like(x, False, dtype=bool)
        end[[0, -1]] = True

        # Collect the attributes.
        self.x = x.flatten()
        self.u = u.flatten()
        self.rhs = g.flatten()

    def plot(self, axis=None):
        r"""
        Plot the deformation and internal static state of the test setup.

        Parameters
        ----------
        axis : list, optional
            A list with two axis, one for the displacement and one for the moment plots. If not provided a new figure
            will be created with the required axis.

        Returns
        -------
        ax_u : matplotlib.Axis
            The plot axis regarding the field :math:`u`.
        ax_g : matplotlib.Axis
            The axis of the applied inhomogeneity function :math:`rhs(x)`.
        """
        # Create plot through parent.
        ax_u, ax_g = super().plot(axis)

        # Annotate the boundary conditions.
        arrowprops = dict(color='C0')
        ax_u.annotate(f"$u_0={self.u[0]:.2f}$", (self.x[0], self.u[0]), xytext=(self.x[0], 0), arrowprops=arrowprops)
        ax_u.annotate(f"$u_L={self.u[-1]:.2f}$", (self.x[-1], self.u[-1]), xytext=(self.x[-1], 0), arrowprops=arrowprops)
        return ax_u, ax_g


if __name__ == '__main__':
    # Import is required.
    from constitutive import LinearMaterial

    # Create problem definition.
    L = 2*np.pi  # Length of the domain.
    dx = L/20  # Spatial discretization.
    a = 0  # Left Dirichlet boundary condition.
    b = L * np.cos(L)  # Right Dirichlet boundary condition.
    rhs = lambda x: 2*np.sin(x) + x*np.cos(x)  # Problem right hand side.
    material = LinearMaterial(1)  # Linear unity material, to create traditional Poisson problem.

    # Formulate the problem.
    test = Laplace_Dirichlet_Dirichlet(L, dx, a, b, rhs, material)
    ax_u, ax_g = test.plot()

    # Add exact solution as reference.
    u_exact = test.x * np.cos(test.x)
    ax_u.plot(test.x, u_exact, "C2", label="Exact")

    # Create the actual problem that we want to solve.
    L = 1.  # Length of the domain.
    h = 0.1  # Length of the loaded section.
    f = 1.  # Load.
    dx = L / 100  # Spatial discretization.
    a = 0  # Left Dirichlet boundary condition.
    b = 0  # Right Dirichlet boundary condition.
    material = LinearMaterial(1)  # Linear unity material, to create traditional Poisson problem.

    def rhs(x):
        # Initialize f(x) to be zero.
        gx = np.zeros_like(x)

        # Find where the load is applied.
        index = np.where(((L-h)/2 <= x) & (x <= (L+h)/2))

        # Set the value to the load at these values.
        gx[index] = f

        return gx

    # Formulate the problem.
    test = Laplace_Dirichlet_Dirichlet(L, dx, a, b, rhs, material)
    ax_u, ax_g = test.plot()


