r"""
The solutions to several beam problems that will be used as in-silico 'test setups'.

The classes below can be called upon to obtain the solution of various beam problems. They result in objects that
contain all direct measurable quantities, that is a global coordinate system (DIC), externally applied moment and shear.
These loads are either applied on the static boundary, or measured as reaction forces at kinematic boundaries.

.. note:: Any future class should be a child of the :py:class:`~test.Test` which specifies the minimal information that
    is to be stored in a test object.

Bram Lagerweij |br|
Mechanics of Composites for Energy and Mobility |br|
KAUST |br|
2021
"""

# Import external modules.
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize_scalar
from scipy.interpolate import InterpolatedUnivariateSpline

# Import own modules.
from helperfunctions import _m
import copy


class Test(object):
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
    M : array
        Externally applied moments at the local coordinates, includes reaction moments.
    M_int : array
        Internal moments, extracted from reaction forces and loading conditions.
    V : array
        Externally applied shear at the local coordinates, includes reaction forces. The pressure will be includes as
        lumped shear forces.
    V_int : array
        Internal shear forces, extracted from reaction forces and loading conditions.
    end : array
        Whether the coordinate intersects with the end of the beam, if so `True`, `False` Otherwise.
    """

    def __init__(self):
        r"""
        In general the initialization of such a class contains the boundary conditions, and geometry. And in our case
        it should also include the sample location where our DIC setup will be able to measure the displacement field.
        """
        self.x = np.array([])
        self.u = np.array([])
        self.M = np.array([])
        self.M_int = np.array([])
        self.V = np.array([])
        self.V_int = np.array([])
        self.end = np.array([])

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
                     and np.all(self.M == other.M)
                     and np.all(self.M_int == other.M_int)
                     and np.all(self.V == other.V)
                     and np.all(self.V_int == other.V_int)
                     and np.all(self.end == other.end))
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
        Returns a deep copy of the Patch object.

        Returns
        -------
        Patch
            The copy made share no memory with the old object.
        """
        return self.__deepcopy__()

    def moment(self, material, x=None):
        r"""
        Obtain the internal moment at location `x`.

        The internal moment is obtained through a constitutive equation `EI` which converts curvature (the second
        derivative of displacement) into a moment. `EI` is not necessarily linear just it's name is taken from the
        classical assumption of a linear relation related to material stiffness and moment of inertia.

        .. math::
            m(x) = EI(x'')

        .. note:: This function is not data-driven as it requites the constitutive equation.

        Parameters
        ----------
        material : Constitutive
            Constitutive equation that obtains the moment from the second derivative of curvature.
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
        u_interpolated = InterpolatedUnivariateSpline(self.x, self.u, k=4)
        curvature = u_interpolated.derivative(2)(x)

        # Obtain the internal moment from the curvature through the constitutive equation.
        m = material.moment(curvature)
        return m

    def shear(self, material, x=None):
        r"""
        Obtain the internal shear at location `x`.

        The internal shear is obtained from the internal moment, in Euler-Bernoulli beam theory internal shear is the
        first derivative of internal moment. The moment is obtained through the constitutive equation, for more details
        see :py:meth:`~patches.Patch.moment`.

        .. math::
            v(x) = m(x)' = \frac{d}{dx} EI(x'')

        .. note:: This function is not data-driven as it requites the constitutive equation.

        Parameters
        ----------
        material : Constitutive
            Constitutive equation that obtains the moment from the second derivative of curvature.
        x : array, optional
            If provided, the locations in local coordinates where the internal shear is calculated, will default to
            `self.x` if left empty.

        Returns
        -------
        array
            Internal moment at every location `x`.
        """
        if x is None:
            x = self.x

        # Calculate the internal moment for all internal locations self.x.
        m = self.moment(material)

        # Compute the derivative to obtain the internal shear.
        m = InterpolatedUnivariateSpline(self.x, m, k=2)
        v = m.derivative(1)
        return v(x)

    def rotate(self):
        """
        Creates a rotated version of the test.

        This is an admissible coordinate transformation, which in the case of beam problems can only be 180 degrees.
        Hence this is hardcoded. The object that is returned is a copy of `self`, hence `self` remains unchanged.

        .. note:: This type of coordinate transformation would be more free and less discrete for problems other then
            beam problems, in those cases this rotate_patches should be integrated as a coordinate transformation that
            has to be determined by the rigid body motion and coordinate transformation solver in
            :py:class:`~configuration.Configuration`, in that case this function should be removed.

        Returns
        -------
        Test
            The rotated copy of the test.
        """
        mirrored_patch = self.__deepcopy__()
        mirrored_patch.u = -mirrored_patch.u[::-1]
        mirrored_patch.M = mirrored_patch.M[::-1]
        mirrored_patch.V = -mirrored_patch.V[::-1]
        mirrored_patch.end = mirrored_patch.end[::-1]
        return mirrored_patch

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
        mirrored_test.M = -mirrored_test.M[::-1]
        mirrored_test.V = mirrored_test.V[::-1]
        mirrored_test.end = mirrored_test.end[::-1]
        return mirrored_test

    def plot(self, axis=None, annotate=True):
        r"""
        Plot the deformation and internal static state of the test setup.

        Parameters
        ----------
        axis : list, optional
            A list with two axis, one for the displacement and one for the moment plots. If not provided a new figure
            will be created with the required axis.
        annotate : bool, optional
            This boolean is `True` if the load conditions need to be added as annotations to the plot.

        Returns
        -------
        ax_u : matplotlib.Axis
            The axis of the displacement plot.
        ax_m : matplotlib.Axis
            The axis of the internal moment curve.
        ax_v : matplotlib.Axis
            The axis of the internal shear curve, which is a `twinx` of `ax_m`.
        """
        if axis is not None:
            ax_u = axis[0]
            ax_m = axis[1]
        else:
            fig, axis = plt.subplots(2, 1, sharex='col', figsize=(10, 6),
                                     gridspec_kw={'height_ratios': [3, 2], 'hspace': 0})
            ax_u = axis[0]  # Displacement field
            ax_m = axis[1]  # Internal load axis for moment

        # Twinx v and V diagrams.
        ax_v = ax_m.twinx()

        # Plot the fields.
        lines = []
        lines += ax_u.plot(self.x, self.u, c='C0', label="Displacement")
        lines += ax_m.plot(self.x, self.M_int, c='C2', label="Internal Moment")
        lines += ax_v.plot(self.x, self.V_int, c='C3', label="Internal Shear")

        # Annotate the loading conditions.
        magnitude = 0.25 * (np.max(self.u) - np.min(self.u))
        if annotate:
            for x, u, V, M in zip(self.x, self.u, self.V, self.M):
                if np.sign(M) > 0:
                    mag = np.sign(M) * magnitude
                    ax_u.annotate(f'{M}kN', xy=(x, u - mag), xytext=(x, u + mag), c="C2", ha='center', clip_on=False,
                                  arrowprops=dict(arrowstyle="<|-", color='C2',
                                                  connectionstyle="angle3,angleA=45,angleB=-45"))
                if np.sign(M) < 0:
                    mag = np.sign(M) * magnitude
                    ax_u.annotate(f'{M}Nm', xy=(x, u - mag), xytext=(x, u + mag), c="C2", ha='center', clip_on=False,
                                  arrowprops=dict(arrowstyle="<|-", color='C2',
                                                  connectionstyle="angle3,angleA=-45,angleB=45"))
                if V != 0:
                    mag = np.sign(V) * magnitude
                    ax_u.annotate(f'{V}kN', xy=(x, u), xytext=(x, u + mag), c="C3", ha='center', clip_on=False,
                                  arrowprops=dict(arrowstyle="<|-", color='C3'))

        # Set max and min locations to the axis.
        ax_u.set_xlim(self.x[0] - 0.02 * self.x[-1], 1.02 * self.x[-1])
        ax_u.set_ylim(np.min(self.u) - magnitude, np.max(self.u) + magnitude)

        # Add the labels to axis.
        ax_u.set_ylabel(_m(r"Displacement $u$ in mm"))
        ax_m.set_ylabel(_m(r"Moment $M$ in kNmm"))
        ax_v.set_ylabel(_m(r"Shear $V$ in kN"))
        ax_m.set_xlabel(_m(r"Location $x$ in mm"))

        # Color the axis.
        ax_u.spines['left'].set_color('C0')
        ax_m.spines['left'].set_color('C2')
        ax_v.spines['left'].set_color('C2')
        ax_v.spines['right'].set_color('C3')

        # Color the axis ticks.
        ax_u.tick_params(axis='y', colors='C0')
        ax_m.tick_params(axis='y', colors='C2')
        ax_v.tick_params(axis='y', colors='C3')

        # Color the labels.
        ax_u.yaxis.label.set_color('C0')
        ax_m.yaxis.label.set_color('C2')
        ax_v.yaxis.label.set_color('C3')

        # Create Legend.
        labels = [line.get_label() for line in lines]
        ax_u.legend(lines, labels, frameon=False)
        return ax_u, ax_m, ax_v


class Cantilever(Test):
    r"""
    Deformation of the Cantilever Beam.

    Approximation is made with finite differences, `dx` should be small for accurate results.

    Parameters
    ----------
    length : float
        Total length of the beam.
    dx : float
        Step size of a finite difference approximation.
    p : float
        Magnitude of the load applied.
    material : Constitutive
        A constitutive equations specifies the material.

    Attributes
    ----------
    x : array
        Local coordinates.
    u : array
        Displacement at these local coordinates.
    M : array
        Externally applied moments at the local coordinates, includes reaction moments.
    M_int : array
        Internal moments, extracted from reaction forces and loading conditions.
    V : array
        Externally applied shear at the local coordinates, includes reaction forces. The pressure will be includes as
        lumped shear forces.
    V_int : array
        Internal shear forces, extracted from reaction forces and loading conditions.
    end : array
        Whether the coordinate intersects with the end of the beam, if so `True`, `False` Otherwise.
    """

    def __init__(self, length, dx, p, material):
        r"""
        A cantilever beam is defined by its length, load and material.
        """
        super().__init__()

        # Extract information of the discretization.
        dof = int(length / dx) + 1
        x = np.linspace(0, length, dof, dtype=float)
        dx = x[1] - x[0]

        # Determine RHS of the problem.
        M_int = -p * (x - length)  # internal moment.
        f = material.curvature(M_int)  # right hand side.

        # Determine matrix of the system.
        shape = (dof, dof)
        diag = [1 / (dx ** 2), -2 / (dx ** 2), 1 / (dx ** 2)]
        K = sparse.diags(diag, [-1, 0, 1], shape=shape, format='lil')

        # Boundary condition u(0) = 0
        K[0, 0] = 1
        K[0, 1] = 0
        f[0] = 0

        # Boundary condition u'(0) = 0 with a finite difference.
        # For this purpose we use the last row of the matrix
        # This row is not yet used
        K[-1, 0] = -1 / dx
        K[-1, 1] = 1 / dx
        K[-1, -2] = 0
        K[-1, -1] = 0
        f[-1] = 0

        # Convert into csr format.
        K = K.tocsr()

        # Solve the system K u = f.
        u = spsolve(K, f)

        # The external moment is zero except at the location where the reaction moment is introduced at the wall.
        # There it counteracts the internal moment.
        M = np.zeros_like(x)
        M[0] = -M_int[0]

        # The external traction is non-zero at the load introduction and the reaction forces at the supports.
        V = np.zeros_like(x)
        V[0] = -p
        V[-1] = p

        # Define what locations are the end of the beam.
        end = np.full_like(x, False, dtype=bool)
        end[[0, -1]] = True

        # Collect the attributes.
        self.x = x
        self.u = u
        self.M = M
        self.M_int = M_int
        self.V = V
        self.V_int = -p * np.ones_like(x)
        self.end = end

    def plot(self, axis=None, annotate=True):
        r"""
        Plot the deformation and internal static state of the test setup.

        This function adds the loading conditions specific to this test to the function :py:meth:`~test.Test.plot`
        function of the parent class.

        Parameters
        ----------
        axis : list, optional
            A list with two axis, one for the displacement and one for the moment plots. If not provided a new figure
            will be created with the required axis.
        annotate : bool, optional
            This boolean is `True` if the load conditions need to be added as annotations to the plot.


        Returns
        -------
        ax_u : matplotlib.Axis
            The axis of the displacement plot.
        ax_m : matplotlib.Axis
            The axis of the internal moment curve.
        ax_v : matplotlib.Axis
            The axis of the internal shear curve, which is a `twinx` of `ax_m`.
        """
        # Call parent plotting for the base plot.
        ax_u, ax_m, ax_v = super().plot(axis, annotate)

        # Add the load introductions as annotations to the plot.
        line_height = -0.05 * (self.u.max() - self.u.min())
        ax_u.plot([line_height, -line_height])
        return ax_u, ax_m, ax_v


class SimplySupported(Test):
    r"""
    Deformation of a Simply Supported Beam where the load is applied in the center.

    Approximation is made with finite differences, `dx` should be small for accurate results.

    Parameters
    ----------
    length : float
        Total length of the beam.
    dx : float
        Step size of a finite difference approximation.
    p : float
        Magnitude of the load applied.
    material : Constitutive
        A constitutive equations specifies the material.

    Attributes
    ----------
    x : array
        Local coordinates.
    u : array
        Displacement at these local coordinates.
    M : array
        Externally applied moments at the local coordinates, includes reaction moments.
    M_int : array
        Internal moments, extracted from reaction forces and loading conditions.
    V : array
        Externally applied shear at the local coordinates, includes reaction forces. The pressure will be includes as
        lumped shear forces.
    V_int : array
        Internal shear forces, extracted from reaction forces and loading conditions.
    end : array
        Whether the coordinate intersects with the end of the beam, if so `True`, `False` Otherwise.
    """

    def __init__(self, length, dx, p, material):
        r"""
        A cantilever beam is defined by its length, load and material.
        """
        super().__init__()

        # Extract information of the discretization.
        dof = int(length / dx) + 1
        x = np.linspace(0, length, dof)
        dx = x[1] - x[0]

        # Determine RHS of the problem.
        M_int = np.zeros_like(x, dtype=float)  # internal moment
        V_int = np.zeros_like(x, dtype=float)  # internal shear
        ind = np.where(x < length / 2)  # where x < L/2
        M_int[ind] = -p / 2 * x[ind]
        V_int[ind] = -p / 2
        ind = np.where(length / 2 <= x)  # where L/2 < x
        M_int[ind] = p / 2 * (x[ind] - length)
        V_int[ind] = p / 2
        f = material.curvature(M_int)  # right hand side.

        # Determine matrix of the system.
        shape = (dof, dof)
        diag = [1 / (dx ** 2), -2 / (dx ** 2), 1 / (dx ** 2)]
        K = sparse.diags(diag, [-1, 0, 1], shape=shape, format='lil')

        # Boundary condition u(0) = 0
        K[0, 0] = 1
        K[0, 1] = 0
        f[0] = 0

        # Boundary condition u(L) = 0.
        K[-1, -2] = 0
        K[-1, -1] = 1
        f[-1] = 0

        # Convert into csr format.
        K = K.tocsr()

        # Solve the system K u = f.
        u = spsolve(K, f)

        # The external moment is zero everywhere.
        M = np.zeros_like(x)

        # The external traction is non-zero at the load introduction and the reaction forces at the supports.
        V = np.zeros_like(x)
        V[0] = -p / 2
        V[int(dof / 2)] = p
        V[-1] = -p / 2

        # Define what locations are the end of the beam.
        end = np.full_like(x, False)
        end[[0, -1]] = True

        # Collect the attributes.
        self.x = x
        self.u = u
        self.M = M
        self.M_int = M_int
        self.V = V
        self.V_int = V_int
        self.end = end

    def plot(self, axis=None, annotate=True):
        r"""
        Plot the deformation and internal static state of the test setup.

        This function adds the loading conditions specific to this test to the function :py:meth:`~test.Test.plot`
        function of the parent class.

        Parameters
        ----------
        axis : list, optional
            A list with two axis, one for the displacement and one for the moment plots. If not provided a new figure
            will be created with the required axis.
        annotate : bool, optional
            This boolean is `True` if the load conditions need to be added as annotations to the plot.

        Returns
        -------
        ax_u : matplotlib.Axis
            The axis of the displacement plot.
        ax_m : matplotlib.Axis
            The axis of the internal moment curve.
        ax_v : matplotlib.Axis
            The axis of the internal shear curve, which is a `twinx` of `ax_m`.
        """
        # Call parent plotting for the base plot.
        ax_u, ax_m, ax_v = super().plot(axis, annotate)
        print(type(ax_u))

        # Add the supports as annotations to the plot.
        tri_height = -0.05 * (self.u.max() - self.u.min())
        tri_center = self.x[-1]
        t1 = plt.Polygon([[0, 0], [8, tri_height], [-8, tri_height]], closed=False, color='k', clip_on=False)
        t2 = plt.Polygon([[tri_center, 0], [tri_center + 8, tri_height], [tri_center - 8, tri_height]], closed=False,
                         color='k', clip_on=False)
        ax_u.add_patch(t1)
        ax_u.add_patch(t2)
        return ax_u, ax_m, ax_v


class Clamped(Test):
    r"""
    Deformation of a beam clamped between two walls where the load is applied in the center.

    Approximation is made with finite differences, `dx` should be small for accurate results.

    .. note :: The calculation of the displacement field is only accurate in case of rotationally symmetric constitutive
        models :math:`-M(\kappa) = M(\kappa)`.

    Parameters
    ----------
    length : float
        Total length of the beam.
    dx : float
        Step size of a finite difference approximation.
    p : float
        Magnitude of the load applied.
    material : Constitutive
        A constitutive equations specifies the material.

    Attributes
    ----------
    x : array
        Local coordinates.
    u : array
        Displacement at these local coordinates.
    M : array
        Externally applied moments at the local coordinates, includes reaction moments.
    M_int : array
        Internal moments, extracted from reaction forces and loading conditions.
    V : array
        Externally applied shear at the local coordinates, includes reaction forces. The pressure will be includes as
        lumped shear forces.
    V_int : array
        Internal shear forces, extracted from reaction forces and loading conditions.
    end : array
        Whether the coordinate intersects with the end of the beam, if so `True`, `False` Otherwise.
    """

    def __init__(self, length, dx, p, material):
        r"""
        A cantilever beam is defined by its length, load and material.
        """
        super().__init__()

        # Extract information of the discretization.
        dof = int(length / dx) + 1
        x = np.linspace(0, length, dof)
        dx = x[1] - x[0]

        # Determine RHS of the problem.
        M_int = np.zeros_like(x, dtype=float)  # internal moment
        V_int = np.zeros_like(x, dtype=float)  # internal shear
        ind = np.where(x < length / 2)  # where x < L/2
        M_int[ind] = -p / 2 * x[ind] + p * length / 8
        V_int[ind] = -p / 2
        ind = np.where(length / 2 <= x)  # where L/2 < x
        M_int[ind] = p / 2 * (x[ind] - length) + p * length / 8
        V_int[ind] = p / 2
        f = material.curvature(M_int)  # right hand side.

        # Determine matrix of the system.
        shape = (dof, dof)
        diag = [1 / (dx ** 2), -2 / (dx ** 2), 1 / (dx ** 2)]
        K = sparse.diags(diag, [-1, 0, 1], shape=shape, format='lil')

        # Boundary condition u(0) = 0
        K[0, 0] = 1
        K[0, 1] = 0
        f[0] = 0

        # Boundary condition u(L) = 0.
        K[-1, -2] = 0
        K[-1, -1] = 1
        f[-1] = 0

        # Convert into csr format.
        K = K.tocsr()

        # Solve the system K u = f.
        u = spsolve(K, f)

        # The external moment is zero everywhere.
        M = np.zeros_like(x)
        M[0] = -p * length / 8
        M[-1] = p * length / 8

        # The external traction is non-zero at the load introduction and the reaction forces at the supports.
        V = np.zeros_like(x)
        V[0] = -p / 2
        V[int(dof / 2)] = p
        V[-1] = -p / 2

        # Define what locations are the end of the beam.
        end = np.full_like(x, False)
        end[[0, -1]] = True

        # Collect the attributes.
        self.x = x
        self.u = u
        self.M = M
        self.M_int = M_int
        self.V = V
        self.V_int = V_int
        self.end = end

    def plot(self, axis=None, annotate=True):
        r"""
        Plot the deformation and internal static state of the test setup.

        This function adds the loading conditions specific to this test to the function :py:meth:`~test.Test.plot`
        function of the parent class.

        Parameters
        ----------
        axis : list, optional
            A list with two axis, one for the displacement and one for the moment plots. If not provided a new figure
            will be created with the required axis.
        annotate : bool, optional
            This boolean is `True` if the load conditions need to be added as annotations to the plot.

        Returns
        -------
        ax_u : matplotlib.Axis
            The axis of the displacement plot.
        ax_m : matplotlib.Axis
            The axis of the internal moment curve.
        ax_v : matplotlib.Axis
            The axis of the internal shear curve, which is a `twinx` of `ax_m`.
        """
        # Call parent plotting for the base plot.
        ax_u, ax_m, ax_v = super().plot(axis, annotate)
        print(type(ax_u))

        # Add the supports as annotations to the plot.
        line_height = -0.05 * (self.u.max() - self.u.min())
        ax_u.plot([line_height, -line_height])
        ax_u.plot([line_height, -line_height])
        return ax_u, ax_m, ax_v


class FourPointBending(Test):
    r"""
    Deformation of a beam under four point bending.

    Approximation is made with finite differences, `dx` should be small for accurate results.

    Parameters
    ----------
    length : float
        Total length of the beam.
    dx : float
        Step size of a finite difference approximation.
    p : float
        Magnitude of the load applied.
    ratio : float
        Fractional location (:math:`0\leq r \leq 0.5) where the loads are applied, :math:`x_1 = r*l` and
        :math:`x_2 = (1-r)*l`.
    material : Constitutive
        A constitutive equations specifies the material.

    Attributes
    ----------
    x : array
        Local coordinates.
    u : array
        Displacement at these local coordinates.
    M : array
        Externally applied moments at the local coordinates, includes reaction moments.
    M_int : array
        Internal moments, extracted from reaction forces and loading conditions.
    V : array
        Externally applied shear at the local coordinates, includes reaction forces. The pressure will be includes as
        lumped shear forces.
    V_int : array
        Internal shear forces, extracted from reaction forces and loading conditions.
    end : array
        Whether the coordinate intersects with the end of the beam, if so `True`, `False` Otherwise.
    """

    def __init__(self, length, dx, p, ratio, material):
        r"""
        A cantilever beam is defined by its length, load and material.
        """
        super().__init__()

        # Extract information of the discretization.
        dof = int(length / dx) + 1
        x = np.linspace(0, length, dof)
        dx = x[1] - x[0]

        # Get the location of the load introductions.
        load_x1 = length * ratio
        load_x1_loc = int(load_x1 / dx)
        load_x2 = length * (1 - ratio)
        load_x2_loc = int(load_x2 / dx)

        # Determine RHS of the problem.
        M_int = np.zeros_like(x, dtype=float)  # internal moment.
        V_int = np.zeros_like(x, dtype=float)  # internal shear.
        ind = np.where(x < load_x1)  # the first segment.
        M_int[ind] = -p * x[ind]
        V_int[ind] = -p
        ind = np.where((load_x1 <= x) & (x <= load_x2))  # the middle part with constant moment.
        M_int[ind] = - p * length * ratio
        V_int[ind] = 0
        ind = np.where(load_x2 <= x)  # the last segment.
        M_int[ind] = p * (x[ind] - length)
        V_int[ind] = p

        # Convert moments into curvature.
        f = material.curvature(M_int)  # right hand side.

        # Determine matrix of the system.
        shape = (dof, dof)
        diag = [1 / (dx ** 2), -2 / (dx ** 2), 1 / (dx ** 2)]
        K = sparse.diags(diag, [-1, 0, 1], shape=shape, format='lil')

        # Boundary condition u(0) = 0
        K[0, 0] = 1
        K[0, 1] = 0
        f[0] = 0

        # Boundary condition u(L) = 0.
        K[-1, -2] = 0
        K[-1, -1] = 1
        f[-1] = 0

        # Convert into csr format.
        K = K.tocsr()

        # Solve the system K u = f.
        u = spsolve(K, f)

        # The external moment is zero everywhere.
        M = np.zeros_like(x)

        # The external traction is non-zero at the load introduction and the reaction forces at the supports.
        V = np.zeros_like(x)
        V[0] = -p
        V[load_x1_loc] = p
        V[load_x2_loc] = p
        V[-1] = -p

        # Define what locations are the end of the beam.
        end = np.full_like(x, False)
        end[[0, -1]] = True

        # Collect the attributes.
        self.x = x
        self.u = u
        self.M = M
        self.M_int = M_int
        self.V = V
        self.V_int = V_int
        self.end = end

    def plot(self, axis=None, annotate=True):
        r"""
        Plot the deformation and internal static state of the test setup.

        This function adds the loading conditions specific to this test to the function :py:meth:`~test.Test.plot`
        function of the parent class.

        Parameters
        ----------
        axis : list, optional
            A list with two axis, one for the displacement and one for the moment plots. If not provided a new figure
            will be created with the required axis.
        annotate : bool, optional
            This boolean is `True` if the load conditions need to be added as annotations to the plot.

        Returns
        -------
        ax_u : matplotlib.Axis
            The axis of the displacement plot.
        ax_m : matplotlib.Axis
            The axis of the internal moment curve.
        ax_v : matplotlib.Axis
            The axis of the internal shear curve, which is a `twinx` of `ax_m`.
        """
        # Call parent plotting for the base plot.
        ax_u, ax_m, ax_v = super().plot(axis, annotate)

        # Add the supports as annotations to the plot.
        tri_height = -0.05 * (self.u.max() - self.u.min())
        tri_center = self.x[-1]
        t1 = plt.Polygon([[0, 0], [8, tri_height], [-8, tri_height]], closed=False, color='k', clip_on=False)
        t2 = plt.Polygon([[tri_center, 0], [tri_center + 8, tri_height], [tri_center - 8, tri_height]], closed=False,
                         color='k', clip_on=False)
        ax_u.add_patch(t1)
        ax_u.add_patch(t2)
        return ax_u, ax_m, ax_v


class ClampedFourPoint(Test):
    r"""
    Deformation of a beam clamped between two walls and loaded symmetrically at two locations.

    Approximation is made with finite differences, `dx` should be small for accurate results. The zero slope boundary
    condition is only approximated as a result the accuracy relates stronger to `dx` then most other tests do. This
    will also affect the comparison between the data-driven and the exact solution as the exact solution is based upon
    this class.

    Parameters
    ----------
    length : float
        Total length of the beam.
    dx : float
        Step size of a finite difference approximation.
    p : float
        Magnitude of the load applied.
    ratio : float
        Fractional location (:math:`0\leq r \leq 0.5) where the loads are applied, :math:`x_1 = r*l` and
        :math:`x_2 = (1-r)*l`.
    material : Constitutive
        A constitutive equations specifies the material.

    Attributes
    ----------
    x : array
        Local coordinates.
    u : array
        Displacement at these local coordinates.
    M : array
        Externally applied moments at the local coordinates, includes reaction moments.
    M_int : array
        Internal moments, extracted from reaction forces and loading conditions.
    V : array
        Externally applied shear at the local coordinates, includes reaction forces. The pressure will be includes as
        lumped shear forces.
    V_int : array
        Internal shear forces, extracted from reaction forces and loading conditions.
    end : array
        Whether the coordinate intersects with the end of the beam, if so `True`, `False` Otherwise.
    """

    def __init__(self, length, dx, p, ratio, material):
        r"""
        A clamped 4 point bending beam is defined by its length, load and material.

        A simply supported 4 point bending beam is initialized, then the only addition is that a constant moment will
        be added. The magnitude of this constant moment is found by:

        .. math :: \min_{m_c} u'(0)^2

        where :math:`m_c` represents the moment at the clamp and :math:`u'` the slope of the beam.
        """

        def du_at_clamp(moment_clamp, ss, material):
            """
            The slope at the at the clamp squared.

            Parameters
            ----------
            moment_clamp : float
                The moment at the clamp.
            ss : FourPointBending
                The simply supported equivalent.
            material : Constitutive
                The constitutive equations that specify the material.

            Returns
            -------
            du2 : float
                The square of the slope at :math:`x=0`.
            """
            # Add padding to the simply supported beam.
            x = ss.x
            moment_ss = ss.moment(material)

            # Get the curvature due to the combined moments.
            moment = moment_ss + moment_clamp
            f = material.curvature(moment)

            # Determine the system matrix.
            dof = len(x)
            shape = (dof, dof)
            diagonals = [1 / (dx ** 2), -2 / (dx ** 2), 1 / (dx ** 2)]
            K = sparse.diags(diagonals, [-1, 0, 1], shape=shape, format='lil')

            # Boundary condition u(0) = 0
            K[0, 0] = 1
            K[0, 1] = 0
            f[0] = 0

            # Boundary condition u(L) = 0.
            K[-1, -2] = 0
            K[-1, -1] = 1
            f[-1] = 0

            # Convert into csr format.
            K = K.tocsr()

            # Solve the system K u = f.
            u = spsolve(K, f)

            # With first order differences du = 0 if u[1] == u[0]
            du = u[1] - u[0]
            return du ** 2

        # Initialize the super class.
        super().__init__()

        # Create the simply supported object and extract the moment and coordinates.
        ss = FourPointBending(length, dx, p, ratio, material)

        # Make the slope at x=0 equal to zero by minimizing.
        result = minimize_scalar(du_at_clamp, method='brent', args=(ss, material), tol=1e-15)
        moment_clamp = result.x

        # Get the curvature due to the combined moments.
        moment = ss.moment(material) + moment_clamp
        f = material.curvature(moment)

        # Determine matrix of the system.
        dof = len(ss.x)
        dx = ss.x[1] - ss.x[0]
        shape = (dof, dof)
        diag = [1 / (dx ** 2), -2 / (dx ** 2), 1 / (dx ** 2)]
        K = sparse.diags(diag, [-1, 0, 1], shape=shape, format='lil')

        # Boundary condition u(0) = 0
        K[0, 0] = 1
        K[0, 1] = 0
        f[0] = 0

        # Boundary condition u(L) = 0.
        K[-1, -2] = 0
        K[-1, -1] = 1
        f[-1] = 0

        # Convert into csr format.
        K = K.tocsr()

        # Solve the system K u = f.
        u = spsolve(K, f)

        # The external moment is zero everywhere.
        M = ss.M
        M[0] = -moment_clamp
        M[-1] = moment_clamp

        # The external traction is non-zero at the load introduction and the reaction forces at the supports.
        V = ss.V

        # Define what locations are the end of the beam.
        end = np.full_like(ss.x, False)
        end[[0, -1]] = True

        # Collect the attributes.
        self.x = ss.x
        self.u = u
        self.M = M
        self.M_int = ss.M_int + moment_clamp
        self.V = V
        self.V_int = ss.V_int
        self.end = end

    def plot(self, axis=None, annotate=True):
        r"""
        Plot the deformation and internal static state of the test setup.

        This function adds the loading conditions specific to this test to the function :py:meth:`~test.Test.plot`
        function of the parent class.

        Parameters
        ----------
        axis : list, optional
            A list with two axis, one for the displacement and one for the moment plots. If not provided a new figure
            will be created with the required axis.
        annotate : bool, optional
            This boolean is `True` if the load conditions need to be added as annotations to the plot.

        Returns
        -------
        ax_u : matplotlib.Axis
            The axis of the displacement plot.
        ax_m : matplotlib.Axis
            The axis of the internal moment curve.
        ax_v : matplotlib.Axis
            The axis of the internal shear curve, which is a `twinx` of `ax_m`.
        """
        # Call parent plotting for the base plot.
        ax_u, ax_m, ax_v = super().plot(axis, annotate)
        print(type(ax_u))

        # Add the supports as annotations to the plot.
        line_height = -0.05 * (self.u.max() - self.u.min())
        ax_u.plot([line_height, -line_height])
        ax_u.plot([line_height, -line_height])
        return ax_u, ax_m, ax_v


class Example(Test):
    r"""
    Deformation of the Example problem, a tip loaded beam with a simple support at :math:`x=0` and :math:`x=L/4`.

    Approximation is made with finite differences, `dx` should be small for accurate results.

    Parameters
    ----------
    length : float
        Total length of the beam.
    dx : float
        Step size of a finite difference approximation.
    p : float
        Magnitude of the load applied.
    material : Constitutive
        A constitutive equations specifies the material.

    Attributes
    ----------
    x : array
        Local coordinates.
    u : array
        Displacement at these local coordinates.
    M : array
        Externally applied moments at the local coordinates, includes reaction moments.
    M_int : array
        Internal moments, extracted from reaction forces and loading conditions.
    V : array
        Externally applied shear at the local coordinates, includes reaction forces. The pressure will be includes as
        lumped shear forces.
    V_int : array
        Internal shear forces, extracted from reaction forces and loading conditions.
    end : array
        Whether the coordinate intersects with the end of the beam, if so `True`, `False` Otherwise.
    """

    def __init__(self, length, dx, p, material):
        r"""
        A cantilever beam is defined by its length, load and material.
        """
        super().__init__()

        # Extract information of the discretization.
        dof = int(length / dx) + 1
        x = np.linspace(0, length, dof)
        dx = x[1] - x[0]

        # Determine RHS of the problem.
        M_int = np.zeros_like(x, dtype=float)  # moment storage
        V_int = np.zeros_like(x, dtype=float)
        ind = np.where(x < length / 4)  # where x < L/4
        M_int[ind] = 3. * p * x[ind]
        V_int[ind] = 3. * p
        ind = np.where(length / 4 <= x)  # where L/4 < x
        M_int[ind] = -p * (x[ind] - length)
        V_int[ind] = -p
        f = material.curvature(M_int)  # right hand side.

        # Determine matrix of the system.
        shape = (dof, dof)
        diag = [1 / (dx ** 2), -2 / (dx ** 2), 1 / (dx ** 2)]
        K = sparse.diags(diag, [-1, 0, 1], shape=shape, format='lil')

        # Boundary condition u(0) = 0
        K[0, 0] = 1
        K[0, 1] = 0
        f[0] = 0

        # Boundary condition u(L/4) = 0.
        index = int(dof / 4)
        K[-1, index] = 1
        f[-1] = 0
        K[-1, -2] = 0
        K[-1, -1] = 0
        f[-1] = 0

        # Convert into csr format.
        K = K.tocsr()

        # Solve the system K u = f.
        u = spsolve(K, f)

        # The external moment is zero everywhere.
        M = np.zeros_like(x)

        # The external traction is non-zero at the load introduction and the reaction forces at the supports.
        V = np.zeros_like(x)
        V[-1] = p
        V[0] = 3 * p
        V[int(dof / 4)] = -4 * p

        # Define what locations are the end of the beam.
        end = np.full_like(x, False)
        end[[0, -1]] = True

        # Collect the attributes.
        self.x = x
        self.u = u
        self.M = M
        self.M_int = M_int
        self.V = V
        self.V_int = V_int
        self.end = end

    def plot(self, axis=None, annotate=True):
        r"""
        Plot the deformation and internal static state of the test setup.

        This function adds the loading conditions specific to this test to the function :py:meth:`~test.Test.plot`
        function of the parent class.

        Parameters
        ----------
        axis : list, optional
            A list with two axis, one for the displacement and one for the moment plots. If not provided a new figure
            will be created with the required axis.
        annotate : bool, optional
            This boolean is `True` if the load conditions need to be added as annotations to the plot.

        Returns
        -------
        ax_u : matplotlib.Axis
            The axis of the displacement plot.
        ax_m : matplotlib.Axis
            The axis of the internal moment curve.
        ax_v : matplotlib.Axis
            The axis of the internal shear curve, which is a `twinx` of `ax_m`.
        """
        # Call parent plotting for the base plot.
        ax_u, ax_m, ax_v = super().plot(axis, annotate)

        # Add the load introductions as annotations to the plot.
        tri_height = -0.05 * (self.u.max() - self.u.min())
        tri_center = self.x[-1] / 4
        t1 = plt.Polygon([[0, 0], [8, tri_height], [-8, tri_height]], closed=False, color='k', clip_on=False)
        t2 = plt.Polygon([[tri_center, 0], [tri_center + 8, tri_height], [tri_center - 8, tri_height]], closed=False,
                         color='k', clip_on=False)
        ax_u.add_patch(t1)
        ax_u.add_patch(t2)
        return ax_u, ax_m, ax_v


# Test whether the test examples contain the correct results.
if __name__ == '__main__':
    # Import the material package
    from constitutive import LinearMaterial

    # Settings for the test cases.
    length = 400.
    dx = 1.
    p = 1.
    ratio = 0.25
    material = LinearMaterial(1e6)

    # Create the test cases.
    cantilever = Cantilever(length, dx, p, material)
    simple = SimplySupported(length, dx, p, material)
    clamped = Clamped(length, dx, p, material)
    fourpoint_simple = FourPointBending(length, dx, p, ratio, material)
    fourpoint_clamped = ClampedFourPoint(length, dx, p, ratio, material)
    example = Example(length, dx, p, material)

    # Verify the solutions.
    # cantilever.plot()
    # simple.plot()
    clamped.plot()
    # fourpoint_simple.plot()
    # fourpoint_clamped.plot()
    # example.plot()
