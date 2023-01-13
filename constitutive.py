r"""
A collection of arbitrary constitutive models for beam equations.

The data-driven method should be independent of the constitutive equation considered. It is however included for two
reasons. Firstly the database is created in-silico which requires the use of a traditional method to calculate the
deflection, these methods require constitutive equations. Secondly the constitutive equation can be used to evaluated
the performance of the data-driven method by comparing it to other traditional norms such as the total potential energy.
This does also require a constitutive equation.

.. note:: Any future class should be a child of the :py:class:`~constitutive.Constitutive` which specifies the minimal
    information that is to be available in a constitutive class. It will provide errors if any of the required functions
    are not implemented by the child class.

Bram Lagerweij |br|
Mechanics of Composites for Energy and Mobility |br|
KAUST |br|
2021
"""

# Importing required modules.
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy.special import expit, logit

# Import own modules
from helperfunctions import _m


class Constitutive(ABC):
    r"""
    The parent class for constitutive equations, all constitutive equations should contain the following functions.

    .. warning:: This class is a parent, only used to specify the definition of any constitutive class. It is not
        implement and should only be used for inheritance.
    """
    def __init__(self):
        r"""
        Initializing the constitutive object.

        A constitutive object generally contains some constants in its underlying function, these are specified in
        the initialization part.
        """
        pass

    @abstractmethod
    def moment(self, uxx):
        r"""
        The material model converts kinematic derivative :math:`\kappa` into internal static :math:`M`.

        Parameters
        ----------
        uxx : array
            Curvatures to be converted into moments.

        Returns
        -------
        array
            The moments that related to the curvature though the constitutive equation..
        """
        pass

    @abstractmethod
    def curvature(self, m):
        r"""
        The material model converts kinematic derivative :math:`\kappa` into internal static :math:`M`.

        Parameters
        ----------
        m : array
            Moments to be converted into curvatures.

        Returns
        -------
        array
            The curvature related to the moment through the constitutive equation.
        """
        pass

    def plot(self, uxx):
        r"""
        Plot the constitutive equation for all given curvature values.

        Parameters
        ----------
        uxx : array
            Curvatures for which the constitutive equation is plotted.
        """
        m = self.moment(uxx)

        plt.plot(uxx, m)
        plt.xlabel(_m(r"Curvature $u''$"))
        plt.ylabel(_m(r"Moment $M$"))


class LinearMaterial(Constitutive):
    r"""
    A simple linear constitutive equation for beams.

    This linear constitutive equation represents is traditionally used in beam problems, it assumes that
    bending-stiffness is constant that will be called :math:`EI`.

    .. math::
            m = EI \kappa\\
            \kappa = \frac{1}{EI} m

    Parameters
    ----------
    EI : float
        The bending stiffness as a constant.

    Attributes
    ----------
    EI : float
        Constant bending stiffness.
    """
    def __init__(self, EI):
        r"""
        This simple linear constitutive equation depends on the bending stiffness :math:`EI`.
        """
        super().__init__()
        self.EI = EI

    def moment(self, uxx):
        r"""
        Converts curvature into internal moment.

        .. math::
            m = EI \kappa

        Parameters
        ----------
        uxx : array
            Second derivative of displacement filed.

        Returns
        -------
        array
            Internal moment :math:`m` obtained from the curvature.
        """
        moment = self.EI * uxx
        return moment

    def curvature(self, m):
        r"""
        Convert moment into curvature.

        .. math::
            \kappa = \frac{1}{EI} m

        Parameters
        ----------
        m : array
            Internal moment.

        Returns
        -------
        array
            Second derivative of the displacement field :math:`\kappa` obtained from the moment..
        """
        uxx = m / self.EI  # Linear material.
        return uxx


class Softening(Constitutive):
    r"""
    A simple non-linear constitutive equation for beams.

    This non-linear constitutive equation represents a simple softening equation. It does retain symmetry between
    tension and compression. It is represented by the following two equations.

    .. math::
            M = a S(\frac{\kappa}{b} - \frac{1}{2}) \quad \text{and} \quad
            \kappa = b S^{-1}(\frac{M}{a} + \frac{1}{2})

    where :math:`S` is the logistic function.

    Parameters
    ----------
    a : float
        The linear scaling of the stiffness response.
    b : float
        The non-linear scaling of the constitutive.

    Attributes
    ----------
    a : float
        The linear scaling of the stiffness response.
    b : float
        The non-linear scaling of the constitutive.
    """
    def __init__(self, a, b):
        r"""
        This simple non-linear constitutive equation contains two constants, `a` and `b`.
        """
        super().__init__()
        self.a = a
        self.b = b

    def moment(self, uxx):
        r"""
        Converts curvature into moment.

        .. math::
            M = a S(\frac{\kappa}{b} - \frac{1}{2})

        Parameters
        ----------
        uxx : array
            Second derivative of displacement field.

        Returns
        -------
        array
            Internal moment :math:`M(x)` obtained from the curvature.
        """
        moment = self.a * (expit(uxx/self.b) - 1/2)  # non-linear example material.
        return moment

    def curvature(self, m):
        r"""
        Converts moment into curvature.

        .. math::
            \kappa = b S^{-1}(\frac{M}{a} + \frac{1}{2})

        Parameters
        ----------
        m : array
            Internal moment.

        Returns
        -------
        array
            Second derivative of the displacement field :math:`\kappa` obtained from the moment.
        """
        uxx = self.b * logit(m/self.a + 0.5)  # Non-linear example material.
        return uxx

    def plot(self, uxx):
        r"""
        Plot the constitutive equation for all given curvature values.

        Parameters
        ----------
        uxx : array
            Curvatures for which the constitutive equation is plotted.
        """
        super().plot(uxx)

        equation = _m(r"$M(u'') = a ( S(\frac{u''}{b}) - \frac{1}{2})$ \\ ")
        constants = _m(rf"$a= {self.a}$\\ $b = {self.b}$")
        loc = (uxx.min(), -0.9*self.moment(uxx.min()))
        plt.annotate(equation+constants, loc)


class Hardening(Constitutive):
    r"""
    A basic hardening equation for beams.

    This non-linear constitutive equation represents hardening. It does retain symmetry between
    tension and compression. It is represented by the following two equations.

    .. math::
            \kappa = a S(\frac{M}{b} - \frac{1}{2}) \quad \text{and} \quad
            M = b S^{-1}(\frac{\kappa}{a} + \frac{1}{2})

    where :math:`S` is the logistic function.

    Parameters
    ----------
    a : float
        The linear scaling of the stiffness response.
    b : float
        The non-linear scaling of the constitutive.

    Attributes
    ----------
    a : float
        The linear scaling of the stiffness response.
    b : float
        The non-linear scaling of the constitutive.
    """
    def __init__(self, a, b):
        r"""
        This simple non-linear constitutive equation contains two constants, `a` and `b`.
        """
        super().__init__()
        self.a = a
        self.b = b

    def moment(self, uxx):
        r"""
        Converts curvature into moment.

        .. math::
            M = b S^{-1}(\frac{\kappa}{a} + \frac{1}{2})

        Parameters
        ----------
        uxx : array
            Second derivative of displacement field.

        Returns
        -------
        array
            Internal moment :math:`M(x)` obtained from the curvature.
        """
        moment = self.b * logit(uxx/self.a + 0.5)
        return moment

    def curvature(self, m):
        r"""
        Converts moment into curvature.

        .. math::
            \kappa = a S(\frac{M}{b} - \frac{1}{2})

        Parameters
        ----------
        m : array
            Internal moment.

        Returns
        -------
        array
            Curvature (second derivative of the displacement) field :math:`\kappa` obtained from the moment.
        """
        uxx = self.a * (expit(m/self.b) - 1/2)
        return uxx
