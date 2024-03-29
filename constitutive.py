r"""
A collection of arbitrary constitutive models for Poisson equations.

The data-driven method should be independent of the constitutive equation considered. It is however included for two
reasons. Firstly the database is created in-silico, which requires a traditional method to solve the Poisson problem,
these methods require constitutive equations. Secondly the constitutive equation can be used to evaluate the
performance of the data-driven method by comparing its solution to reference solutions that also depend on
traditional methods.

In general the Poisson problem is formulated as:

.. math::
    \nabla^2 \psi = rhs(x)\\

    \text{where:} \psi = f(u(x))\\

    \text{where:} u(x) = \tilde{u} \quad \forall x\in\Gamma

Here we define math:`f(u(x))` as the constitutive equation.

.. note:: Any future class should be a child of the :py:class:`~constitutive.Constitutive` which specifies the minimal
    information that is to be available in a constitutive class. It will provide errors if any of the required functions
    are not implemented by the child class.

Bram van der Heijden |br|
Mechanics of Composites for Energy and Mobility |br|
KAUST |br|
2023
"""

# Importing required modules.
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

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
    def field_to_potential(self, u):
        r"""
        The constitutive equation.

        The constitutive equation model converts the field :math:`u` into the potential :math:`\psi=f(u)`.

        Parameters
        ----------
        u : array
            The field to be converted into the potential.

        Returns
        -------
        potential
            The potential :math:`\psi` related to this field.
        """
        pass

    @abstractmethod
    def potential_to_field(self, potential):
        r"""
        The inverse constitutive equation.

        The material model converts the field :math:`u` into the potential :math:`\psi`, this performs the inverse
        operation.

        Parameters
        ----------
        potential : array
            Potential :math:`\psi` to be converted back into the main field.

        Returns
        -------
        array
            The unknown field :math:`u`.
        """
        pass

    def plot(self, u):
        r"""
        Plot the constitutive equation for the given field values.

        Parameters
        ----------
        u : array
            The field for which the constitutive equation is plotted.
        """
        m = self.field_to_potential(u)

        plt.plot(u, m)
        plt.xlabel(_m(r"Temperature gradient $\frac{du}{dx}$"))
        plt.ylabel(_m(r"Heat flux $q$"))


class LinearMaterial(Constitutive):
    r"""
    A simple linear constitutive equation.

    This linear constitutive equation is traditionally the one used.

    .. math::
            \psi = a u(x)

    Parameters
    ----------
    a : float
        The stiffness constant.

    Attributes
    ----------
    a : float
        The stiffness constant.
    """
    def __init__(self, a):
        r"""
        This simple linear constitutive equation depends on the bending stiffness :math:`EI`.
        """
        super().__init__()
        self.a = float(a)

    def field_to_potential(self, u):
        r"""
        The constitutive equation.

        The constitutive equation model converts the field :math:`u` into the potential :math:`\psi`. This is
        represented by the following equation:

        .. math::
            \psi = a u(x)

        Parameters
        ----------
        u : array
            The field to be converted in the potential.

        Returns
        -------
        array
            The potential :math:`\psi`.
        """
        potential = self.a * u
        return potential

    def potential_to_field(self, potential):
        r"""
        The inverse constitutive equation.

        The material model converts the field :math:`u` into the potential :math:`f(u)`, this performs the inverse
        operation. This occurs according to the following linear equation:

        .. math::
            u(x) = \frac{1}{a} \psi

        Parameters
        ----------
        potential : array
            Potential :math:`\psi` to be converted back into the main field.

        Returns
        -------
        array
            The unknown field :math:`u`.
        """
        u = potential / self.a  # Linear material.
        return u


class Hardening(Constitutive):
    r"""
    A non-linear constitutive equation with increasing stiffness.

    .. math::
        \psi(x) = a u**2 + b u

    where :math:`a` and :math:`b` are both positive numbers.

    Parameters
    ----------
    a : float
        The non-linear (third order) scaling factor.
    b : float
        The linear scaling factor.

    Parameters
    ----------
    a : float
        The non-linear (third order) scaling factor.
    b : float
        The linear scaling factor.
    """

    def __init__(self, a, b):
        r"""
        This simple non-linear constitutive equation.
        """
        super().__init__()
        self.a = float(a)
        self.b = float(b)

    def field_to_potential(self, u):
        r"""
        The constitutive equation.

        The constitutive equation model converts the field :math:`u` into the potential :math:`\psi`. This is
        represented by the following equation:

        .. math::
            \psi = a u**2 + b u

        Parameters
        ----------
        u : array
            The field to be converted in the potential.

        Returns
        -------
        array
            The potential :math:`\psi`.
        """
        potential = self.a * u ** 3 + self.b * u
        return potential

    def potential_to_field(self, potential):
        r"""
        The inverse constitutive equation.

        The material model converts the field :math:`u` into the potential :math:`f(u)`, this performs the inverse
        operation. This occurs according to the following linear equation:

        .. math::
            u(x) = - \frac{\sqrt[3]{\frac{\sqrt{\frac{729 \psi^{2}}{a^{2}} + \frac{108 b^{3}}{a^{3}}}}{2} -
            \frac{27 \psi}{2 a}}}{3} + \frac{b}{a \sqrt[3]{\frac{\sqrt{\frac{729 \psi^{2}}{a^{2}} +
            \frac{108 b^{3}}{a^{3}}}}{2} - \frac{27 \psi}{2 a}}}

        Parameters
        ----------
        potential : array
            Potential :math:`\psi` to be converted back into the main field.

        Returns
        -------
        array
            The unknown field :math:`u`.
        """
        u = -(np.sqrt(729*potential**2/self.a**2 + 108*self.b**3/self.a**3)/2 - 27*potential/(2*self.a))**(1/3)/3 \
            + self.b/(self.a*(np.sqrt(729*potential**2/self.a**2 + 108*self.b**3/self.a**3)/2 - 27*potential/(2*self.a))**(1/3))
        return u


class Softening(Constitutive):
    r"""
    A non-linear constitutive equation with decreasing stiffness.

    .. math::
        \psi = - \frac{\sqrt[3]{\frac{\sqrt{\frac{729 u^{2}}{a^{2}} + \frac{108 b^{3}}{a^{3}}}}{2} -
        \frac{27 u}{2 a}}}{3} + \frac{b}{a \sqrt[3]{\frac{\sqrt{\frac{729 u^{2}}{a^{2}} +
        \frac{108 b^{3}}{a^{3}}}}{2} - \frac{27 u}{2 a}}}

    where :math:`a` and :math:`b` are both positive numbers.

    Parameters
    ----------
    a : float
        The non-linear (third order) scaling factor.
    b : float
        The linear scaling factor.

    Parameters
    ----------
    a : float
        The non-linear (third order) scaling factor.
    b : float
        The linear scaling factor.
    """

    def __init__(self, a, b):
        r"""
        This simple non-linear constitutive equation.
        """
        super().__init__()
        self.a = float(a)
        self.b = float(b)

    def field_to_potential(self, u):
        r"""
        The constitutive equation.

        The constitutive equation model converts the field :math:`u` into the potential :math:`\psi`. This is
        represented by the following equation:

        .. math::
            \psi = - \frac{\sqrt[3]{\frac{\sqrt{\frac{729 u^{2}}{a^{2}} + \frac{108 b^{3}}{a^{3}}}}{2} -
            \frac{27 u}{2 a}}}{3} + \frac{b}{a \sqrt[3]{\frac{\sqrt{\frac{729 u^{2}}{a^{2}} +
            \frac{108 b^{3}}{a^{3}}}}{2} - \frac{27 u}{2 a}}}

        Parameters
        ----------
        u : array
            The field to be converted in the potential.

        Returns
        -------
        array
            The potential :math:`\psi`.
        """
        potential = -(np.sqrt(729*u**2/self.a**2 + 108*self.b**3/self.a**3)/2 - 27*u/(2*self.a))**(1/3)/3 \
            + self.b/(self.a*(np.sqrt(729*u**2/self.a**2 + 108*self.b**3/self.a**3)/2 - 27*u/(2*self.a))**(1/3))
        return potential

    def potential_to_field(self, potential):
        r"""
        The inverse constitutive equation.

        The material model converts the field :math:`u` into the potential :math:`f(u)`, this performs the inverse
        operation. This occurs according to the following linear equation:

        .. math::
            u(x) = a\psi**2 + b\psi

        Parameters
        ----------
        potential : array
            Potential :math:`\psi` to be converted back into the main field.

        Returns
        -------
        array
            The unknown field :math:`u`.
        """
        u = self.a * potential ** 3 + self.b * potential
        return u
