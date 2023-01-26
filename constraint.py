r"""
A collection of constraints that can be used in beam problems.

The classes below can be called to define a variety of constraints, the :py:class:`~constraint.PointConstraint` allows
for the definition of a constraint at a singular point, think of a point load or simple support. Whereas the
:py:class:`~constraint.LinearConstraint` defines a continuous constraint within a given range, this constraint can
be of constant magnitude or be linearly changing.

.. note:: Any future class should be a child of the :py:class:`~constraint.Constraint` which specifies the minimal
    information that is to be defined by this type of object.

.. note:: At some point it might be better to separate the kinematic and static constraints as some functions, such as
    :py:meth:`~constraint.Constraint.satisfy_value_free_translation` and
    :py:meth:`~constraint.Constraint.global_equilibrium_contribution` are never called for a kinematic constant.

The boundary conditions are given as a point constraint, and the forcing function can consist both of linear and
point constraints.

Bram Lagerweij |br|
Mechanics of Composites for Energy and Mobility |br|
KAUST |br|
2021
"""

# Import external modules.
from abc import ABC, abstractmethod
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


class Constraint(ABC):
    r"""
    The definition of a constraint which describes the state at a certain point or range.

    All other constraints should be based upon this one.
    """

    def __init__(self):
        r"""
        In general the initialization of a constraint contains the range where it is active and its magnitude.
        """
        pass

    @abstractmethod
    def crop(self, x_start, x_end, closed_start=True, closed_end=True):
        r"""
        Crop the constraint to the domain prescribed.

        If the constraint is active in the prescribed domain, a new version of the constraint should be returned, but
        such that it only includes information on the domain `x_start` to `x_end`. Constraints will be enforced
        strongly, however if a static constraints acts on an internal boundary it can be assumed open-ended, that is the
        constraint is inactive at the place where the subdomain is cut from the problem domain.

        Parameters
        ----------
        x_start : float
            Starting location of the subdomain.
        x_end : float
            Ending location of the subdomain.
        closed_start : bool, optional
            Whether the constraint should be active if it is exactly on the start of the boundary, 'True' by default.
        closed_end : bool, optional
            Whether the constraint should be active if it is exactly on the start of the boundary, 'True' by default.

        Returns
        -------
        Constraint
            A constraint for the region x_start to x_end
        """
        pass

    def satisfy_value(self, x, field):
        r"""
        Verify whether the field :math:`f(x)` satisfies the constraint.

        Parameters
        ----------
        x : array
            Location where the field is defined.
        field : array
            The field that is proposed.

        Returns
        -------
        bool
            `True` if the proposed field satisfies the constraint, `False` otherwise.
        """
        # Obtain value of the constraint where active.
        constraint_x = self.value(x)

        # Constraint equal field -> pass, and when constraint is inactive (nan) -> pass.
        satisfy = np.all((constraint_x == field) | np.isnan(constraint_x))
        return satisfy

    @abstractmethod
    def value(self, x):
        r"""
        The value of the constraint at :math:`x` if any, `None` otherwise.

        Parameters
        ----------
        x : array
            Location where the constraint is to be evaluated.

        Returns
        -------
        array
            The value of the constraint.
        """
        pass

    @abstractmethod
    def satisfy_value_free_translation(self, xi, field, translation):
        r"""
        Find the range of translations :math:`x=\xi+t` for which field :math:`f(\xi)` satisfy the constraint.

        Parameters
        ----------
        xi : array
            The local coordinates :math:`\xi` in which the field is defined.
        field : array
            The field :math:`f(\xi)` in question.
        translation : tuple
            The maximum (`float`) and minimum (`float`) coordinates translations that are considered.

        Returns
        -------
        list
            A list with all possible translation ranges for which the field satisfies the constraint. The list is
            empty if the constraint cannot be satisfied.
        """
        pass

    @abstractmethod
    def plot(self, ax, marker, color, offset=0):
        r"""
        Find the range of translations :math:`x=\xi+t` for which field :math:`f(\xi)` satisfy the constraint.

        Parameters
        ----------
        ax : matplotlib.Axis
            The axis in which the constraint has to be plotted.
        marker : string
            The string that describes the marker that has to be placed.
        color : string
            The color of the marker.
        offset : float, optional
            The offset from the zero line that defines the bar.
        """
        pass


class PointConstraint(Constraint):
    r"""
    The definition of a point constraint.

    The constraint is prescribes a certain value at a specific point.

    Parameters
    ----------
    x : float
        Coordinate where the point constraint is active.
    magnitude : float
        Magnitude of the specified constraint.

    Attributes
    ----------
    x : float
        Coordinate where the point constraint is active.
    magnitude : float
        Magnitude of the specified constraint.
    """

    def __init__(self, x, magnitude):
        """
        Initialize a point constraint.
        """
        super().__init__()
        self.x = x
        self.magnitude = magnitude

    def crop(self, x_start, x_end, closed_start=True, closed_end=True):
        r"""
        Crop the constraint to the domain prescribed.

        If the constraint is active in the prescribed domain, a new version of the constraint should be returned, but
        such that it only includes information on the domain `x_start` to `x_end`. Constraints will be enforced
        strongly, however if a static constraints acts on an internal boundary it can be assumed open-ended, that is the
        constraint is inactive at the place where the subdomain is cut from the problem domain.

        Parameters
        ----------
        x_start : float
            Starting location of the subdomain.
        x_end : float
            Ending location of the subdomain.
        closed_start : bool, optional
            Whether the constraint should be active if it is exactly on the start of the boundary, 'True' by default.
        closed_end : bool, optional
            Whether the constraint should be active if it is exactly on the start of the boundary, 'True' by default.

        Returns
        -------
        PointConstraint
            A point constraint for the region x_start to x_end
        """
        # Test if the constraint is within the subdomain.
        within_subdomain = x_start <= self.x <= x_end

        # Test if the constraint should be disregarded because it is at the start or end of an internal domain.
        open_start = self.x == x_start and not closed_start
        open_end = self.x == x_end and not closed_end

        if not within_subdomain or open_start or open_end:
            return None
        else:
            return PointConstraint(self.x, self.magnitude)

    def value(self, x, rtol=1e-8, atol=1e-8):
        r"""
        The value of the constraint at :math:`x` if any, `np.NaN` otherwise.

        .. warning::
            Tests location for `x` using the `isclose()` funtion, this can be inaccurate for very small numbers. In
            general this should not result in any issues, if it does the `rtol` and `atol` can be tweaked.

        Parameters
        ----------
        x : array
            Location where the constraint is to be evaluated.
        rtol : float, optional
            The relative tolerance parameter.
        atol : float, optional
            The absolute tolerance parameter.

        Returns
        -------
        array
            The value of the constraint at :math:`x`.
        """
        # Determine if any x is close to the location of this constraint.
        close = np.isclose(x, self.x, rtol=rtol, atol=atol)

        if isinstance(x, np.ndarray):  # Return array if x is an array:
            constraint_x = np.full_like(x, np.NaN, dtype=float)
            constraint_x[close] = self.magnitude
        elif isinstance(x, (float, int)):  # Return a float if x is a number.
            if close:
                constraint_x = self.magnitude
            else:
                constraint_x = np.NaN
        else:
            raise TypeError("x is not a number or array and cannot be compared to the location `self.x`")
        return constraint_x

    def satisfy_value_free_translation(self, xi, field, translation):
        r"""
        Find the range of translations :math:`x=\xi+t` for which field :math:`f(\xi)` satisfy the constraint.

        Parameters
        ----------
        xi : array
            The local coordinates :math:`\xi` in which the field is defined.
        field : array
            The field :math:`f(\xi)` in question.
        translation : tuple
            The maximum (`float`) and minimum (`float`) coordinates translations that are considered.

        Returns
        -------
        list
            A list with all possible translation ranges for which the field satisfies the constraint. The list is
            empty if the constraint cannot be satisfied.
        """
        # In case that the translation range is fixed, only checking if it satisfies the constraint is required.
        if translation[0] == translation[1]:
            t = translation[0]
            if self.satisfy_value(xi + t, field):  # The constraint is satisfied for the given translation.
                admissible_translations = [translation]
            else:  # The constraint is violated for the prescribed translation.
                admissible_translations = []

        # In case the the translation is still free we find at which locations we can satisfy it.
        else:
            satisfy = (field == self.magnitude)

            # Obtain all the chunks where the constrained is satisfied.
            chunk_list = []
            satisfy_masked = np.ma.MaskedArray(satisfy, np.zeros_like(satisfy))
            while True:
                # Find the first next index where we satisfy the constraint magnitude.
                where_start = np.ma.where(satisfy_masked)[0]
                if len(where_start) == 0:  # There is no next chunk anymore.
                    break
                start = where_start[0]
                satisfy_masked.mask[:start] = 1
                where_end = np.ma.where(np.invert(satisfy_masked))[0]
                if len(where_end) == 0:  # This chuck does not end within this patch
                    end = len(satisfy)
                else:  # The end of this chunk is within the patch.
                    end = where_end[0]
                satisfy_masked.mask[:end] = 1
                chunk_list.append((xi[start], xi[end-1]))

            # For all the chunks verify whether they are long enough and satisfy the set range of translation.
            admissible_translations = []
            for chunk in chunk_list:
                t_min = self.x - chunk[1]
                t_max = self.x - chunk[0]
                if not (t_min > translation[1] or t_max < translation[0]):  # Outside allowed translations.
                    translation_chunk = (max(translation[0], t_min), min(translation[1], t_max))
                    admissible_translations.append(translation_chunk)

        return admissible_translations

    def plot(self, ax, marker, color, offset=0):
        r"""
        Find the range of translations :math:`x=\xi+t` for which field :math:`f(\xi)` satisfy the constraint.

        Parameters
        ----------
        ax : matplotlib.Axis
            The axis in which the constraint has to be plotted.
        marker : string
            The string that describes the marker that has to be placed.
        color : string
            The color of the marker.
        offset : float, optional
            The offset from the zero line that defines the bar.
        """
        # Marker style
        if marker == "bvp":
            xy = [[self.x, 0], [self.x - offset / 2, -offset], [self.x + offset / 2, -offset]]
            patch = mpatches.Polygon(xy, color=color, closed=True, clip_on=False)
            ax.add_patch(patch)

        elif marker == "rhs":
            patch = mpatches.FancyArrow(self.x, 0, 0, self.magnitude * offset, width=offset / 4, color=color,
                                        clip_on=False)
            ax.add_patch(patch)


class LinearConstraint(Constraint):
    r"""
    The definition of a constraint that varies linearly within a specific range.

    Parameters
    ----------
    x_start : float
        Starting coordinate of the constraint, `x_start` itself is not included in the range.
    x_end : float
        Ending coordinate of the constraint, `x_start` itself is not included in the range.
    magnitude_start : float
        Magnitude of the constraint at `x_start`.
    magnitude_end : float, optional
        Magnitude of the constraint at `x_end`, if not provided it is assumed to be equal to `magnitude_start`.
    incl_start : bool, optional
        Whether to include the start coordinate or not, `True` by default.
    incl_end : bool, optional
        Whether to include the end coordinate or not, `True` by default.

    Attributes
    ----------
    x_start : float
        Starting coordinate of the constraint, `x_start` itself is not included in the range.
    x_end : float
        Ending coordinate of the constraint, `x_start` itself is not included in the range.
    magnitude_start : float
        Magnitude of the constraint at `x_start`.
    magnitude_end : float
        Magnitude of the constraint at `x_end`.
    incl_start : bool
        Whether to include the start coordinate or not.
    incl_end : bool
        Whether to include the end coordinate or not.
    """

    def __init__(self, x_start, x_end, magnitude_start, magnitude_end=None, incl_start=True, incl_end=True):
        """
        Initialize the linear range constraint.

        This is a generic linear constraint between `x_start` and `x_end`. If `magnitude_end` is not given
        the constraint is assumed to be a constant.
        """
        super().__init__()

        if magnitude_end is None:
            magnitude_end = magnitude_start

        self.x_start = x_start
        self.x_end = x_end
        self.magnitude_start = magnitude_start
        self.incl_start = incl_start
        self.magnitude_end = magnitude_end
        self.incl_end = incl_end

    def crop(self, x_start, x_end, closed_start=True, closed_end=True):
        r"""
        Crop the constraint to the domain prescribed.

        If the constraint is active in the prescribed domain, a new version of the constraint should be returned, but
        such that it only includes information on the domain `x_start` to `x_end`. Constraints will be enforced
        strongly, however if a static constraints acts on an internal boundary it can be assumed open-ended, that is the
        constraint is inactive at the place where the subdomain is cut from the problem domain.

        Parameters
        ----------
        x_start : float
            Starting location of the subdomain.
        x_end : float
            Ending location of the subdomain.
        closed_start : bool, optional
            Whether the constraint should be active if it is exactly on the start of the boundary, 'True' by default.
        closed_end : bool, optional
            Whether the constraint should be active if it is exactly on the start of the boundary, 'True' by default.

        Returns
        -------
        LinearConstraint
            A constraint for the region x_start to x_end
        """
        # Any constraint acting outside of the patch will be removed.
        if self.x_start <= x_end and x_start <= self.x_end:
            # Determine the start and end points of the new constraint domain.
            start = max(self.x_start, x_start)
            end = min(self.x_end, x_end)

            # Calculate the magnitude of the constraints at the start and end of the new constraint domain.
            value_start = self._value(start)
            value_end = self._value(end)

            # Check whether the start of the constraint should be included.
            if (self.x_start < x_start and closed_start) \
                    or (self.x_start == x_start and closed_start and self.incl_start) \
                    or (self.x_start > x_start and self.incl_start):
                incl_start = True

            else:
                incl_start = False

            # Check whether the end of the constraint should be included.
            if (self.x_end > x_end and closed_end) \
                    or (self.x_end == x_end and closed_end and self.incl_end) \
                    or (self.x_end < x_end and self.incl_end):
                incl_end = True
            else:
                incl_end = False

            # Return a new Linear Constraint only if it has a length longer then zero.
            if start != end:
                return LinearConstraint(start, end, value_start, value_end, incl_start=incl_start, incl_end=incl_end)
            else:
                return None
        else:
            return None

    def _value(self, x):
        r"""
        The value of the constraint at :math:`x` at any location, even if it is outside the domain.

        .. warning:: For internal use only.

        Parameters
        ----------
        x : array
            Location where the constraint is to be evaluated.

        Returns
        -------
        array
            The value of the constraint at :math:`x`.
        """
        slope = (self.magnitude_end - self.magnitude_start) / (self.x_end - self.x_start)
        constant = - slope * self.x_start + self.magnitude_start
        value_x = slope * x + constant
        return value_x

    def value(self, x):
        r"""
        The value of the constraint at :math:`x` if any, `np.NaN` otherwise.

        Parameters
        ----------
        x : array
            Location where the constraint is to be evaluated.

        Returns
        -------
        array
            The value of the constraint at :math:`x`.
        """
        # Determine whether x is within the constraint range.
        if self.incl_start:
            active_start = (self.x_start <= x)
        else:
            active_start = (self.x_start < x)

        if self.incl_end:
            active_end = (x <= self.x_end)
        else:
            active_end = (x < self.x_end)

        active = active_start & active_end

        if isinstance(x, np.ndarray):  # Return array if x is an array:
            value_x_in_range = np.full_like(x, np.NaN, dtype=float)
            value_x_in_range[active] = self._value(x)[active]
        elif isinstance(x, (float, int)):  # Return a float if x is a number.
            if active:
                value_x_in_range = self.value(x)
            else:
                value_x_in_range = np.NaN
        else:
            raise TypeError("x is not a number or array and cannot be compared to the location `self.x`")
        return value_x_in_range

    def satisfy_value_free_translation(self, xi, field, translation):
        r"""
        Find the translations :math:`x=\xi+t` for which field :math:`f(\xi)` satisfy the constraint.

        Parameters
        ----------
        xi : array
            The local coordinates :math:`\xi` in which the field is defined.
        field : array
            The field :math:`f(\xi)` in question.
        translation : tuple
            The maximum (`float`) and minimum (`float`) coordinates translations that are considered.

        Returns
        -------
        list
            A list with all possible translation ranges for which the field satisfies the constraint. The list is
            empty if the constraint cannot be satisfied.
        """
        # In case that the translation range is fixed, only checking if it satisfies the constraint is required.
        if translation[0] == translation[1]:
            t = translation[0]
            if self.satisfy_value(xi + t, field):  # The constraint is satisfied for the given translation.
                admissible_translations = [translation]
            else:  # The constraint is violated for the prescribed translation.
                admissible_translations = []

        # In case the the translation is still free we find at which locations we can satisfy it.
        else:
            # Differentiate between a constant constraint and a linear constraint.
            slope = (self.magnitude_end - self.magnitude_start) / (self.x_end - self.x_start)
            if slope == 0:  # The constraint has a constant magnitude
                satisfy = (field == self.magnitude_start)

                # Obtain all the chunks where the constrained is satisfied.
                chunk_list = []
                satisfy_masked = np.ma.MaskedArray(satisfy, np.zeros_like(satisfy))
                while True:
                    # Find the first next index where we satisfy the constraint magnitude.
                    where_start = np.ma.where(satisfy_masked)[0]
                    if len(where_start) == 0:  # There is no next chunk anymore.
                        break
                    start = where_start[0]
                    satisfy_masked.mask[:start] = 1
                    where_end = np.ma.where(np.invert(satisfy_masked))[0]
                    if len(where_end) == 0:  # This chuck does not end within this patch
                        end = len(satisfy)
                    else:
                        end = where_end[0]
                    satisfy_masked.mask[:end] = 1

                    if self.incl_start:  # Start point is included.
                        chunk_start = start
                    else:
                        chunk_start = max(start - 1, 0)

                    if self.incl_end:  # End point is included.
                        chunk_end = end - 1
                    else:
                        chunk_end = min(len(satisfy) - 1, end)
                    chunk_list.append((xi[chunk_start], xi[chunk_end]))

                # For all the chunks verify whether they are long enough and satisfy the set range of translation.
                admissible_translations = []
                for chunk in chunk_list:
                    if chunk[1] - chunk[0] >= self.x_end - self.x_start:  # If satisfied domain is long enough.
                        t_min = self.x_end - chunk[1]
                        t_max = self.x_start - chunk[0]
                        if not (t_min > translation[1] or t_max < translation[0]):  # Outside allowed translations.
                            translation_chunk = (max(translation[0], t_min), min(translation[1], t_max))
                            admissible_translations.append(translation_chunk)

            else:
                # The following translations would be required for each possible point.
                t_possible = set((field - slope * (xi - self.x_start) + self.magnitude_start) / slope)

                # Verify for all different potential translations the boundary condition.
                admissible_translations = []
                for t in t_possible:
                    if self.satisfy_value(xi + t, field):
                        if translation[0] <= t <= translation[1]:
                            admissible_translations.append((t, t))

        return admissible_translations

    def plot(self, ax, marker, color, offset=0):
        r"""
        Find the range of translations :math:`x=\xi+t` for which field :math:`f(\xi)` satisfy the constraint.

        Parameters
        ----------
        ax : matplotlib.Axis
            The axis in which the constraint has to be plotted.
        marker : string
            The string that describes the marker that has to be placed.
        color : string
            The color of the marker.
        offset : float, optional
            The offset from the zero line that defines the bar.
        """
        x = np.linspace(self.x_start, self.x_end, 10)
        y = self._value(x) + offset / 2

        if self.magnitude_start == self.magnitude_end == 0:
            marker = None

        plt.plot(x, y, marker=marker, color=color, clip_on=False)

        if self.incl_start:
            plt.plot(x[0], y[0], marker='o', color=color, clip_on=False)
        else:
            plt.plot(x[0], y[0], marker='o', color=color, markerfacecolor="None", clip_on=False)

        if self.incl_end:
            plt.plot(x[-1], y[-1], marker='o', color=color, clip_on=False)
        else:
            plt.plot(x[-1], y[-1], marker='o', color=color, markerfacecolor="None", clip_on=False)
