r"""
The patch data-format, specification and creation of patches.

The database will contain all patches, where every patch contains at least a local coordinate system, a displacement
field and external loading fields. Essentially a patch is a region of a test, hence it will be based upon the general
:py:class:`~test.Test` object. Because the data-driven nature it will not contain internal loading. Here the
type 'patch' will be defined to ensure that this is implemented consistently.

Bram Lagerweij |br|
Mechanics of Composites for Energy and Mobility |br|
KAUST |br|
2021
"""

# Import external packages
import numpy as np
import matplotlib.pyplot as plt

# Import from my own scripts.
from test import Test
from helperfunctions import _m


class PatchDatabase(object):
    r"""
    The PatchDatabase object contains all patches.

    A patch is the region of a test which was performed. As a result it can only include information that can be
    obtained from classical test setups. That is the geometry, deformation and external loading. These properties are
    know at discrete coordinates, as a point cloud. For simplicity the material is assumed to be uniform, hence it is
    not an attribute of the patch.

    Attributes
    ----------
    database : list
        A list containing all the different patches.
    """

    def __init__(self):
        r"""
        Initialize the database by creating an empty list with patches.
        """
        self.database = []

    def num_patches(self):
        """
        Calculate the number of patches in the database.

        Returns
        -------
        int
            The number of patches in the database.
        """
        return len(self.database)

    def add_patches_from_test(self, test, patch_length, overlap_length=None):
        r"""
        Cut the test results in patches and add those to the database.

        This function adds a test to the database, it harvest the patches from the tests and adds it the the list with
        patches. The overlapping length is set to half the patch length (which is optimal). If the last patch extents
        beyond the test domain it is shifted to fit exactly in the testing domain.

        Parameters
        ----------
        test : Test
            The test from which the patches will be harvested and added to the database.
        patch_length : float
            The length of a patch.
        overlap_length : float, optional
            The length of the overlapping region, defaults to `patch_length/2`.
        """
        if overlap_length is None:
            overlap_length = patch_length/2

        # Calculate the amount of patches that will be harvested.
        test_length = test.x[-1]
        tot_num_patches = int(np.ceil((test_length - patch_length)/(patch_length - overlap_length))) + 1

        # Initialize an empty database.
        for patch_num in range(tot_num_patches):
            # Define the start and end of a patch.
            x_start = patch_num * (patch_length - overlap_length)
            x_end = x_start + patch_length

            # When the patch extents out of the specimen, it is shift forward as to make it fit.
            if x_end > test_length:
                x_end = test_length
                x_start = x_end - patch_length

            # Obtain the patch and add it to the database.
            patch = Patch(test, x_start, x_end)
            self.add_patch(patch)

    def add_test(self, test):
        """
        Add the results of a test to the database.

        This function adds a test to the database, it creates a single patch that contains the entirety of the test
        result.

        Parameters
        ----------
        test : Test
            The test from which the patches will be harvested and added to the database.
        """
        patch = Patch(test, test.x[0], test.x[-1])
        self.add_patch(patch)

    def add_patch(self, patch):
        r"""
        Adding a single patch to the database.
        
        Parameters
        ----------
        patch : Patch
            The patch object that is to be added.
        """
        self.database.append(patch)

    def rotate(self):
        r"""
        Adds a rotated version of all patches to the database.

        Rotating patches is an admissible coordinate transformation, in beam problems the only rotation available
        rotates the patch by exactly 180 degrees. This function will copy the patches in the database, rotate them and
        append them to the end of the database.

        .. note:: This type of coordinate transformation would be more free and less discrete for problems other then
            beam problems, in those cases this rotate_patches should be integrated as a coordinate transformation that
            has to be determined by the rigid body motion and coordinate transformation solver in
            :py:class:`~configuration.Configuration`, in that case this function should be removed.

        .. warning:: Only rotate the database once, rotating multiple time will create many duplicate patches, this will
            increase the computational complexity and memory used by the solver and will not result in different or more
            accurate solutions. Only add this rotation after adding all patches to the database.

        Returns
        -------
        PatchDatabase
            The instance itself is returned.
        """
        # Create the rotated versions.
        rotated_database = [patch.rotate() for patch in self.database]

        # Append this rotated database to this database.
        self.database += rotated_database
        return self

    def mirror(self):
        r"""
        Adds a mirrored version of all patches to the database.

        Mirroring patches is an admissible coordinate transformation, This function will copy the patches in the
        database, mirror them and append them to the end of the database.

        .. note:: This type of coordinate transformation will always be discrete, it is either a mirrored version of the
            patch or not. Hence this should not be moved into the coordinate transformation optimization.

        .. warning:: Only mirror the database once, mirroring multiple time will create many duplicate patches, this
            will increase the computational complexity and memory used by the solver and will not result in different or
            more accurate solutions. Only add this rotation after adding all patches to the database.

        Returns
        -------
        PatchDatabase
            The instance itself is returned.
        """
        # Create the mirrors.
        mirrored_database = [patch.mirror() for patch in self.database]

        # Append this rotated database to this database.
        self.database += mirrored_database
        return self

    def plot(self, num=None):
        r"""
        Plot the deformation and internal static state of the test setup.

        Parameters
        ----------
        material : Constitutive
            The material's constitutive equation is required to plot the moment and shear curves.
        num : int, optional
            The number of patches that you want to plot, by default it plots all patches.
        """
        # Obtain the number of patches in the database
        if num is None:
            num = self.num_patches()
        if 40 < num:
            raise ValueError("To many subplots to make. You should not plot this manny patches.")

        # Create the axis objects.
        fig, axis = plt.subplots(2, num, figsize=(10, 6),
                                 gridspec_kw={'height_ratios': [3, 2], 'wspace': 0, 'hspace': 0})

        u_max = 0
        u_min = 0
        M_max = 0
        M_min = 0
        V_max = 0
        V_min = 0
        for p in range(num):
            patch = self.database[p]
            patch.plot([axis[0, p], axis[1, p]], annotate=False)

            # set max.
            u_max = max(patch.u.max(), u_max)
            u_min = min(patch.u.min(), u_min)
            M_max = max(patch.M_int.max(), M_max)
            M_min = min(patch.M_int.min(), M_min)
            V_max = max(patch.V_int.max(), V_max)
            V_min = min(patch.V_int.min(), V_min)

        # Remove all double objects, such as axis ticks and labels.
        for j in range(num):
            # Create a single legend.
            if j == 0:
                handles = axis[0, j].get_legend().legendHandles
                labels = []
                for line in handles:
                    labels.append(line.get_label())
                plt.figlegend(handles, labels, loc='upper center', ncol=3, frameon=False)
            axis[0, j].get_legend().remove()

            # Remove the axis labels that are not on the bottom row.
            axis[0, j].set_xticks([])
            axis[0, j].set_xlabel('')
            axis[1, j].set_xlabel(_m(r"$\xi$ in mm"))

            # Set the limits according to the u_min and u_max.
            axis[0, j].set_ylim(1.05*u_min, 1.05*u_max)
            axis[0, j].set_title(f'Patch {j}')

            # Set the limits according to the M_min and M_max.
            axis[1, j].set_ylim(1.05*M_min, 1.05*M_max)

            # Set the limits according to the V_min and V_max.
            twin = axis[1, j].get_shared_x_axes().get_siblings(axis[1, j])[0]
            twin.set_ylim(1.05*V_min, 1.05*V_max)

            # For all axis that are not the left column, remove the y-label.
            if j > 0:
                axis[0, j].set_yticks([])
                axis[0, j].set_ylabel('')
                axis[0, j].spines['left'].set_color('k')
                axis[1, j].set_yticks([])
                axis[1, j].set_ylabel('')
                axis[1, j].spines['left'].set_color('k')
                twin.spines['left'].set_color('k')

            if j < num - 1:
                # For all the axis that are on the bottom row and not the right column, remove the y-label
                # and all the ticks.
                twin.set_yticks([])
                twin.set_ylabel('')


class Patch(Test):
    r"""
    The definition of a patch for a beam problem.

    A patch is the region of a test which was performed. As a result it can only include information that can be
    obtained from classical test setups. That is the geometry, deformation and external loading. These properties are
    know at discrete coordinates, as a point cloud. For simplicity the material is assumed to be uniform, hence it is
    not an attribute of the patch.

    .. note:: The `moment()` and `shear()` functions are not Data-Driven as they require the specification of an
        constitutive equation, which is called EI here.

    Parameters
    ----------
    test : Test
        The test class obtains the test results from which the patch is harvested.
    x_start : float
        The coordinate that specifies the start of where the patch is cut out of the test.
    x_end : float
        The coordinate that specifies the end of where the patch is cut out of the test.

    Attributes
    ----------
    x : array
        Local coordinates.
    u : array
        Displacement at these local coordinates.
    M : array
        Externally applied moments at the local coordinates, includes reaction moments.
    V : array
        Externally applied shear at the local coordinates, includes reaction forces. The pressure will be includes as
        lumped shear forces.
    end : array
        Whether the coordinate intersects with the end of the beam, if so `True`, `False` Otherwise.
    """

    def __init__(self, test, x_start, x_end):
        r"""
        Extract the patch from a given test problem and domain.
        """
        # Initialize parent class.
        super().__init__()

        # Obtain the index related to the specified domain.
        index = np.where((x_start <= test.x) & (test.x <= x_end))[0]

        # Extract the patch attributes:
        self.x = test.x[index] - test.x[index][0]
        self.u = test.u[index]
        self.M = test.M[index]
        self.M_int = test.M_int[index]
        self.V = test.V[index]
        self.V_int = test.V_int[index]
        self.end = test.end[index]
