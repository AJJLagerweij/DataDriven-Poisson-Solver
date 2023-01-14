r"""
The patch data-format, specification and creation of patches.

The database will contain all patches, where every patch contains at least a local coordinate system, the primal
field and right hand side loading. Essentially a patch is a region of a test, hence it will be based upon the general
:py:class:`~test.Test` object. Here the type 'patch' will be defined to ensure that this is implemented consistently.

Bram van der Heijden |br|
Mechanics of Composites for Energy and Mobility |br|
KAUST |br|
2023
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

    A patch is the region of a test, which was performed. As a result it can only include information that can be
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
        g_min = 0
        g_max = 0
        for p in range(num):
            patch = self.database[p]
            patch.plot([axis[0, p], axis[1, p]])

            # set max.
            u_max = max(patch.u.max(), u_max)
            u_min = min(patch.u.min(), u_min)
            g_max = max(patch.g.max(), g_max)
            g_min = min(patch.g.min(), g_min)

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
            axis[1, j].set_ylim(1.05*g_min, 1.05*g_max)

            # For all axis that are not the left column, remove the y-label.
            if j > 0:
                axis[0, j].set_yticks([])
                axis[0, j].set_ylabel('')
                axis[0, j].spines['left'].set_color('k')
                axis[1, j].set_yticks([])
                axis[1, j].set_ylabel('')
                axis[1, j].spines['left'].set_color('k')


class Patch(Test):
    r"""
    The definition of a patch for a beam problem.

    A patch is the region of a test which was performed. As a result it can only include information that can be
    obtained from classical test setups. That is the geometry, primal and right hand side external loading. These
    properties are know at discrete coordinates, as a point cloud. For simplicity the material is assumed to be
    uniform, hence it is not an attribute of the patch.

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
        Primal field at these local coordinates.
    g : array
        Applied right hand side loading :math:`g(x)`.
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
        self.g = test.g[index]


if __name__ == '__main__':
    # Import is required.
    from test import Laplace_Dirichlet_Dirichlet
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

    # Put this into the database.
    database = PatchDatabase()
    database.add_test(test)
    database.add_patches_from_test(test, np.pi)

    # Plot the database.
    database.plot()