r"""
Some helper function.

These helper functions wil not affect the data-driven method itself, these are just scripts that can help plotting or
assist in other minor tasks.

Bram Lagerweij |br|
Mechanics of Composites for Energy and Mobility |br|
KAUST |br|
2021
"""

# Turn this to `True` to turn latex rendering off for all the strings in the project.
RAW_MATH = False


def _m(s):
    r"""
    Use latex to render math equations, or print the raw latex strings.

    Parameters
    ----------
    s : str
        String to be edited.

    Returns
    -------
    str
        String `s` with added `\$` in case the latex is turned to raw.
    """
    return s.replace("$", r"\$") if RAW_MATH else s
