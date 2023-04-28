"""Definitions of global constants used in cofe.space module."""


# site tolerance is the absolute tolerance passed to the pymatgen coord utils
# to determine mappings and subsets. The default in pymatgen is 1E-8.
# If you want to tighten or loosen this tolerance, do it at runtime in your
# script by adding these lines somewhere at the top.
# import smol.cofe.space.constants
# constants.SITE_TOL = your_desired_value
SITE_TOL = 1e-6
