'''
Definitions for site functions spaces.
These include the basis functions and measure that defines the inner product
'''

from abc import ABC
#TODO implement this

# A site function spaces specifies the function and probability measure at a site
# An expansion can have different site function spaces for each distinct site with different species
# that are allowed


class SiteFunctionSpace(ABC):

    def __init__(self, species):
        pass

    def