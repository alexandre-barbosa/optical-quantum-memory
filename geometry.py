# -*- coding: utf-8 -*-
"""
Author: Alexandre Barbosa
Contact: alexandre.barbosa@tecnico.ulisboa.pt
Last Updated: 18-07-2023
"""
from pde import SphericalSymGrid

ngrid = 1024 # number of grid points
grid = SphericalSymGrid(radius=(0, 1), shape=ngrid)  # generate grid