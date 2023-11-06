"""
This module adds upsampling functionality to the SDS_preprocess module

"""

# load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb

# image processing modules
import skimage.transform as transform
import skimage.morphology as morphology
import sklearn.decomposition as decomposition
import skimage.exposure as exposure
from skimage.io import imsave
from skimage import img_as_ubyte

# other modules
from osgeo import gdal
from pyproj import CRS
from pylab import ginput
import pickle
import geopandas as gpd
import pandas as pd
from shapely import geometry
import re

# CoastSat modules
from coastsat import SDS_tools

np.seterr(all='ignore') # raise/ignore divisions by 0 and nans

###################################################################################################
# UPSAMPLING
###################################################################################################

def upsampling(im, factor):
    #this is a stub function to add upsampling
    
    return True
