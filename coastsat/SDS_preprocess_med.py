"""
This module contains all the functions needed to preprocess the satellite images
before the shorelines can be extracted. This includes creating a cloud mask and
pansharpening/downsampling the multispectral bands.

Author: Kilian Vos, Water Research Laboratory, University of New South Wales
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
from coastsat import SDS_preprocess

np.seterr(all='ignore') # raise/ignore divisions by 0 and nans

# function to preprocess a med res satellite image
def preprocess(fn, satname, cloud_mask_issue, pan_off=True, s2cloudless_prob=40):
    """
    Reads the image and outputs the pansharpened/down-sampled multispectral bands,
    the georeferencing vector of the image (coordinates of the upper left pixel),
    the cloud mask, the QA band and a no_data image.
    For Landsat 7-8 it also outputs the panchromatic band and for Sentinel-2 it
    also outputs the 20m SWIR band.

    KV WRL 2018

    Arguments:
    -----------
    fn: str or list of str
        filename of the .TIF file containing the image. 
    satname: str
        name of the satellite mission (e.g., 'L5')
    cloud_mask_issue: boolean
        True if there is an issue with the cloud mask and sand pixels are being masked on the images
    pan_off : boolean
        if True, disable panchromatic sharpening and ignore pan band
    s2cloudless_prob: float [0,100)
        threshold to identify cloud pixels in the s2cloudless probability mask
        
    Returns:
    -----------
    im_ms: np.array
        3D array containing the pansharpened/down-sampled bands (B,G,R,NIR,SWIR1)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale] defining the
        coordinates of the top-left pixel of the image
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_extra : np.array
        2D array containing the 20m resolution SWIR band for Sentinel-2 and the 15m resolution
        panchromatic band for Landsat 7 and Landsat 8. This field is empty for Landsat 5.
    im_QA: np.array
        2D array containing the QA band, from which the cloud_mask can be computed.
    im_nodata: np.array
        2D array with True where no data values (-inf) are located

    """
    
    if isinstance(fn, list):
        fn_to_split=fn[0]
    elif isinstance(fn, str):
        fn_to_split=fn
    # split by os.sep and only get the filename at the end then split again to remove file extension
    fn_to_split=fn_to_split.split(os.sep)[-1].split('.')[0]
    # search for the year the tif was taken with regex and convert to int
    year = int(re.search('[0-9]+',fn_to_split).group(0))
                  
    #=============================================================================================#
    # S2 images
    #=============================================================================================#
    if satname == 'S2':
        # read 10m bands (R,G,B,NIR)
        if isinstance(fn, list):
            fn_ms=fn[0]
        elif isinstance(fn, str):
            fn_ms=fn

        data = gdal.Open(fn_ms, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount-1)]
        im_ms = np.stack(bands, 2)
        im_ms = im_ms/10000 # TOA scaled to 10000
        # read s2cloudless cloud probability (last band in ms image)
        cloud_prob = data.GetRasterBand(data.RasterCount).ReadAsArray()

        # image size
        nrows = im_ms.shape[0]
        ncols = im_ms.shape[1]
        # if image contains only zeros (can happen with S2), skip the image
        if sum(sum(sum(im_ms))) < 1:
            im_ms = []
            georef = []
            # skip the image by giving it a full cloud_mask
            cloud_mask = np.ones((nrows,ncols)).astype('bool')
            return im_ms, georef, cloud_mask, [], [], []

        # read 20m band (SWIR1)
        fn_swir = fn[1]
        data = gdal.Open(fn_swir, gdal.GA_ReadOnly)
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_swir = bands[0]
        im_swir = im_swir/10000 # TOA scaled to 10000
        im_swir = np.expand_dims(im_swir, axis=2)

        # append down-sampled SWIR1 band to the other 10m bands
        im_ms = np.append(im_ms, im_swir, axis=2)

        # create cloud mask using 60m QA band (not as good as Landsat cloud cover)
        fn_mask = fn[2]
        data = gdal.Open(fn_mask, gdal.GA_ReadOnly)
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
        im_QA = bands[0]
        # compute cloud mask using QA60 band
        cloud_mask_QA60 = SDS_preprocess.create_cloud_mask(im_QA, satname, cloud_mask_issue, 'C02')
        # compute cloud mask using s2cloudless probability band
        cloud_mask_s2cloudless = SDS_preprocess.create_s2cloudless_mask(cloud_prob, s2cloudless_prob)
        # combine both cloud masks
        cloud_mask = np.logical_or(cloud_mask_QA60,cloud_mask_s2cloudless)
        
        # check if -inf or nan values on any band and create nodata image
        im_nodata = np.zeros(cloud_mask.shape).astype(bool)
        for k in range(im_ms.shape[2]):
            im_inf = np.isin(im_ms[:,:,k], -np.inf)
            im_nan = np.isnan(im_ms[:,:,k])
            im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
        # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
        # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
        im_zeros = np.ones(im_nodata.shape).astype(bool)
        im_zeros = np.logical_and(np.isin(im_ms[:,:,1],0), im_zeros) # Green
        im_zeros = np.logical_and(np.isin(im_ms[:,:,3],0), im_zeros) # NIR
        im_zeros = np.logical_and(np.isin(im_ms[:,:,4],0), im_zeros) # SWIR
        # add to im_nodata
        im_nodata = np.logical_or(im_zeros, im_nodata)
        # dilate if image was merged as there could be issues at the edges
        if 'merged' in fn_ms:
            im_nodata = morphology.dilation(im_nodata,morphology.square(5))

        # update cloud mask with all the nodata pixels
        cloud_mask = np.logical_or(cloud_mask, im_nodata)

        # no extra image
        im_extra = []

    #=============================================================================================#
    # PLSD images
    #=============================================================================================#
    if satname == 'PLSD':
        if isinstance(fn, list):
            fn_ms=fn[0]
        elif isinstance(fn, str):
            fn_ms=fn
        data = gdal.Open(fn_ms, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount-1)]
        im_ms = np.stack(bands, 2)
        im_ms = im_ms/10000 # TOA scaled to 10000
        # read s2cloudless cloud probability (last band in ms image)
        cloud_prob = data.GetRasterBand(data.RasterCount).ReadAsArray()

        # image size
        nrows = im_ms.shape[0]
        ncols = im_ms.shape[1]
        # if image contains only zeros (can happen with S2), skip the image
        if sum(sum(sum(im_ms))) < 1:
            im_ms = []
            georef = []
            # skip the image by giving it a full cloud_mask
            cloud_mask = np.ones((nrows,ncols)).astype('bool')
            return im_ms, georef, cloud_mask, [], [], []

        # create empty cloud mask because med res does not have bands to calc a cloud mask
        cloud_mask = np.zeros((nrows,ncols)).astype('bool')
        
        # check if -inf or nan values on any band and create nodata image
        im_nodata = np.zeros(cloud_mask.shape).astype(bool)
        for k in range(im_ms.shape[2]):
            im_inf = np.isin(im_ms[:,:,k], -np.inf)
            im_nan = np.isnan(im_ms[:,:,k])
            im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
        # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
        # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
        im_zeros = np.ones(im_nodata.shape).astype(bool)
        im_zeros = np.logical_and(np.isin(im_ms[:,:,1],0), im_zeros) # Green
        im_zeros = np.logical_and(np.isin(im_ms[:,:,3],0), im_zeros) # NIR
        im_zeros = np.logical_and(np.isin(im_ms[:,:,4],0), im_zeros) # SWIR
        # add to im_nodata
        im_nodata = np.logical_or(im_zeros, im_nodata)
        # dilate if image was merged as there could be issues at the edges
        if 'merged' in fn_ms:
            im_nodata = morphology.dilation(im_nodata,morphology.square(5))

        # update cloud mask with all the nodata pixels
        cloud_mask = np.logical_or(cloud_mask, im_nodata)

        # no extra image
        im_extra = []
        im_QA = []

    return im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata
