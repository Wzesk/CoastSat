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

# image processing modules
import skimage.filters as filters
import skimage.measure as measure
import skimage.morphology as morphology

# machine learning modules
import sklearn
if sklearn.__version__[:4] == '0.20':
    from sklearn.externals import joblib
else:
    import joblib
from shapely.geometry import LineString

# other modules
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.cm as cm
from matplotlib import gridspec
import pickle
from datetime import datetime, timedelta
from pylab import ginput

np.seterr(all='ignore') # raise/ignore divisions by 0 and nans



# function to preprocess a med res satellite image
def preprocess(fn, satname='S2', 
               band_order =['blue', 'green','red', 'near-infrared','short-wave-infrared-1', 'short-wave-infrared-2','red','red'], 
               cloud_mask_issue=False, 
               pan_off=True, 
               s2cloudless_prob=40):
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
    # S2 images or PLSD images (from skywatch)
    #=============================================================================================#
    if satname == 'S2' or satname == 'PLSD':
        # read 10m bands (R,G,B,NIR)
        if isinstance(fn, list):
            fn_ms=fn[0]
        elif isinstance(fn, str):
            fn_ms=fn

        data = gdal.Open(fn_ms, gdal.GA_ReadOnly)

        bandlist = get_bandlist(fn_ms)

        georef = np.array(data.GetGeoTransform())
        bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]

        # custom band stacks
        im_ms,missing,not_used = stack_bands(bandlist, bands, band_order)
        print('Missing bands:',missing)
        print('unused bands:',not_used)

        # im_ms = np.stack(bands, 2)
        # im_ms = np.append(im_ms, im_ms, axis=2)

        im_ms = im_ms/10000 # TOA scaled to 10000

        #crop array reduce the sice along the third axis
        im_ms = im_ms[:,:,:5]

        # image size
        nrows = im_ms.shape[0]
        ncols = im_ms.shape[1]

        # combine both cloud masks
        cloud_mask = np.zeros((nrows,ncols)).astype('bool')
        im_nodata = np.zeros(cloud_mask.shape).astype('bool')

        # no extra image
        im_extra = []
        im_QA = []

    return im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata

#stack bands in a custom order using the band list to look up each band
#returns a new array with the custom stack of bands, and a list of any missing bands (that were not in the band list)
def stack_bands(bandlist, bands, order):
    bands_out = []
    missing = []
    band_indices = []
    not_used = []
    #for each band name in order, get the index in the band list and ad the band to the output list
    for band_name in order:
        if band_name in bandlist:
            band_indices.append(bandlist.index(band_name))
            bands_out.append(bands[bandlist.index(band_name)])
        else:
            missing.append(band_name)

    #create a list of the bands that were not picked
    for i in range(len(bandlist)):
        if i not in band_indices:
            not_used.append(bandlist[i]) 
    im_out = np.stack(bands_out, 2)

    return im_out, missing, not_used

#get list of bands from metadata file
def get_bandlist(fn_ms):
    #get list of bands from metadata file
    fn_meta = fn_ms.replace('_ms.tif','.txt').replace('ms','meta') 
    bandlist = []
    with open(fn_meta, 'r') as file:
        data = file.readlines()
        bandstr = data[6].split('\t')[1]
        #parse bands into list
        bandlist = bandstr .replace('[','').replace(']','').replace('\'','').replace(' ','').split(',')
    return bandlist

# could use -- does entire set of images
def save_jpg(metadata, settings, use_matplotlib=False):
    """
    Saves a .jpg image for all the images contained in metadata.

    KV WRL 2018

    Arguments:
    -----------
    metadata: dict
        contains all the information about the satellite images that were downloaded
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'cloud_thresh': float
            value between 0 and 1 indicating the maximum cloud fraction in
            the cropped image that is accepted
        'cloud_mask_issue': boolean
            True if there is an issue with the cloud mask and sand pixels
            are erroneously being masked on the images
        's2cloudless_prob': float [0,100)
            threshold to identify cloud pixels in the s2cloudless probability mask
        'use_matplotlib': boolean
            False to save a .jpg and True to save as matplotlib plots

    Returns:
    -----------
    Stores the images as .jpg in a folder named /preprocessed

    """

    sitename = settings['inputs']['sitename']
    cloud_thresh = settings['cloud_thresh']
    s2cloudless_prob = settings['s2cloudless_prob']
    filepath_data = settings['inputs']['filepath']
    collection = settings['inputs']['landsat_collection']
    
    # create subfolder to store the jpg files
    filepath_jpg = os.path.join(filepath_data, sitename, 'jpg_files', 'preprocessed')
    if not os.path.exists(filepath_jpg):
            os.makedirs(filepath_jpg)

    # loop through satellite list
    print('Saving images as jpg:')
    for satname in metadata.keys():
        
        filepath = SDS_tools.get_filepath(settings['inputs'],satname)
        filenames = metadata[satname]['filenames']
        print('%s: %d images'%(satname,len(filenames)))
        # loop through images
        for i in range(len(filenames)):
            print('\r%d%%' %int((i+1)/len(filenames)*100), end='')
            # image filename
            fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
            # read and preprocess image
            im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = preprocess(fn)

            # compute cloud_cover percentage (with no data pixels)
            cloud_cover_combined = np.divide(sum(sum(cloud_mask.astype(int))),
                                    (cloud_mask.shape[0]*cloud_mask.shape[1]))
            

            ##################################
            cloud_cover_combined = 0# manually setting this so it always runs
            ##################################



            if cloud_cover_combined > 0.99: # if 99% of cloudy pixels in image skip
                continue

            # remove no data pixels from the cloud mask (for example L7 bands of no data should not be accounted for)
            cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
            # compute updated cloud cover percentage (without no data pixels)
            cloud_cover = np.divide(sum(sum(cloud_mask_adv.astype(int))),
                                    (sum(sum((~im_nodata).astype(int)))))
            

            ##################################
            cloud_cover = 0# manually setting this so it always runs
            ##################################


            # skip image if cloud cover is above threshold
            if cloud_cover > cloud_thresh or cloud_cover == 1:
                continue
            # save .jpg with date and satellite in the title
            date = filenames[i][:19]
            plt.ioff()  # turning interactive plotting off
            create_jpg(im_ms, cloud_mask, date, satname, filepath_jpg, use_matplotlib)
        print('')
    # print the location where the images have been saved
    print('Satellite images saved as .jpg in ' + os.path.join(filepath_data, sitename,
                                                    'jpg_files', 'preprocessed'))


# could use -- only does one image
def create_jpg(im_ms, cloud_mask, date, satname, filepath, use_matplotlib=True):
    """
    Saves a .jpg file with the RGB image as well as the NIR and SWIR1 grayscale images.
    This functions can be modified to obtain different visualisations of the
    multispectral images.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        3D array containing the pansharpened/down-sampled bands (B,G,R,NIR,SWIR1)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    date: str
        string containing the date at which the image was acquired
    satname: str
        name of the satellite mission (e.g., 'L5')
    filepath: str
        directory in which to save the images
    use_matplotlib: boolean
        False to save a .jpg and True to save as matplotlib plots

    Returns:
    -----------
        Saves a .jpg image corresponding to the preprocessed satellite image

    """
    # rescale image intensity for display purposes
    im_RGB = rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
    im_NIR = rescale_image_intensity(im_ms[:,:,3], cloud_mask, 99.9)
    im_SWIR = rescale_image_intensity(im_ms[:,:,4], cloud_mask, 99.9)
    
    # creates raw jpg files that can be used for ML applications
    if not use_matplotlib:
        # convert images to bytes so they can be saved
        im_RGB = img_as_ubyte(im_RGB)
        im_NIR = img_as_ubyte(im_NIR)
        im_SWIR = img_as_ubyte(im_SWIR)
        # Save each kind of image with skimage.io
        file_types = ["RGB","SWIR","NIR"]
        # create folders RGB, SWIR, and NIR to hold each type of image
        for ext in file_types:
            ext_filepath = filepath + os.sep + ext
            if not os.path.exists(ext_filepath):
                os.mkdir(ext_filepath)
            # location to save image rgb image would be in sitename/RGB/sitename.jpg
            fname=os.path.join(ext_filepath, date + '_'+ ext +'_' + satname + '.jpg')
            if ext == "RGB":
                imsave(fname, im_RGB, quality=100)
            if ext == "SWIR":
                imsave(fname, im_SWIR, quality=100)
            if ext == "NIR":
                imsave(fname, im_NIR, quality=100)
                
    # if use_matplotlib=True, creates a nicer plot
    else:
        fig = plt.figure()
        fig.set_size_inches([18,9])
        fig.set_tight_layout(True)
        ax1 = fig.add_subplot(111)
        ax1.axis('off')
        ax1.imshow(im_RGB)
        ax1.set_title(date + '   ' + satname, fontsize=16)
        
        # choose vertical or horizontal based on image size
        # if im_RGB.shape[1] > 2.5*im_RGB.shape[0]:
        #     ax1 = fig.add_subplot(311)
        #     ax2 = fig.add_subplot(312)
        #     ax3 = fig.add_subplot(313)
        # else:
        #     ax1 = fig.add_subplot(131)
        #     ax2 = fig.add_subplot(132)
        #     ax3 = fig.add_subplot(133)
        # # RGB
        # ax1.axis('off')
        # ax1.imshow(im_RGB)
        # ax1.set_title(date + '   ' + satname, fontsize=16)
        # # NIR
        # ax2.axis('off')
        # ax2.imshow(im_NIR, cmap='seismic')
        # ax2.set_title('Near Infrared', fontsize=16)
        # # SWIR
        # ax3.axis('off')
        # ax3.imshow(im_SWIR, cmap='seismic')
        # ax3.set_title('Short-wave Infrared', fontsize=16)
    
        # save figure
        fig.savefig(os.path.join(filepath, date + '_' + satname + '.jpg'), dpi=150)


def rescale_image_intensity(im, cloud_mask, prob_high):
    """
    Rescales the intensity of an image (multispectral or single band) by applying
    a cloud mask and clipping the prob_high upper percentile. This functions allows
    to stretch the contrast of an image, only for visualisation purposes.

    KV WRL 2018

    Arguments:
    -----------
    im: np.array
        Image to rescale, can be 3D (multispectral) or 2D (single band)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    prob_high: float
        probability of exceedence used to calculate the upper percentile

    Returns:
    -----------
    im_adj: np.array
        rescaled image
    """

    # lower percentile is set to 0
    prc_low = 0

    # reshape the 2D cloud mask into a 1D vector
    vec_mask = cloud_mask.reshape(im.shape[0] * im.shape[1])

    # if image contains several bands, stretch the contrast for each band
    if len(im.shape) > 2:
        # reshape into a vector
        vec =  im.reshape(im.shape[0] * im.shape[1], im.shape[2])
        # initiliase with NaN values
        vec_adj = np.ones((len(vec_mask), im.shape[2])) * np.nan
        # loop through the bands
        for i in range(im.shape[2]):
            # find the higher percentile (based on prob)
            prc_high = np.percentile(vec[~vec_mask, i], prob_high)
            # clip the image around the 2 percentiles and rescale the contrast
            vec_rescaled = exposure.rescale_intensity(vec[~vec_mask, i],
                                                      in_range=(prc_low, prc_high))
            vec_adj[~vec_mask,i] = vec_rescaled
        # reshape into image
        im_adj = vec_adj.reshape(im.shape[0], im.shape[1], im.shape[2])

    # if image only has 1 bands (grayscale image)
    else:
        vec =  im.reshape(im.shape[0] * im.shape[1])
        vec_adj = np.ones(len(vec_mask)) * np.nan
        prc_high = np.percentile(vec[~vec_mask], prob_high)
        vec_rescaled = exposure.rescale_intensity(vec[~vec_mask], in_range=(prc_low, prc_high))
        vec_adj[~vec_mask] = vec_rescaled
        im_adj = vec_adj.reshape(im.shape[0], im.shape[1])

    return im_adj


def get_reference_sl(metadata, settings):
    """
    Allows the user to manually digitize a reference shoreline that is used seed
    the shoreline detection algorithm. The reference shoreline helps to detect
    the outliers, making the shoreline detection more robust.

    KV WRL 2018

    Arguments:
    -----------
    metadata: dict
        contains all the information about the satellite images that were downloaded
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'cloud_thresh': float
            value between 0 and 1 indicating the maximum cloud fraction in
            the cropped image that is accepted
        'cloud_mask_issue': boolean
            True if there is an issue with the cloud mask and sand pixels
            are erroneously being masked on the images
        'output_epsg': int
            output spatial reference system as EPSG code

    Returns:
    -----------
    reference_shoreline: np.array
        coordinates of the reference shoreline that was manually digitized.
        This is also saved as a .pkl and .geojson file.

    """

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    collection = settings['inputs']['landsat_collection']
    pts_coords = []
    # check if reference shoreline already exists in the corresponding folder
    fp_ref_shoreline = os.path.join(filepath_data, sitename, sitename + '_reference_shoreline.geojson')
    # if it exist, load it and load the geojson file
    if os.path.exists(fp_ref_shoreline):
        print('Reference shoreline already exists and was loaded')
        refsl_geojson = gpd.read_file(fp_ref_shoreline,driver='GeoJSON')
        refsl = np.array(refsl_geojson.iloc[0]['geometry'].coords)
        print('Reference shoreline coordinates are in epsg:%d'%refsl_geojson.crs.to_epsg())
        return refsl

    # otherwise get the user to manually digitise a shoreline on 
    # S2, L8, L9 or L5 images (no L7 because of scan line error)
    # first try to use S2 images (10m res for manually digitizing the reference shoreline)
    if 'S2' in metadata.keys(): satname = 'S2'
    # if no S2 images, use L8 or L9 (15m res in the RGB with pansharpening)
    elif 'L8' in metadata.keys(): satname = 'L8'
    elif 'L9' in metadata.keys(): satname = 'L9'
    # if no S2, L8 or L9 use L5 (30m res)
    elif 'L5' in metadata.keys(): satname = 'L5'
    # if only L7 images, ask user to download other images
    else:
        raise Exception('You cannot digitize the shoreline on L7 images (because of gaps in the images), add another L8, S2 or L5 to your dataset.')
    filepath = SDS_tools.get_filepath(settings['inputs'],satname)
    filenames = metadata[satname]['filenames']
    # create figure
    fig, ax = plt.subplots(1,1, figsize=[18,9], tight_layout=True)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    # loop trhough the images
    for i in range(len(filenames)):

        # read image
        fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
        im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = preprocess(fn)

        # compute cloud_cover percentage (with no data pixels)
        cloud_cover_combined = np.divide(sum(sum(cloud_mask.astype(int))),
                                (cloud_mask.shape[0]*cloud_mask.shape[1]))
        if cloud_cover_combined > 0.99: # if 99% of cloudy pixels in image skip
            continue

        # remove no data pixels from the cloud mask (for example L7 bands of no data should not be accounted for)
        cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
        # compute updated cloud cover percentage (without no data pixels)
        cloud_cover = np.divide(sum(sum(cloud_mask_adv.astype(int))),
                                (sum(sum((~im_nodata).astype(int)))))

        # skip image if cloud cover is above threshold
        if cloud_cover > settings['cloud_thresh']:
            continue

        # rescale image intensity for display purposes
        im_RGB = rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)

        # plot the image RGB on a figure
        ax.axis('off')
        ax.imshow(im_RGB)

        # decide if the image if good enough for digitizing the shoreline
        ax.set_title('Press <right arrow> if image is clear enough to digitize the shoreline.\n' +
                  'If the image is cloudy press <left arrow> to get another image.\n'+
                  'Select points clockwise (for closed curve) or in order with the land\n'+
                  'to the right of the shorline for an open curve.', fontsize=14)
        # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
        # this variable needs to be immuatable so we can access it after the keypress event
        skip_image = False
        key_event = {}
        def press(event):
            # store what key was pressed in the dictionary
            key_event['pressed'] = event.key
        # let the user press a key, right arrow to keep the image, left arrow to skip it
        # to break the loop the user can press 'escape'
        while True:
            btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            plt.draw()
            fig.canvas.mpl_connect('key_press_event', press)
            plt.waitforbuttonpress()
            # after button is pressed, remove the buttons
            btn_skip.remove()
            btn_keep.remove()
            btn_esc.remove()
            # keep/skip image according to the pressed key, 'escape' to break the loop
            if key_event.get('pressed') == 'right':
                skip_image = False
                break
            elif key_event.get('pressed') == 'left':
                skip_image = True
                break
            elif key_event.get('pressed') == 'escape':
                plt.close()
                raise StopIteration('User cancelled checking shoreline detection')
            else:
                plt.waitforbuttonpress()

        if skip_image:
            ax.clear()
            continue
        else:
            # create two new buttons
            add_button = plt.text(0, 0.9, 'add', size=16, ha="left", va="top",
                                   transform=plt.gca().transAxes,
                                   bbox=dict(boxstyle="square", ec='k',fc='w'))
            end_button = plt.text(1, 0.9, 'end', size=16, ha="right", va="top",
                                   transform=plt.gca().transAxes,
                                   bbox=dict(boxstyle="square", ec='k',fc='w'))
            # add multiple reference shorelines (until user clicks on <end> button)
            pts_sl = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
            geoms = []
            while 1:
                add_button.set_visible(False)
                end_button.set_visible(False)
                # update title (instructions)
                ax.set_title('Click points along the shoreline (enough points to capture the beach curvature).\n' +
                          'Start at one end of the beach.\n' + 'When finished digitizing, click <ENTER>',
                          fontsize=14)
                plt.draw()

                # let user click on the shoreline
                pts = ginput(n=50000, timeout=-1, show_clicks=True)
                pts_pix = np.array(pts)
                # convert pixel coordinates to world coordinates
                pts_world = SDS_tools.convert_pix2world(pts_pix[:,[1,0]], georef)

                # interpolate between points clicked by the user (1m resolution)
                pts_world_interp = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
                for k in range(len(pts_world)-1):
                    pt_dist = np.linalg.norm(pts_world[k,:]-pts_world[k+1,:])
                    xvals = np.arange(0,pt_dist)
                    yvals = np.zeros(len(xvals))
                    pt_coords = np.zeros((len(xvals),2))
                    pt_coords[:,0] = xvals
                    pt_coords[:,1] = yvals
                    phi = 0
                    deltax = pts_world[k+1,0] - pts_world[k,0]
                    deltay = pts_world[k+1,1] - pts_world[k,1]
                    phi = np.pi/2 - np.math.atan2(deltax, deltay)
                    tf = transform.EuclideanTransform(rotation=phi, translation=pts_world[k,:])
                    pts_world_interp = np.append(pts_world_interp,tf(pt_coords), axis=0)
                pts_world_interp = np.delete(pts_world_interp,0,axis=0)

                # save as geometry (to create .geojson file later)
                geoms.append(geometry.LineString(pts_world_interp))

                # convert to pixel coordinates and plot
                pts_pix_interp = SDS_tools.convert_world2pix(pts_world_interp, georef)
                pts_sl = np.append(pts_sl, pts_world_interp, axis=0)
                ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--')
                ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko')
                ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko')

                # update title and buttons
                add_button.set_visible(True)
                end_button.set_visible(True)
                ax.set_title('click on <add> to digitize another shoreline or on <end> to finish and save the shoreline(s)',
                          fontsize=14)
                plt.draw()

                # let the user click again (<add> another shoreline or <end>)
                pt_input = ginput(n=1, timeout=-1, show_clicks=False)
                pt_input = np.array(pt_input)

                # if user clicks on <end>, save the points and break the loop
                if pt_input[0][0] > im_ms.shape[1]/2:
                    add_button.set_visible(False)
                    end_button.set_visible(False)
                    plt.title('Reference shoreline saved as ' + sitename + '_reference_shoreline.pkl and ' + sitename + '_reference_shoreline.geojson')
                    plt.draw()
                    ginput(n=1, timeout=3, show_clicks=False)
                    plt.close()
                    break

            pts_sl = np.delete(pts_sl,0,axis=0)
            # convert world image coordinates to user-defined coordinate system
            image_epsg = metadata[satname]['epsg'][i]
            pts_coords = SDS_tools.convert_epsg(pts_sl, image_epsg, settings['output_epsg'])

            # save the reference shoreline as .pkl
            filepath = os.path.join(filepath_data, sitename)
            with open(os.path.join(filepath, sitename + '_reference_shoreline.pkl'), 'wb') as f:
                pickle.dump(pts_coords, f)

            # also store as .geojson in case user wants to drag-and-drop on GIS for verification
            for k,line in enumerate(geoms):
                gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(line))
                gdf.index = [k]
                gdf.loc[k,'name'] = 'reference shoreline ' + str(k+1)
                # store into geodataframe
                if k == 0:
                    gdf_all = gdf
                else:
                    gdf_all = pd.concat([gdf_all, gdf])
            gdf_all.crs = CRS(image_epsg)
            # convert from image_epsg to user-defined coordinate system
            gdf_all = gdf_all.to_crs(epsg=settings['output_epsg'])
            # save as geojson
            gdf_all.to_file(os.path.join(filepath, sitename + '_reference_shoreline.geojson'),
                            driver='GeoJSON', encoding='utf-8')

            print('Reference shoreline has been saved in ' + filepath)
            break

    # check if a shoreline was digitised
    if len(pts_coords) == 0:
        raise Exception('No cloud free images are available to digitise the reference shoreline,'+
                        'download more images and try again')
    
    return pts_coords

###################################################################################################
###################################################################################################

# Functions below are imported from SDS_shoreline

###################################################################################################
###################################################################################################

# Main function for batch shoreline detection
def extract_shorelines(metadata, settings):
    """
    Main function to extract shorelines from satellite images

    KV WRL 2018

    Arguments:
    -----------
    metadata: dict
        contains all the information about the satellite images that were downloaded
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'cloud_thresh': float
            value between 0 and 1 indicating the maximum cloud fraction in 
            the cropped image that is accepted
        'cloud_mask_issue': boolean
            True if there is an issue with the cloud mask and sand pixels
            are erroneously being masked on the image
        'min_beach_area': int
            minimum allowable object area (in metres^2) for the class 'sand',
            the area is converted to number of connected pixels
        'min_length_sl': int
            minimum length (in metres) of shoreline contour to be valid
        'sand_color': str
            default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
        'output_epsg': int
            output spatial reference system as EPSG code
        'check_detection': bool
            if True, lets user manually accept/reject the mapped shorelines
        'save_figure': bool
            if True, saves a -jpg file for each mapped shoreline
        'adjust_detection': bool
            if True, allows user to manually adjust the detected shoreline
        'pan_off': bool
            if True, no pan-sharpening is performed on Landsat 7,8 and 9 imagery
            are erroneously being masked on the images
        's2cloudless_prob': float [0,100)
            threshold to identify cloud pixels in the s2cloudless probability mask
            
    Returns:
    -----------
    output: dict
        contains the extracted shorelines and corresponding dates + metadata

    """

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    collection = settings['inputs']['landsat_collection']
    filepath_models = os.path.join(os.getcwd(), 'classification', 'models')
    # initialise output structure
    output = dict([])
    # create a subfolder to store the .jpg images showing the detection
    filepath_jpg = os.path.join(filepath_data, sitename, 'jpg_files', 'detection')
    if not os.path.exists(filepath_jpg):
            os.makedirs(filepath_jpg)
    # close all open figures
    plt.close('all')

    print('Mapping shorelines:')

    default_min_length_sl = settings['min_length_sl']
    # loop through satellite list
    for satname in metadata.keys():

        # get images
        filepath = SDS_tools.get_filepath(settings['inputs'],satname)
        filenames = metadata[satname]['filenames']

        # initialise the output variables
        output_timestamp = []  # datetime at which the image was acquired (UTC time)
        output_shoreline = []  # vector of shoreline points
        output_filename = []   # filename of the images from which the shorelines where derived
        output_cloudcover = [] # cloud cover of the images
        output_geoaccuracy = []# georeferencing accuracy of the images
        output_idxkeep = []    # index that were kept during the analysis (cloudy images are skipped)
        output_t_mndwi = []    # MNDWI threshold used to map the shoreline

        # load classifiers (if sklearn version above 0.20, learn the new files)
        str_new = ''
        if not sklearn.__version__[:4] == '0.20':
            str_new = '_new'
        if satname in ['L5','L7','L8','L9']:
            pixel_size = 15
            if settings['sand_color'] == 'dark':
                clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat_dark%s.pkl'%str_new))
            elif settings['sand_color'] == 'bright':
                clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat_bright%s.pkl'%str_new))
            elif settings['sand_color'] == 'default':
                clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat%s.pkl'%str_new))
            elif settings['sand_color'] == 'latest':
                clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_Landsat_latest%s.pkl'%str_new))   
        elif satname == 'S2':
            pixel_size = 10
            clf = joblib.load(os.path.join(filepath_models, 'NN_4classes_S2%s.pkl'%str_new))

        # convert settings['min_beach_area'] from metres to pixels
        min_beach_area_pixels = np.ceil(settings['min_beach_area']/pixel_size**2)
        # reduce min shoreline length for L7 because of the diagonal bands
        if satname == 'L7': settings['min_length_sl'] = 200
        else: settings['min_length_sl'] = default_min_length_sl
        
        # loop through the images
        for i in range(len(filenames)):

            print('\r%s:   %d%%' % (satname,int(((i+1)/len(filenames))*100)), end='')

            # get image filename
            fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
            # preprocess image (cloud mask + pansharpening/downsampling)
            im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = preprocess(fn)  

            # get image spatial reference system (epsg code) from metadata dict
            image_epsg = metadata[satname]['epsg'][i]
            
            # compute cloud_cover percentage (with no data pixels)
            cloud_cover_combined = np.divide(sum(sum(cloud_mask.astype(int))),
                                    (cloud_mask.shape[0]*cloud_mask.shape[1]))
            if cloud_cover_combined > 0.99: # if 99% of cloudy pixels in image skip
                continue
            # remove no data pixels from the cloud mask 
            # (for example L7 bands of no data should not be accounted for)
            cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata) 
            # compute updated cloud cover percentage (without no data pixels)
            cloud_cover = np.divide(sum(sum(cloud_mask_adv.astype(int))),
                                    (sum(sum((~im_nodata).astype(int)))))
            # skip image if cloud cover is above user-defined threshold
            if cloud_cover > settings['cloud_thresh']:
                continue

            # calculate a buffer around the reference shoreline (if any has been digitised)
            im_ref_buffer = create_shoreline_buffer(cloud_mask.shape, georef, image_epsg,
                                                    pixel_size, settings)

            # classify image in 4 classes (sand, whitewater, water, other) with NN classifier
            im_classif, im_labels = classify_image_NN(im_ms, cloud_mask, min_beach_area_pixels, clf)
            
            # if adjust_detection is True, let the user adjust the detected shoreline
            if settings['adjust_detection']:
                date = filenames[i][:19]
                skip_image, shoreline, t_mndwi = adjust_detection(im_ms, cloud_mask, im_nodata, im_labels,
                                                                  im_ref_buffer, image_epsg, georef,
                                                                  settings, date, satname)
                # if the user decides to skip the image, continue and do not save the mapped shoreline
                if skip_image:
                    continue
                
            # otherwise map the contours automatically with one of the two following functions:
            # if there are pixels in the 'sand' class --> use find_wl_contours2 (enhanced)
            # otherwise use find_wl_contours1 (traditional)
            else:
                try: # use try/except structure for long runs
                    if sum(im_labels[im_ref_buffer,0]) < 50: # minimum number of sand pixels
                        # compute MNDWI image (SWIR-G)
                        im_mndwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
                        # find water contours on MNDWI grayscale image
                        contours_mwi, t_mndwi = find_wl_contours1(im_mndwi, cloud_mask, im_ref_buffer)
                    else:
                        # use classification to refine threshold and extract the sand/water interface
                        contours_mwi, t_mndwi = find_wl_contours2(im_ms, im_labels, cloud_mask, im_ref_buffer)
                except:
                    print('Could not map shoreline for this image: ' + filenames[i])
                    continue
    
                # process the water contours into a shoreline
                shoreline = process_shoreline(contours_mwi, cloud_mask_adv, im_nodata,
                                              georef, image_epsg, settings)
    
                # visualise the mapped shorelines, there are two options:
                # if settings['check_detection'] = True, shows the detection to the user for accept/reject
                # if settings['save_figure'] = True, saves a figure for each mapped shoreline
                if settings['check_detection'] or settings['save_figure']:
                    date = filenames[i][:19]
                    if not settings['check_detection']:
                        plt.ioff() # turning interactive plotting off
                    skip_image = show_detection(im_ms, cloud_mask, im_labels, shoreline,
                                                image_epsg, georef, settings, date, satname)
                    # if the user decides to skip the image, continue and do not save the mapped shoreline
                    if skip_image:
                        continue

            # append to output variables
            output_timestamp.append(metadata[satname]['dates'][i])
            output_shoreline.append(shoreline)
            output_filename.append(filenames[i])
            output_cloudcover.append(cloud_cover)
            output_geoaccuracy.append(metadata[satname]['acc_georef'][i])
            output_idxkeep.append(i)
            output_t_mndwi.append(t_mndwi)

        # create dictionnary of output
        output[satname] = {
                'dates': output_timestamp,
                'shorelines': output_shoreline,
                'filename': output_filename,
                'cloud_cover': output_cloudcover,
                'geoaccuracy': output_geoaccuracy,
                'idx': output_idxkeep,
                'MNDWI_threshold': output_t_mndwi,
                }
        print('')

    # close figure window if still open
    if plt.get_fignums():
        plt.close()

    # change the format to have one list sorted by date with all the shorelines (easier to use)
    output = SDS_tools.merge_output(output)

    # save outputput structure as output.pkl
    filepath = os.path.join(filepath_data, sitename)
    with open(os.path.join(filepath, sitename + '_output.pkl'), 'wb') as f:
        pickle.dump(output, f)

    return output

###################################################################################################
# IMAGE CLASSIFICATION FUNCTIONS
###################################################################################################

def calculate_features(im_ms, cloud_mask, im_bool):
    """
    Calculates features on the image that are used for the supervised classification. 
    The features include spectral normalized-difference indices and standard 
    deviation of the image for all the bands and indices.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_bool: np.array
        2D array of boolean indicating where on the image to calculate the features

    Returns:    
    -----------
    features: np.array
        matrix containing each feature (columns) calculated for all
        the pixels (rows) indicated in im_bool
        
    """

    # add all the multispectral bands
    features = np.expand_dims(im_ms[im_bool,0],axis=1)
    for k in range(1,im_ms.shape[2]):
        feature = np.expand_dims(im_ms[im_bool,k],axis=1)
        features = np.append(features, feature, axis=-1)
    # NIR-G
    im_NIRG = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRG[im_bool],axis=1), axis=-1)
    # SWIR-G
    im_SWIRG = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_SWIRG[im_bool],axis=1), axis=-1)
    # NIR-R
    im_NIRR = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRR[im_bool],axis=1), axis=-1)
    # SWIR-NIR
    im_SWIRNIR = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,3], cloud_mask)
    features = np.append(features, np.expand_dims(im_SWIRNIR[im_bool],axis=1), axis=-1)
    # B-R
    im_BR = SDS_tools.nd_index(im_ms[:,:,0], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_BR[im_bool],axis=1), axis=-1)
    # calculate standard deviation of individual bands
    for k in range(im_ms.shape[2]):
        im_std =  SDS_tools.image_std(im_ms[:,:,k], 1)
        features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    # calculate standard deviation of the spectral indices
    im_std = SDS_tools.image_std(im_NIRG, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_SWIRG, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_NIRR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_SWIRNIR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_BR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)

    return features

def classify_image_NN(im_ms, cloud_mask, min_beach_area, clf):
    """
    Classifies every pixel in the image in one of 4 classes:
        - sand                                          --> label = 1
        - whitewater (breaking waves and swash)         --> label = 2
        - water                                         --> label = 3
        - other (vegetation, buildings, rocks...)       --> label = 0

    The classifier is a Neural Network that is already trained.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        Pansharpened RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    min_beach_area: int
        minimum number of pixels that have to be connected to belong to the SAND class
    clf: joblib object
        pre-trained classifier

    Returns:    
    -----------
    im_classif: np.array
        2D image containing labels
    im_labels: np.array of booleans
        3D image containing a boolean image for each class (im_classif == label)

    """

    # calculate features
    vec_features = calculate_features(im_ms, cloud_mask, np.ones(cloud_mask.shape).astype(bool))
    vec_features[np.isnan(vec_features)] = 1e-9 # NaN values are create when std is too close to 0

    # remove NaNs and cloudy pixels
    vec_cloud = cloud_mask.reshape(cloud_mask.shape[0]*cloud_mask.shape[1])
    vec_nan = np.any(np.isnan(vec_features), axis=1)
    vec_inf = np.any(np.isinf(vec_features), axis=1)    
    vec_mask = np.logical_or(vec_cloud,np.logical_or(vec_nan,vec_inf))
    vec_features = vec_features[~vec_mask, :]

    # classify pixels
    labels = clf.predict(vec_features)

# ###########################################################
#     #reclassify white water and other as sand
#     labels[labels == 2] = 1
#     labels[labels == 0] = 1
# ###########################################################
    # recompose image
    vec_classif = np.nan*np.ones((cloud_mask.shape[0]*cloud_mask.shape[1]))
    vec_classif[~vec_mask] = labels
    im_classif = vec_classif.reshape((cloud_mask.shape[0], cloud_mask.shape[1]))

    # create a stack of boolean images for each label
    im_sand = im_classif == 1
    im_swash = im_classif == 2
    im_water = im_classif == 3
    # remove small patches of sand or water that could be around the image (usually noise)
    im_sand = morphology.remove_small_objects(im_sand, min_size=min_beach_area, connectivity=2)
    im_water = morphology.remove_small_objects(im_water, min_size=min_beach_area, connectivity=2)

    im_labels = np.stack((im_sand,im_swash,im_water), axis=-1)

    return im_classif, im_labels

###################################################################################################
# CONTOUR MAPPING FUNCTIONS
###################################################################################################

def find_wl_contours1(im_ndwi, cloud_mask, im_ref_buffer):
    """
    Traditional method for shoreline detection using a global threshold.
    Finds the water line by thresholding the Normalized Difference Water Index 
    and applying the Marching Squares Algorithm to contour the iso-value 
    corresponding to the threshold.

    KV WRL 2018

    Arguments:
    -----------
    im_ndwi: np.ndarray
        Image (2D) with the NDWI (water index)
    cloud_mask: np.ndarray
        2D cloud mask with True where cloud pixels are
    im_ref_buffer: np.array
        Binary image containing a buffer around the reference shoreline

    Returns:    
    -----------
    contours: list of np.arrays
        contains the coordinates of the contour lines
    t_mwi: float
        Otsu threshold used to map the contours

    """
    nrows = cloud_mask.shape[0]
    ncols = cloud_mask.shape[1]
    # use im_ref_buffer and dilate it by 5 pixels
    se = morphology.disk(5)
    im_ref_buffer_extra = morphology.binary_dilation(im_ref_buffer,se)
    vec_buffer = im_ref_buffer_extra.reshape(nrows*ncols)
    # reshape spectral index image to vector
    vec_ndwi = im_ndwi.reshape(nrows*ncols)
    # keep pixels that are in the buffer and not in the cloud mask
    vec_mask = cloud_mask.reshape(nrows*ncols)
    vec = vec_ndwi[np.logical_and(vec_buffer,~vec_mask)]
    # apply otsu's threshold
    vec = vec[~np.isnan(vec)]
    t_otsu = filters.threshold_otsu(vec)
    # use Marching Squares algorithm to detect contours on ndwi image
    im_ndwi_buffer = np.copy(im_ndwi)
    im_ndwi_buffer[~im_ref_buffer] = np.nan
    contours = measure.find_contours(im_ndwi_buffer, t_otsu)
    # remove contours that contain NaNs (due to cloud pixels in the contour)
    contours = process_contours(contours)

    return contours, t_otsu

def find_wl_contours2(im_ms, im_labels, cloud_mask, im_ref_buffer):
    """
    New robust method for extracting shorelines. Incorporates the classification
    component to refine the treshold and make it specific to the sand/water interface.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    im_labels: np.array
        3D image containing a boolean image for each class in the order (sand, swash, water)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_ref_buffer: np.array
        binary image containing a buffer around the reference shoreline

    Returns:    
    -----------
    contours_mwi: list of np.arrays
        contains the coordinates of the contour lines extracted from the
        MNDWI (Modified Normalized Difference Water Index) image
    t_mwi: float
        Otsu sand/water threshold used to map the contours

    """

    nrows = cloud_mask.shape[0]
    ncols = cloud_mask.shape[1]

    # calculate Normalized Difference Modified Water Index (SWIR - G)
    im_mwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
    # calculate Normalized Difference Modified Water Index (NIR - G)
    im_wi = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    # stack indices together
    im_ind = np.stack((im_wi, im_mwi), axis=-1)
    vec_ind = im_ind.reshape(nrows*ncols,2)

    # reshape labels into vectors
    vec_sand = im_labels[:,:,0].reshape(ncols*nrows)
    vec_water = im_labels[:,:,2].reshape(ncols*nrows)

    # use im_ref_buffer and dilate it by 5 pixels
    se = morphology.disk(5)
    im_ref_buffer_extra = morphology.binary_dilation(im_ref_buffer, se)
    # create a buffer around the sandy beach
    vec_buffer = im_ref_buffer_extra.reshape(nrows*ncols)
    
    # select water/sand pixels that are within the buffer
    int_water = vec_ind[np.logical_and(vec_buffer,vec_water),:]
    int_sand = vec_ind[np.logical_and(vec_buffer,vec_sand),:]

    # make sure both classes have the same number of pixels before thresholding
    if len(int_water) > 0 and len(int_sand) > 0:
        if np.argmin([int_sand.shape[0],int_water.shape[0]]) == 1:
            int_sand = int_sand[np.random.choice(int_sand.shape[0],int_water.shape[0], replace=False),:]
        else:
            int_water = int_water[np.random.choice(int_water.shape[0],int_sand.shape[0], replace=False),:]

    # threshold the sand/water intensities
    int_all = np.append(int_water,int_sand, axis=0)
    t_mwi = filters.threshold_otsu(int_all[:,0])
    t_wi = filters.threshold_otsu(int_all[:,1])

    # find contour with Marching-Squares algorithm
    im_wi_buffer = np.copy(im_wi)
    im_wi_buffer[~im_ref_buffer] = np.nan
    im_mwi_buffer = np.copy(im_mwi)
    im_mwi_buffer[~im_ref_buffer] = np.nan
    contours_wi = measure.find_contours(im_wi_buffer, t_wi)
    contours_mwi = measure.find_contours(im_mwi_buffer, t_mwi)
    # remove contour points that are NaNs (around clouds)
    contours_wi = process_contours(contours_wi)
    contours_mwi = process_contours(contours_mwi)

    # only return MNDWI contours and threshold
    return contours_mwi, t_mwi

###################################################################################################
# SHORELINE PROCESSING FUNCTIONS
###################################################################################################

def create_shoreline_buffer(im_shape, georef, image_epsg, pixel_size, settings):
    """
    Creates a buffer around the reference shoreline. The size of the buffer is 
    given by settings['max_dist_ref'].

    KV WRL 2018

    Arguments:
    -----------
    im_shape: np.array
        size of the image (rows,columns)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    image_epsg: int
        spatial reference system of the image from which the contours were extracted
    pixel_size: int
        size of the pixel in metres (15 for Landsat, 10 for Sentinel-2)
    settings: dict with the following keys
        'output_epsg': int
            output spatial reference system
        'reference_shoreline': np.array
            coordinates of the reference shoreline
        'max_dist_ref': int
            maximum distance from the reference shoreline in metres

    Returns:    
    -----------
    im_buffer: np.array
        binary image, True where the buffer is, False otherwise

    """
    # initialise the image buffer
    im_buffer = np.ones(im_shape).astype(bool)

    if 'reference_shoreline' in settings.keys():

        # convert reference shoreline to pixel coordinates
        ref_sl = settings['reference_shoreline']
        ref_sl_conv = SDS_tools.convert_epsg(ref_sl, settings['output_epsg'],image_epsg)
        ref_sl_pix = SDS_tools.convert_world2pix(ref_sl_conv, georef)
        ref_sl_pix_rounded = np.round(ref_sl_pix).astype(int)

        # make sure that the pixel coordinates of the reference shoreline are inside the image
        idx_row = np.logical_and(ref_sl_pix_rounded[:,0] > 0, ref_sl_pix_rounded[:,0] < im_shape[1])
        idx_col = np.logical_and(ref_sl_pix_rounded[:,1] > 0, ref_sl_pix_rounded[:,1] < im_shape[0])
        idx_inside = np.logical_and(idx_row, idx_col)
        ref_sl_pix_rounded = ref_sl_pix_rounded[idx_inside,:]

        # create binary image of the reference shoreline (1 where the shoreline is 0 otherwise)
        im_binary = np.zeros(im_shape)
        for j in range(len(ref_sl_pix_rounded)):
            im_binary[ref_sl_pix_rounded[j,1], ref_sl_pix_rounded[j,0]] = 1
        im_binary = im_binary.astype(bool)

        # dilate the binary image to create a buffer around the reference shoreline
        max_dist_ref_pixels = np.ceil(settings['max_dist_ref']/pixel_size)
        se = morphology.disk(max_dist_ref_pixels)
        im_buffer = morphology.binary_dilation(im_binary, se)

    return im_buffer

def process_contours(contours):
    """
    Remove contours that contain NaNs, usually these are contours that are in contact 
    with clouds.
    
    KV WRL 2020
    
    Arguments:
    -----------
    contours: list of np.array
        image contours as detected by the function skimage.measure.find_contours    
    
    Returns:
    -----------
    contours: list of np.array
        processed image contours (only the ones that do not contains NaNs) 
        
    """
    
    # initialise variable
    contours_nonans = []
    # loop through contours and only keep the ones without NaNs
    for k in range(len(contours)):
        if np.any(np.isnan(contours[k])):
            index_nan = np.where(np.isnan(contours[k]))[0]
            contours_temp = np.delete(contours[k], index_nan, axis=0)
            if len(contours_temp) > 1:
                contours_nonans.append(contours_temp)
        else:
            contours_nonans.append(contours[k])
    
    return contours_nonans
    
def process_shoreline(contours, cloud_mask, im_nodata, georef, image_epsg, settings):
    """
    Converts the contours from image coordinates to world coordinates. This function also removes the contours that:
        1. are too small to be a shoreline (based on the parameter settings['min_length_sl'])
        2. are too close to cloud pixels (based on the parameter settings['dist_clouds'])
        3. are adjacent to noData pixels
    
    KV WRL 2018

    Arguments:
    -----------
        contours: np.array or list of np.array
            image contours as detected by the function find_contours
        cloud_mask: np.array
            2D cloud mask with True where cloud pixels are
        im_nodata: np.array
            2D mask with True where noData pixels are
        georef: np.array
            vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
        image_epsg: int
            spatial reference system of the image from which the contours were extracted
        settings: dict
            contains the following fields:
        output_epsg: int
            output spatial reference system
        min_length_sl: float
            minimum length of shoreline perimeter to be kept (in meters)
        dist_clouds: int
            distance in metres defining a buffer around cloudy pixels where the shoreline cannot be mapped

    Returns:    
    -----------
        shoreline: np.array
            array of points with the X and Y coordinates of the shoreline

    """

    # convert pixel coordinates to world coordinates
    contours_world = SDS_tools.convert_pix2world(contours, georef)
    # convert world coordinates to desired spatial reference system
    contours_epsg = SDS_tools.convert_epsg(contours_world, image_epsg, settings['output_epsg'])
    
    # 1. Remove contours that have a perimeter < min_length_sl (provided in settings dict)
    # this enables to remove the very small contours that do not correspond to the shoreline
    contours_long = []
    for l, wl in enumerate(contours_epsg):
        coords = [(wl[k,0], wl[k,1]) for k in range(len(wl))]
        a = LineString(coords) # shapely LineString structure
        if a.length >= settings['min_length_sl']:
            contours_long.append(wl)
    # format points into np.array
    x_points = np.array([])
    y_points = np.array([])
    for k in range(len(contours_long)):
        x_points = np.append(x_points,contours_long[k][:,0])
        y_points = np.append(y_points,contours_long[k][:,1])
    contours_array = np.transpose(np.array([x_points,y_points]))
    
    shoreline = contours_array
    
    # 2. Remove any shoreline points that are close to cloud pixels (effect of shadows)
    if sum(sum(cloud_mask)) > 0:
        # get the coordinates of the cloud pixels
        idx_cloud = np.where(cloud_mask)
        idx_cloud = np.array([(idx_cloud[0][k], idx_cloud[1][k]) for k in range(len(idx_cloud[0]))])
        # convert to world coordinates and same epsg as the shoreline points
        coords_cloud = SDS_tools.convert_epsg(SDS_tools.convert_pix2world(idx_cloud, georef),
                                               image_epsg, settings['output_epsg'])
        # only keep the shoreline points that are at least 30m from any cloud pixel
        idx_keep = np.ones(len(shoreline)).astype(bool)
        for k in range(len(shoreline)):
            if np.any(np.linalg.norm(shoreline[k,:] - coords_cloud, axis=1) < settings['dist_clouds']):
                idx_keep[k] = False     
        shoreline = shoreline[idx_keep] 
        
    # 3. Remove any shoreline points that are attached to nodata pixels
    if sum(sum(im_nodata)) > 0:
        # get the coordinates of the cloud pixels
        idx_cloud = np.where(im_nodata)
        idx_cloud = np.array([(idx_cloud[0][k], idx_cloud[1][k]) for k in range(len(idx_cloud[0]))])
        # convert to world coordinates and same epsg as the shoreline points
        coords_cloud = SDS_tools.convert_epsg(SDS_tools.convert_pix2world(idx_cloud, georef),
                                               image_epsg, settings['output_epsg'])
        # only keep the shoreline points that are at least 30m from any nodata pixel
        idx_keep = np.ones(len(shoreline)).astype(bool)
        for k in range(len(shoreline)):
            if np.any(np.linalg.norm(shoreline[k,:] - coords_cloud, axis=1) < 30):
                idx_keep[k] = False     
        shoreline = shoreline[idx_keep] 

    return shoreline

###################################################################################################
# INTERACTIVE/PLOTTING FUNCTIONS
###################################################################################################

def show_detection(im_ms, cloud_mask, im_labels, shoreline,image_epsg, georef,
                   settings, date, satname):
    """
    Shows the detected shoreline to the user for visual quality control. 
    The user can accept/reject the detected shorelines  by using keep/skip
    buttons.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_labels: np.array
        3D image containing a boolean image for each class in the order (sand, swash, water)
    shoreline: np.array
        array of points with the X and Y coordinates of the shoreline
    image_epsg: int
        spatial reference system of the image from which the contours were extracted
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    date: string
        date at which the image was taken
    satname: string
        indicates the satname (L5,L7,L8 or S2)
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'output_epsg': int
            output spatial reference system as EPSG code
        'check_detection': bool
            if True, lets user manually accept/reject the mapped shorelines
        'save_figure': bool
            if True, saves a -jpg file for each mapped shoreline

    Returns:
    -----------
    skip_image: boolean
        True if the user wants to skip the image, False otherwise

    """

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    # subfolder where the .jpg file is stored if the user accepts the shoreline detection
    filepath = os.path.join(filepath_data, sitename, 'jpg_files', 'detection')

    im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)

    # compute classified image
    im_class = np.copy(im_RGB)
    cmap = cm.get_cmap('tab20c')
    colorpalette = cmap(np.arange(0,13,1))
    colours = np.zeros((3,4))
    colours[0,:] = colorpalette[5]
    colours[1,:] = np.array([204/255,1,1,1])
    colours[2,:] = np.array([0,91/255,1,1])
    for k in range(0,im_labels.shape[2]):
        im_class[im_labels[:,:,k],0] = colours[k,0]
        im_class[im_labels[:,:,k],1] = colours[k,1]
        im_class[im_labels[:,:,k],2] = colours[k,2]

    # compute MNDWI grayscale image
    im_mwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)

    # transform world coordinates of shoreline into pixel coordinates
    # use try/except in case there are no coordinates to be transformed (shoreline = [])
    try:
        sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                    settings['output_epsg'],
                                                                    image_epsg), georef)
    except:
        # if try fails, just add nan into the shoreline vector so the next parts can still run
        sl_pix = np.array([[np.nan, np.nan],[np.nan, np.nan]])

    if plt.get_fignums():
        # get open figure if it exists
        fig = plt.gcf()
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
        ax3 = fig.axes[2]
        ax4 = fig.axes[3]     

    else:
        # else create a new figure
        fig = plt.figure()
        fig.set_size_inches([18, 9])
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()

        # according to the image shape, decide whether it is better to have the images
        # in vertical subplots or horizontal subplots
        if im_RGB.shape[1] > 2.5*im_RGB.shape[0]:
            # horizontal subplots
            gs = gridspec.GridSpec(4, 1,height_ratios=[0.1,1,1,1])
            gs.update(bottom=0.03, top=0.97, left=0.03, right=0.97)
            ax1 = fig.add_subplot(gs[1])
            ax2 = fig.add_subplot(gs[2], sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(gs[3], sharex=ax1, sharey=ax1)
            ax4 = fig.add_subplot(gs[0])
        else:
            # vertical subplots
            gs = gridspec.GridSpec(2, 3,height_ratios=[1,8],hspace=0.1)
            gs.update(bottom=0.03, top=0.99, left=0.03, right=0.97)
            ax1 = fig.add_subplot(gs[1,0])
            ax2 = fig.add_subplot(gs[1,1], sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(gs[1,2], sharex=ax1, sharey=ax1)
            ax4 = fig.add_subplot(gs[0,:])
            
    # add timeline on top
    date_start = datetime.strptime(settings['inputs']['dates'][0],'%Y-%m-%d')
    date_end = datetime.strptime(settings['inputs']['dates'][1],'%Y-%m-%d')
    ax4.axis('off')
    ax4.axhline(y=0,ls='-',lw=2,c='k')
    ax4.set(xlim=[date_start-timedelta(days=30),date_end+timedelta(days=30)],ylim=[-0.1,0.1])
    for k in range(date_start.year,date_end.year+1):
        ax4.plot(datetime(k,1,1),0,'ko',ms=6)
        ax4.text(datetime(k,1,1),-0.05,str(k)[-2:],ha='center',va='center')
    ax4.plot(datetime.strptime(date[:10],'%Y-%m-%d'),0,'rs',ms=10,mec='k')
    # change the color of nans to either black (0.0) or white (1.0) or somewhere in between
    nan_color = 1.0
    im_RGB = np.where(np.isnan(im_RGB), nan_color, im_RGB)
    im_class = np.where(np.isnan(im_class), 1.0, im_class)

    # create image 1 (RGB)
    ax1.imshow(im_RGB)
    ax1.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax1.axis('off')
    ax1.set_title(sitename, fontweight='bold', fontsize=12)

    # create image 2 (classification)
    ax2.imshow(im_class)
    ax2.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax2.axis('off')
    orange_patch = mpatches.Patch(color=colours[0,:], label='sand')
    white_patch = mpatches.Patch(color=colours[1,:], label='whitewater')
    blue_patch = mpatches.Patch(color=colours[2,:], label='water')
    black_line = mlines.Line2D([],[],color='k',linestyle='-', label='shoreline')
    ax2.legend(handles=[orange_patch,white_patch,blue_patch, black_line],
               bbox_to_anchor=(1, 0.5), fontsize=10)
    ax2.set_title(date, fontweight='bold', fontsize=12)

    # create image 3 (MNDWI)
    ax3.imshow(im_mwi, cmap='bwr')
    ax3.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    ax3.axis('off')
    ax3.set_title(satname, fontweight='bold', fontsize=12)
    
    # additional options
    ax1.set_anchor('W')
    ax2.set_anchor('C')
    ax3.set_anchor('E')
    
    # add colorbar for MNDWI
    # cb = plt.colorbar(mwi_plot)
    # cb.ax.tick_params(labelsize=10)
    # cb.set_label('MNDWI values')
        
    # if check_detection is True, let user manually accept/reject the images
    skip_image = False
    if settings['check_detection']:

        # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
        # this variable needs to be immuatable so we can access it after the keypress event
        key_event = {}
        def press(event):
            # store what key was pressed in the dictionary
            key_event['pressed'] = event.key
        # let the user press a key, right arrow to keep the image, left arrow to skip it
        # to break the loop the user can press 'escape'
        while True:
            btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                transform=ax1.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            plt.draw()
            fig.canvas.mpl_connect('key_press_event', press)
            plt.waitforbuttonpress()
            # after button is pressed, remove the buttons
            btn_skip.remove()
            btn_keep.remove()
            btn_esc.remove()

            # keep/skip image according to the pressed key, 'escape' to break the loop
            if key_event.get('pressed') == 'right':
                skip_image = False
                break
            elif key_event.get('pressed') == 'left':
                skip_image = True
                break
            elif key_event.get('pressed') == 'escape':
                plt.close()
                raise StopIteration('User cancelled checking shoreline detection')
            else:
                plt.waitforbuttonpress()

    # if save_figure is True, save a .jpg under /jpg_files/detection
    if settings['save_figure'] and not skip_image:
        fig.savefig(os.path.join(filepath, date + '_' + satname + '.jpg'), dpi=150)

    # don't close the figure window, but remove all axes and settings, ready for next plot
    for ax in fig.axes:
        ax.clear()

    return skip_image

def adjust_detection(im_ms, cloud_mask, im_nodata, im_labels, im_ref_buffer, image_epsg, georef,
                     settings, date, satname):
    """
    Advanced version of show detection where the user can adjust the detected 
    shorelines with a slide bar.

    KV WRL 2020

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_labels: np.array
        3D image containing a boolean image for each class in the order (sand, swash, water)
    im_ref_buffer: np.array
        Binary image containing a buffer around the reference shoreline
    image_epsg: int
        spatial reference system of the image from which the contours were extracted
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    date: string
        date at which the image was taken
    satname: string
        indicates the satname (L5,L7,L8 or S2)
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'output_epsg': int
            output spatial reference system as EPSG code
        'save_figure': bool
            if True, saves a -jpg file for each mapped shoreline

    Returns:
    -----------
    skip_image: boolean
        True if the user wants to skip the image, False otherwise
    shoreline: np.array
        array of points with the X and Y coordinates of the shoreline 
    t_mndwi: float
        value of the MNDWI threshold used to map the shoreline

    """

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    # subfolder where the .jpg file is stored if the user accepts the shoreline detection
    filepath = os.path.join(filepath_data, sitename, 'jpg_files', 'detection')
    # format date
    date_str = datetime.strptime(date,'%Y-%m-%d-%H-%M-%S').strftime('%Y-%m-%d  %H:%M:%S')
    im_RGB = SDS_preprocess.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)

    # compute classified image
    im_class = np.copy(im_RGB)
    cmap = cm.get_cmap('tab20c')
    colorpalette = cmap(np.arange(0,13,1))
    colours = np.zeros((3,4))
    colours[0,:] = colorpalette[5]
    colours[1,:] = np.array([204/255,1,1,1])
    colours[2,:] = np.array([0,91/255,1,1])
    for k in range(0,im_labels.shape[2]):
        im_class[im_labels[:,:,k],0] = colours[k,0]
        im_class[im_labels[:,:,k],1] = colours[k,1]
        im_class[im_labels[:,:,k],2] = colours[k,2]

    # compute MNDWI grayscale image
    im_mndwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
    # buffer MNDWI using reference shoreline
    im_mndwi_buffer = np.copy(im_mndwi)
    im_mndwi_buffer[~im_ref_buffer] = np.nan

    # get MNDWI pixel intensity in each class (for histogram plot)
    int_sand = im_mndwi[im_labels[:,:,0]]
    int_ww = im_mndwi[im_labels[:,:,1]]
    int_water = im_mndwi[im_labels[:,:,2]]
    labels_other = np.logical_and(np.logical_and(~im_labels[:,:,0],~im_labels[:,:,1]),~im_labels[:,:,2])
    int_other = im_mndwi[labels_other]
    
    # create figure
    if plt.get_fignums():
            # if it exists, open the figure 
            fig = plt.gcf()
            ax1 = fig.axes[0]
            ax2 = fig.axes[1]
            ax3 = fig.axes[2]
            ax4 = fig.axes[3]      
    else:
        # else create a new figure
        fig = plt.figure()
        fig.set_size_inches([18, 9])
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        gs = gridspec.GridSpec(2, 3, height_ratios=[4,1])
        gs.update(bottom=0.05, top=0.95, left=0.03, right=0.97)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1], sharex=ax1, sharey=ax1)
        ax3 = fig.add_subplot(gs[0,2], sharex=ax1, sharey=ax1)
        ax4 = fig.add_subplot(gs[1,:])
    ##########################################################################
    # to do: rotate image if too wide
    ##########################################################################

    # change the color of nans to either black (0.0) or white (1.0) or somewhere in between
    nan_color = 1.0
    im_RGB = np.where(np.isnan(im_RGB), nan_color, im_RGB)
    im_class = np.where(np.isnan(im_class), 1.0, im_class)

    # plot image 1 (RGB)
    ax1.imshow(im_RGB)
    ax1.axis('off')
    ax1.set_title('%s - %s'%(sitename, satname), fontsize=12)

    # plot image 2 (classification)
    ax2.imshow(im_class)
    ax2.axis('off')
    orange_patch = mpatches.Patch(color=colours[0,:], label='sand')
    white_patch = mpatches.Patch(color=colours[1,:], label='whitewater')
    blue_patch = mpatches.Patch(color=colours[2,:], label='water')
    black_line = mlines.Line2D([],[],color='k',linestyle='-', label='shoreline')
    ax2.legend(handles=[orange_patch,white_patch,blue_patch, black_line],
               bbox_to_anchor=(1.1, 0.5), fontsize=10)
    ax2.set_title(date_str, fontsize=12)

    # plot image 3 (MNDWI)
    ax3.imshow(im_mndwi, cmap='bwr')
    ax3.axis('off')
    ax3.set_title('MNDWI', fontsize=12)
    
    # plot histogram of MNDWI values
    binwidth = 0.01
    ax4.set_facecolor('0.75')
    ax4.yaxis.grid(color='w', linestyle='--', linewidth=0.5)
    ax4.set(ylabel='PDF',yticklabels=[], xlim=[-1,1])
    if len(int_sand) > 0 and sum(~np.isnan(int_sand)) > 0:
        bins = np.arange(np.nanmin(int_sand), np.nanmax(int_sand) + binwidth, binwidth)
        ax4.hist(int_sand, bins=bins, density=True, color=colours[0,:], label='sand')
    if len(int_ww) > 0 and sum(~np.isnan(int_ww)) > 0:
        bins = np.arange(np.nanmin(int_ww), np.nanmax(int_ww) + binwidth, binwidth)
        ax4.hist(int_ww, bins=bins, density=True, color=colours[1,:], label='whitewater', alpha=0.75) 
    if len(int_water) > 0 and sum(~np.isnan(int_water)) > 0:
        bins = np.arange(np.nanmin(int_water), np.nanmax(int_water) + binwidth, binwidth)
        ax4.hist(int_water, bins=bins, density=True, color=colours[2,:], label='water', alpha=0.75) 
    if len(int_other) > 0 and sum(~np.isnan(int_other)) > 0:
        bins = np.arange(np.nanmin(int_other), np.nanmax(int_other) + binwidth, binwidth)
        ax4.hist(int_other, bins=bins, density=True, color='C4', label='other', alpha=0.5) 
    
    # automatically map the shoreline based on the classifier if enough sand pixels
    try:
        if sum(sum(im_labels[:,:,0])) > 50:
            # use classification to refine threshold and extract the sand/water interface
            contours_mndwi, t_mndwi = find_wl_contours2(im_ms, im_labels, cloud_mask, im_ref_buffer)
        else:       
            # find water contours on MNDWI grayscale image
            contours_mndwi, t_mndwi = find_wl_contours1(im_mndwi, cloud_mask, im_ref_buffer)    
    except:
        print('Could not map shoreline so image was skipped')
        # clear axes and return skip_image=True, so that image is skipped above
        for ax in fig.axes:
            ax.clear()
        return True,[],[]

    # process the water contours into a shoreline
    cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata) 
    shoreline = process_shoreline(contours_mndwi, cloud_mask_adv, im_nodata, georef, image_epsg, settings)
    # convert shoreline to pixels
    if len(shoreline) > 0:
        sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                    settings['output_epsg'],
                                                                    image_epsg), georef)
    else: sl_pix = np.array([[np.nan, np.nan],[np.nan, np.nan]])
    # plot the shoreline on the images
    sl_plot1 = ax1.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    sl_plot2 = ax2.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    sl_plot3 = ax3.plot(sl_pix[:,0], sl_pix[:,1], 'k.', markersize=3)
    t_line = ax4.axvline(x=t_mndwi,ls='--', c='k', lw=1.5, label='threshold')
    ax4.legend(loc=1)
    plt.draw() # to update the plot
    # adjust the threshold manually by letting the user change the threshold
    ax4.set_title('Click on the plot below to change the location of the threhsold and adjust the shoreline detection. When finished, press <Enter>')
    while True:  
        # let the user click on the threshold plot
        pt = ginput(n=1, show_clicks=True, timeout=-1)
        # if a point was clicked
        if len(pt) > 0: 
            # if user clicked somewhere wrong and value is not between -1 and 1
            if np.abs(pt[0][0]) >= 1: continue
            # update the threshold value
            t_mndwi = pt[0][0]
            # update the plot
            t_line.set_xdata([t_mndwi,t_mndwi])
            # map contours with new threshold
            contours = measure.find_contours(im_mndwi_buffer, t_mndwi)
            # remove contours that contain NaNs (due to cloud pixels in the contour)
            contours = process_contours(contours) 
            # process the water contours into a shoreline
            shoreline = process_shoreline(contours, cloud_mask, im_nodata, georef, image_epsg, settings)
            # convert shoreline to pixels
            if len(shoreline) > 0:
                sl_pix = SDS_tools.convert_world2pix(SDS_tools.convert_epsg(shoreline,
                                                                            settings['output_epsg'],
                                                                            image_epsg), georef)
            else: sl_pix = np.array([[np.nan, np.nan],[np.nan, np.nan]])
            # update the plotted shorelines
            sl_plot1[0].set_data([sl_pix[:,0], sl_pix[:,1]])
            sl_plot2[0].set_data([sl_pix[:,0], sl_pix[:,1]])
            sl_plot3[0].set_data([sl_pix[:,0], sl_pix[:,1]])
            fig.canvas.draw_idle()
        else:
            ax4.set_title('MNDWI pixel intensities and threshold')
            break
    
    # let user manually accept/reject the image
    skip_image = False
    # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
    # this variable needs to be immuatable so we can access it after the keypress event
    key_event = {}
    def press(event):
        # store what key was pressed in the dictionary
        key_event['pressed'] = event.key
    # let the user press a key, right arrow to keep the image, left arrow to skip it
    # to break the loop the user can press 'escape'
    while True:
        btn_keep = plt.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k',fc='w'))
        btn_skip = plt.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k',fc='w'))
        btn_esc = plt.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle="square", ec='k',fc='w'))
        plt.draw()
        fig.canvas.mpl_connect('key_press_event', press)
        plt.waitforbuttonpress()
        # after button is pressed, remove the buttons
        btn_skip.remove()
        btn_keep.remove()
        btn_esc.remove()

        # keep/skip image according to the pressed key, 'escape' to break the loop
        if key_event.get('pressed') == 'right':
            skip_image = False
            break
        elif key_event.get('pressed') == 'left':
            skip_image = True
            break
        elif key_event.get('pressed') == 'escape':
            plt.close()
            raise StopIteration('User cancelled checking shoreline detection')
        else:
            plt.waitforbuttonpress()

    # if save_figure is True, save a .jpg under /jpg_files/detection
    if settings['save_figure'] and not skip_image:
        fig.savefig(os.path.join(filepath, date + '_' + satname + '.jpg'), dpi=150)

    # don't close the figure window, but remove all axes and settings, ready for next plot
    for ax in fig.axes:
        ax.clear()

    return skip_image, shoreline, t_mndwi
