
# load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb

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

# CoastSat modules
from coastsat import SDS_tools, SDS_preprocess,SDS_shoreline,SDS_preprocess_med

np.seterr(all='ignore') # raise/ignore divisions by 0 and nans

###################################################################################################
# shoreline extraction 

#!pip install simplification

# from simplification.cutil import simplify_coords
# from PIL import Image
###################################################################################################

def get_shoreline(mask_img):
  # Load the mask image
  img_in = Image.open(mask_img)
  # Convert the image to a NumPy array
  mask_array = np.array(img_in)

  # Find contours (boundaries) in the mask
  contours = measure.find_contours(mask_array, 0.5)

  # Assuming you want the longest contour (outer boundary)
  longest_contour = max(contours, key=len)

  # Extract the coordinates of the shoreline points
  shoreline_points = np.array(longest_contour).squeeze()

  # Simplify the shoreline points using the Visvalingam-Whyatt algorithm
  simplified_shoreline = simplify_coords(shoreline_points, 1)  # Adjust 0.01 as needed for desired simplification level

  # Smooth the simplified shoreline using a moving average filter
  window_size = 3  # Adjust as needed for desired smoothing level
  smoothed_shoreline = np.convolve(simplified_shoreline[:, 0], np.ones(window_size)/window_size, mode='same')
  smoothed_shoreline = np.stack((smoothed_shoreline, np.convolve(simplified_shoreline[:, 1], np.ones(window_size)/window_size, mode='same')), axis=-1)

  smoothed_shoreline = smoothed_shoreline[1:-1,:]
  #append the first point to the end of smooth shoreline
  return np.vstack((smoothed_shoreline,smoothed_shoreline[0,:]))

def masks_to_shorelines(settings,metadata):
    """
    

    """
    satname = 'S2'

    sitename = settings['inputs']['sitename']
    filepath_data = settings['inputs']['filepath']
    collection = settings['inputs']['landsat_collection']
    filepath_models = os.path.join(os.getcwd(), 'classification', 'models')
    # initialise output structure
    output = dict([])


    print('Mapping shorelines:')

    # get images
    filepath = SDS_tools.get_filepath(settings['inputs'],satname)
    filenames = metadata[satname]['filenames']

    # initialise the output variables
    output_timestamp = []  # datetime at which the image was acquired (UTC time)
    output_shoreline = []  # vector of shoreline points
    output_filename = []   # filename of the images from which the shorelines where derived
    #output_cloudcover = [] # cloud cover of the images
    output_geoaccuracy = []# georeferencing accuracy of the images
    output_idxkeep = []    # index that were kept during the analysis (cloudy images are skipped)
    #output_t_mndwi = []    # MNDWI threshold used to map the shoreline

    
    # loop through the images
    for i in range(len(filenames)):

        print('\r%s:   %d%%' % (satname,int(((i+1)/len(filenames))*100)), end='')
        # get image filename
        fn = SDS_tools.get_filenames(filenames[i],filepath, satname)
        # get mask image
        fn_mask = fn.replace('ms','mask')
        fn_mask = fn.replace('tif','jpg')
        # get the shoreline from the mask image
        shoreline = get_shoreline(fn_mask)
        
        im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = SDS_preprocess_med.preprocess_single(fn)

        world_shoreline = SDS_tools.convert_pix2world(shoreline, georef)

        # append to output variables
        output_timestamp.append(metadata[satname]['dates'][i])
        output_shoreline.append(world_shoreline)
        output_filename.append(filenames[i])
        #output_cloudcover.append(cloud_cover)
        output_geoaccuracy.append(metadata[satname]['acc_georef'][i])
        output_idxkeep.append(i)
        #output_t_mndwi.append(t_mndwi)

    # create dictionnary of output
    output[satname] = {
            'dates': output_timestamp,
            'shorelines': output_shoreline,
            'filename': output_filename,
            #'cloud_cover': output_cloudcover,
            'geoaccuracy': output_geoaccuracy,
            'idx': output_idxkeep,
            #'MNDWI_threshold': output_t_mndwi,
            }
    print('')

    # change the format to have one list sorted by date with all the shorelines (easier to use)
    output = SDS_tools.merge_output(output)

    # save outputput structure as output.pkl
    filepath = os.path.join(filepath_data, sitename)
    with open(os.path.join(filepath, sitename + '_output.pkl'), 'wb') as f:
        pickle.dump(output, f)

    return output

