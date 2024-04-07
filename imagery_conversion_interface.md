TODO Item

Documentation:  

Document the different imagery formats we need.
- need jpg for upsampling

how are the sentinel images stored for coastsat? --> does this mean how is the metadata of an image
from sentinel stored in coastsat?


What format do we get images from earthcache?
- tif files


What format do we need to do upsampling?
- we need RGB image type taken from RGB bands of satellite image
   - An RGB image, sometimes referred to as a truecolor image, is stored in MATLAB as an m-by-n-by-3 data array that defines red, green, and blue color components for each individual pixel
   - we haven't tested it for Infrared/other satellite bands --> should we test to see if it would work?
- jpg is a compressed version of bmp
    - bmp is a bit-map 

Walt's Note: We want to be able to convert images back and forth easily.  It is likely something like this already exists in a python library


## Things we figured out:

1. the create_jpg() function from Coastsat depends on a cloud_mask that is extracted from QA Bands provided
by Landsat --> we may not have that if we're using satellites other than Landsat

2. Coastsat has the following flow of methods
 a. # if you have already downloaded the images, just load the metadata file
    metadata = SDS_download.get_metadata(inputs)
    This gets the metadata in a dicitionary format from the downloaded images
    Would be good to tweak this function depending on what format we get metadata on the images from Earthcache
 b. def save_jpg(metadata, settings, use_matplotlib=False):
    This converts all the images in the metadata dictionary into jpgs (either as a matplotlib plot or as a jpg file)
    Again, this would also be a good one to tweak to run on the entire batch. It does expect some cloud masking stuff.


- could try pretending the images are s2 and feed it through

Code snippets to convert from TIF to JPG:
plt.ioff()  # turning interactive plotting off to use matplotlib
might be good to look at code from bottom part of this notebook:
https://github.com/chris010970/earthcache/blob/main/notebooks/spot.ipynb

def transformimage(sourcePath, destinationPath, currentFileType, targetFileType):
    # uses 'import matplotlib as plt'

    fig = plt.figure()

    fig.set_size_inches([18,9])
    fig.set_tight_layout(True)
    ax1 = fig.add_subplot(111)
    ax1.axis('off')

    ax1.imshow(im_RGB) // key function to use
    ax1.set_title(date + '   ' + satname, fontsize=16)
    fig.savefig(os.path.join(filepath, date + '_' + satname + '.jpg'), dpi=150)