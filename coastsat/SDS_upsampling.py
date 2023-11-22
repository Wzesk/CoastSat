"""
This module adds upsampling functionality to the SDS_preprocess module

"""

# load modules
import os
import numpy as np

# other modules
import pandas as pd

from PIL import Image
import math
import cv2

# CoastSat modules
from coastsat import SDS_tools

np.seterr(all='ignore') # raise/ignore divisions by 0 and nans

###################################################################################################
# UPSAMPLING
###################################################################################################

def upsampling(im, factor):
    #this is a stub function to add upsampling
    
    return True


def extract_sub_images(original_image, sub_image_size, min_overlap):
    # Open the original image
    img = Image.open(original_image)
    width, height = img.size

    # extend the original image if needed to fit the sub_image_size using the pixel color at the edge
    if width < sub_image_size:
        img = img.crop((0, 0, sub_image_size, height))
        width = sub_image_size
        x_steps = 1
        x_step_size = width
    else:
        x_steps = math.ceil((width) / (sub_image_size-min_overlap))
        x_step_size = math.ceil((width - sub_image_size) / (x_steps-1))

    if height < sub_image_size:
        img = img.crop((0, 0, width, sub_image_size))
        height = sub_image_size
        y_steps = 1
        y_step_size = height      
    else:
        y_steps = math.ceil((height) / (sub_image_size-min_overlap))
        y_step_size = math.ceil((height - sub_image_size) / (y_steps-1))

    #inpaint missing pixels
    img_array = infill_missing_pixels(np.array(img), threshold=10, inpaint_radius=3, inpaint_method=cv2.INPAINT_TELEA)

    print(img_array.shape)
    print('x_steps='+str(x_steps)+', y_steps='+str(y_steps)+', x_step_size='+str(x_step_size)+', y_step_size='+str(y_step_size))
    img = Image.fromarray(img_array)

    # List to hold sub-images
    sub_images = []

    # Loop over the image to extract sub-images
    x = 0
    while x <= (width-sub_image_size):
        y = 0
        while y <= (height-sub_image_size):
            # Compute the dimensions of the sub-image
            right = min(x + sub_image_size, width)
            bottom = min(y + sub_image_size, height)

            # Extract and add the sub-image to the list
            sub_image = img.crop((x, y, right, bottom))
            print('sub image at: x='+str(x)+', y='+str(y))
            sub_images.append(sub_image)
            y += y_step_size
        x+=x_step_size


    return sub_images, x_steps, y_steps, x_step_size, y_step_size

def infill_missing_pixels(image, threshold=10, inpaint_radius=3, inpaint_method=cv2.INPAINT_TELEA):
    """
    Replaces black or near-black pixels in an image with the color of the nearest non-black pixel using OpenCV inpainting.

    Parameters:
    image_path: the image file.
    threshold (int): The threshold below which a pixel is considered black (default 10).
    inpaint_radius (int): Radius of a circular neighborhood of each point inpainted.
    inpaint_method (cv2 constant): Inpainting method (cv2.INPAINT_TELEA or cv2.INPAINT_NS).

    Returns:
    numpy.ndarray: The modified image with black pixels infilled.
    """


    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a mask where black pixels are marked
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Apply inpainting
    inpainted_image = cv2.inpaint(image, mask, inpaint_radius, inpaint_method)

    return inpainted_image

def upsample_subimage(sub_images, brightness_factor):
    """
    Adjusts the brightness of each sub-image.

    Parameters:
    sub_images (list of PIL.Image): List of sub-images to adjust.
    brightness_factor (float): Factor to adjust brightness. 
                               >1 to increase brightness, <1 to decrease.

    Returns:
    list of PIL.Image: List of brightness-adjusted sub-images.
    """
    enhanced_images = []

    for img in sub_images:
        enhancer = ImageEnhance.Brightness(img)
        enhanced_img = enhancer.enhance(brightness_factor)
        enhanced_images.append(enhanced_img)

    return enhanced_images
