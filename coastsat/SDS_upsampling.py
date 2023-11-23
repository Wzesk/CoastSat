"""
This module adds upsampling functionality to the SDS_preprocess module
# additional requirements: opencv-python, diffusers, pillow   ( pip install -qq diffusers==0.11.1 accelerate )
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

import torch
import requests
from io import BytesIO
from diffusers import LDMSuperResolutionPipeline

np.seterr(all='ignore') # raise/ignore divisions by 0 and nans

###################################################################################################
# UPSAMPLING
###################################################################################################


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
    padding_list = []
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

            # Compute the padding needed to make the sub-image the same size as the others
            padding = {
                        "left":x,
                        "right":width - (x+sub_image_size),
                        "top":y,
                        "bottom":height - (y+sub_image_size)
                       }
            padding_list.append(padding)
            y += y_step_size
        x+=x_step_size


    return sub_images, padding_list,img.size

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

def upsample_subimages(sub_images):
    """
    upsaple each sub-image.

    Parameters:
    sub_images (list of PIL.Image): List of sub-images to upsample.

    Returns:
    list of PIL.Image: List of upsampled sub-images.
    """

    # List to hold the upsampled images
    upsampled_images = []

    # Loop over the sub-images
    for sub_image in sub_images:
        # Upsample the sub-image
        upsampled_image = upsample_subimage(sub_image)

        # Add the upsampled image to the list
        upsampled_images.append(upsampled_image)

    return upsampled_images

def upsample_subimage(sub_image_128):
    """
    Adjusts the brightness of each sub-image.

    Parameters:
    image to upsample

    Returns:
    upsamped image
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = LDMSuperResolutionPipeline.from_pretrained( "CompVis/ldm-super-resolution-4x-openimages")
    pipe = pipe.to(device)

    sub_image_128 = sub_image_128.crop((0, 0, 128, 128))

    upsampled_image = pipe(sub_image_128, num_inference_steps=100, eta=1).images[0]

    return upsampled_image

def pad_sub_images(sub_images, padding_list,target_size,scale=4):
    """
    Pads each sub-image to make it the same size as the others.

    Parameters:
    sub_images (list of PIL.Image): List of sub-images to pad.
    padding_list (list of dict): List of padding dictionaries for each sub-image.

    Returns:
    list of PIL.Image: List of padded sub-images.
    """
    target_size = (target_size[0]*scale,target_size[1]*scale)
    print(target_size)
    padded_images = []
    for i in range(len(sub_images)):
        padded_image = Image.new("RGB", target_size, (0, 0, 0))
        padded_image.paste(sub_images[i], box=(padding_list[i]["left"]*scale,padding_list[i]["top"]*scale) )
        padded_images.append(padded_image)  

    return padded_images

def reassemble_images(padded_sub_images):
    """
    Reassembles sub-images into a full-size image, averaging overlapping areas.

    Parameters:
    sub_images (list of PIL.Image): The list of sub-images.

    Returns:
    PIL.Image: The reassembled image.
    """
    # Get the size of the full image
    full_size = padded_sub_images[0].size

    # Overlay and average the sub-images
    
    # create a temporary image to hold the count of non-black pixels for the current sub-image
    temp_img = Image.new("RGB", full_size, (0, 0, 0))

    for x in range(full_size[0]):
        for y in range(full_size[1]):
            pixel_array = []
            #get the pixel values from each sub-image
            for img in padded_sub_images:
                #if the pixel is black, skip it
                if img.getpixel((x,y)) == (0,0,0):
                    continue
                else:
                    pixel_array.append(img.getpixel((x,y)))
            #average the pixel values
            if len(pixel_array) > 0:
                pixel_average = tuple([int(sum(x) / len(x)) for x in zip(*pixel_array)])
            else:
                pixel_average = (0,0,0)
            #set the pixel value in the temporary image
            temp_img.putpixel((x,y), pixel_average)
    return temp_img

