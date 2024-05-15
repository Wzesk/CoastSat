"""
  CREDIT ACKNOWLEDGEMENT: a lot of the code used in this file has been borrowed from
  the following public github repository: https://github.com/chris010970/earthcache
"""

import os
import shutil
import pandas as pd
import sys
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
# from coastsat import SDS_earthcache_client

# always call this function first before using anything else
# does an initial setup
def initialize_client(api_key,max_cost=0):
  repo = 'CoastSat'
  root_path = os.getcwd()[ 0 : os.getcwd().find( repo ) + len ( repo )]
  cfg_path = os.path.join( root_path, 'earthcache-cfg' )
  global client
  client = SDS_earthcache_client.EcClient(cfg_path, api_key,max_cost)
  return client

# use this function to directly create a pipeline
# the 'image_type_id' parameter refers to what type of
# images you want for the output
# these are the available id's: https://support.skywatch.com/hc/en-us/articles/7297565063067-Available-Outputs
def retrieve_images_earthcache(client, name, aoi, start_date, end_date, image_type_id, interval='30d', resolution='low', **kwargs):
  pipeline_name = name
  
  status, result = client.createPipeline(    
                                          name=pipeline_name,
                                          start_date=start_date,
                                          end_date=end_date,
                                          aoi=aoi,
                                          interval=interval,
                                          resolution=resolution,
                                          output={
                                             "id": image_type_id,
                                              "format": "geotiff",
                                              "mosaic": "stitched"
                                          },
                                          **kwargs
                                        )
  print(status, result)
  
  pipeline_id = client.getPipelineIdFromName(pipeline_name)
  status, result = client.getPipeline( pipeline_id )
  print(status, result)
  
  
# use this function to check the status of a given pipeline
# 200 is a good result!  
def checkStatus(client, pipeline_name):
  pipeline_id = client.getPipelineIdFromName(pipeline_name)
  status, result = client.getIntervalResults( pipeline_id )
  return status, result
  

# call this function to get the images once the pipeline is ready
# make sure to pass in the name of the pipeline 
# returns a list of the images   
def download_images(client, pipeline_name):
  id = client.getPipelineIdFromName(pipeline_name)
  
  status, result = client.getIntervalResults(id)
  if(status == 404):
    print("results not found, most likely an invalid id!")
    return
  
  root_path = 'data/' + pipeline_name
  images = []

  # convert to dataframe
  df = pd.DataFrame( result[ 'data' ] )
  for row in df.itertuples():
    out_path = os.path.join( root_path, row.id )
    print(out_path)
    images.append( client.getImages( row.results, out_path ) )  
  
  return images

# a function to actually view the images (needs testing!)
# be sure to call download_images first ^
# and then pass in the returned result of that to this function

# not entirely sure how this function works
def view_first_image(images):
  ds = gdal.Open( images[ 0 ][ 0 ] )
  data = ds.ReadAsArray()
  # this is just showing the first image in the list
  np.amin( data[ 0, : , : ]), np.amax( data[ 0, : , : ])
  plt.imshow( data[ 0, :, :] )
  plt.show()
  

# posts a search request with the given parameters
# full list of parameters: https://api-docs.earthcache.com/#tag/post
# returns status, result, search_df, search_id
# best usage:
# status, result, search_df, search_id = SDS_earthcache_api.search(__, __, __, ___)
def search(aoi, window, resolution, **kwargs):
  pass
  
  
# allows you to create a pipeline directly from a previously run search
# need to call the function above first though (aka need to search first)
# so that you can save the search_id and search_results
# and pass them in as parameters to this function
# returns the status and result
# best usage:
# status, result = SDS_earthcache_api.create_pipeline_from_search(___, ___)
# https://api-docs.earthcache.com/#tag/pipelines/operation/PipelineCreate

def create_pipeline_from_search(client, search_id, search_results):
  status, result = client.createPipelineFromSearch(search_id, search_results)
  return status, result
  
# still needs to be tested! 
# Calculate cost of area and intervals of a pipeline, 
# and the probability of collection of any tasking intervals
# https://api-docs.earthcache.com/#tag/pipelinePost 
def calculatePrice(client, resolution, location, start_date, end_date):
    status, result = client.calculatePrice(resolution, location, start_date, end_date)
    return status, result
  
  
# This function rearranges the downloaded images into the format that Coastsat
# expects for some of their preprocessing functions.
# args:
  # Pass in the root directory with all of the downloaded files from the pipelines
  # Pass in True if this is the first time this function is being called on the directory
  # pass in False if it's not the first time this function is being called on the directory
    # if passing in false, remember that the function expects for there to already be 
    # an S2 folder in the passed in directory with a json_files and ms folder inside. 
  # Pass in the name of the site (like Ukulhas)
def format_downloads(directory, isFirstTime, sitename):
    # create a new folder S2
    # create new folders:
      # meta
      # ms
      # swir
      # mask    
    print(os.path)
    if(isFirstTime):
      # TODO: this should be changed based on which satellite data was taken from.
      # probably would need to look through the json file and create accordingly
      satellite_path = os.path.join(directory, "S2")
      os.mkdir(satellite_path)
      meta_path = os.path.join(satellite_path, "json_files")
      os.mkdir(meta_path)
      meta_coastsat_path = os.path.join(satellite_path, "meta")
      os.mkdir(meta_coastsat_path)
      ms_path = os.path.join(satellite_path, "ms")
      os.mkdir(ms_path)
      os.mkdir(os.path.join(satellite_path, "mask"))
      os.mkdir(os.path.join(satellite_path, "swir"))
    else:
      satellite_path = os.path.join(directory, "S2")
      meta_path = os.path.join(satellite_path, "json_files")
      meta_coastsat_path = os.path.join(satellite_path, "meta")
      ms_path = os.path.join(satellite_path, "ms")
      
    for foldername, subfolders, filenames in os.walk(directory):
      # Check if there is a TIF or JSON file in the current folder
      if any(file.endswith('.json') for file in filenames) or any(file.endswith(('.tif')) for file in filenames):
          # Move the image and JSON files to their respective folders in the root directory
          for file in filenames:
              if file.endswith('.json'):
                if(not(os.path.exists(os.path.join(meta_path, file)))):
                  shutil.move(os.path.join(foldername, file), meta_path)
              elif file.endswith(('.tif')):
                if(not(os.path.exists(os.path.join(ms_path, file)))):
                  shutil.move(os.path.join(foldername, file), ms_path)
    rename_files(meta_path, ms_path, sitename)
                  
# helper functions for format_downloads() -- specifically renaming the files
# --
def get_image_date(json_file):
     with open(json_file, 'r') as f:
        data = json.load(f)
        start_time = data["ProductInfo"]["PRODUCT_SCENE_RASTER_START_TIME"]
        # Parse the start time string to extract the date
        date = datetime.strptime(start_time, "%d-%b-%Y %H:%M:%S.%f")
        # Format the date as YYYYMMDD
        formatted_date = date.strftime('%Y-%m-%d-%H-%M-%S')
        return formatted_date

def rename_files(json_folder, tif_folder, sitename):
    json_files = sorted(os.listdir(json_folder))
    tif_files = sorted(os.listdir(tif_folder))

    for json_file, tif_file in zip(json_files, tif_files):
        if json_file.endswith('.json') and tif_file.endswith('.tif'):
            # Get the date from the JSON file
            date = get_image_date(os.path.join(json_folder, json_file))
            
            # Construct new filenames
            new_json_name = f"{date}_S2_{sitename}.json"
            new_tif_name = f"{date}_S2_{sitename}_ms.tif"
            
            # Rename JSON and TIFF files
            os.rename(os.path.join(json_folder, json_file), os.path.join(json_folder, new_json_name))
            os.rename(os.path.join(tif_folder, tif_file), os.path.join(tif_folder, new_tif_name))
#--

    
# This function goes through the json files provided by Earthcache
# and creates text files that CoastSat expects for subsequent functions. 
# args:
  # Pass in the directory with the json files of the specific site
  # Pass in the directory where the text files will be stored
# WARNING: THIS MUST BE CHANGED!
# right now, a lot of this stuff is hardcoded for Ukulhas only.   
def extract_metadata(json_directory, output_directory):
    json_files = sorted(os.listdir(json_directory))

    for json_file in json_files:
        if json_file.endswith('.json'):
            # Get the date from the JSON file
            date = get_image_date(os.path.join(json_directory, json_file))
            
            # Construct new filenames with additional information
            new_text_filename = os.path.splitext(json_file)[0] + "_ms.txt"
          
            # Construct full path for the output text file
            output_text_path = os.path.join(output_directory, new_text_filename)  
            
            width = get_image_width(os.path.join(json_directory, json_file))
            height = get_image_height(os.path.join(json_directory, json_file))
            
            # Write information to the text file
            with open(output_text_path, 'w') as txt_file:
                txt_file.write(f"filename\t{date}_S2_ukulhas_ms.tif\n")
                txt_file.write("epsg\t32643\n")
                txt_file.write("acc_georef\tPASSED\n")
                txt_file.write("image_quality\tPASSED\n")
                txt_file.write(f"im_width\t{width}\n")
                txt_file.write(f"im_height\t{height}\n")
                
# helper functions for extract_metadata()!
#--
def get_image_height(json_file):
     with open(json_file, 'r') as f:
        data = json.load(f)
        height = data["ProductInfo"]["NROWS"]
        return height
      
def get_image_width(json_file):
  with open(json_file, 'r') as f:
        data = json.load(f)
        width = data["ProductInfo"]["NCOLS"]
        return width
#--

