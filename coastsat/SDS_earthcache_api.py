"""
  CREDIT ACKNOWLEDGEMENT: a lot of the code used in this file has been borrowed from
  the following public github repository: https://github.com/chris010970/earthcache
"""

import os
import pandas as pd
import sys
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from coastsat import SDS_earthcache_client

# always call this function first before using anything else
# does an initial setup
def initialize(api_key):
  repo = 'CoastSat'
  root_path = os.getcwd()[ 0 : os.getcwd().find( repo ) + len ( repo )]
  cfg_path = os.path.join( root_path, 'earthcache-cfg' )
  global client
  client = SDS_earthcache_client.EcClient(cfg_path, api_key)

# use this function to directly create a pipeline
# the 'image_type_id' parameter refers to what type of
# images you want for the output
# these are the available id's: https://support.skywatch.com/hc/en-us/articles/7297565063067-Available-Outputs
def retrieve_images_earthcache(api_key, name, aoi, start_date, end_date, image_type_id, **kwargs):
  pipeline_name = name
  
  # we can change the resolution here
  resolution = [ 'low' ]
  
  global client
  status, result = client.createPipeline(    
                                          name=pipeline_name,
                                          start_date=start_date,
                                          end_date=end_date,
                                          aoi=aoi,
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
def checkStatus(pipeline_name):
  global client
  pipeline_id = client.getPipelineIdFromName(pipeline_name)
  status, result = client.getIntervalResults( pipeline_id )
  return status, result
  

# call this function to get the images once the pipeline is ready
# make sure to pass in the name of the pipeline 
# returns a list of the images   
def download_images(pipeline_name):
  global client
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
  
  
# allows you to create a pipeline directly from a previously run search
# need to call the function above first though (aka need to search first)
# so that you can save the search_id and search_results
# and pass them in as parameters to this function
# returns the status and result
# best usage:
# status, result = SDS_earthcache_api.create_pipeline_from_search(___, ___)
# https://api-docs.earthcache.com/#tag/pipelines/operation/PipelineCreate

  def create_pipeline_from_search(search_id, search_results):
    global client
    status, result = client.createPipelineFromSearch(search_id, search_results)
    return status, result
  
# still needs to be tested! 
# Calculate cost of area and intervals of a pipeline, 
# and the probability of collection of any tasking intervals
# https://api-docs.earthcache.com/#tag/pipelinePost 
  def calculatePrice(self, resolution, location, start_date, end_date):
    global client
    status, result = client.calculatePrice(resolution, location, start_date, end_date)
    return status, result