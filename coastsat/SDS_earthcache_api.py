import os
import pandas as pd
import sys

from coastsat import SDS_earthcache_client

# use this function to directly create a pipeline
# the 'image_type_id' parameter refers to what type of
# images you want for the output
# these are the available id's: https://support.skywatch.com/hc/en-us/articles/7297565063067-Available-Outputs
def retrieve_images_earthcache(api_key, name, aoi, start_date, end_date, image_type_id, **kwargs):
  pipeline_name = name
  
  # define repo name and get root working directory
  repo = 'CoastSat'
  root_path = os.getcwd()[ 0 : os.getcwd().find( repo ) + len ( repo )]

  # get path to configuration files
  cfg_path = os.path.join( root_path, 'earthcache-cfg' )

  # create instance of shclient class
  global client
  client = SDS_earthcache_client.EcClient(cfg_path, api_key, max_cost=10)

  # we can change the resolution here
  resolution = [ 'low' ]
  
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
  print(status, result)
  

# call this function to get the images once the pipeline is ready
# make sure to pass in the name of the pipeline    
def download_images(pipeline_name):
  global client
  id = client.getPipelineIdFromName(pipeline_name)
  
  status, result = client.getIntervalResults(id)
  if(status == 404):
    print("results not found, most likely an invalid id!")
    return
  
  root_path = pipeline_name + '/images'
  images = []

  # convert to dataframe
  df = pd.DataFrame( result[ 'data' ] )
  for row in df.itertuples():
    out_path = os.path.join( root_path, row.id )
    images.append( client.getImages( row.results, out_path ) )  
