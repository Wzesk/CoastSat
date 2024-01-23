import os
import pandas as pd
import sys

from coastsat import SDS_earthcache_client

#output ID should be 4 band

def retrieve_images_earthcache(api_key, pipeline_name, aoi, start_date, end_date):
  
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
                                             "id": "a8fc3dde-a3e8-11e7-9793-ae4260ee3b4b",
                                              "format": "geotiff",
                                              "mosaic": "stitched"
                                          }
                                        )
  print(status, result)
  
  global pipeline_id
  pipeline_id = client.getPipelineIdFromName(pipeline_name)
  status, result = client.getPipeline( pipeline_id )
  print(status, result)
  
  
def checkStatus():
  global client
  global pipeline_id
  global status, result
  status, result = client.getIntervalResults( pipeline_id )
  print(status, result)
  
    
def download_images():
  root_path = 'Users/Tishya/Documents/CoastSat/earthcache-cfg'
  images = []

  # convert to dataframe
  df = pd.DataFrame( result[ 'data' ] )
  for row in df.itertuples():
    out_path = os.path.join( root_path, row.id )
    images.append( client.getImages( row.results, out_path ) )  
