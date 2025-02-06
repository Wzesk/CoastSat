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
import pickle
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
def format_downloads(directory, isFirstTime):
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
      if not os.path.exists(satellite_path):
        os.mkdir(satellite_path)
      meta_path = os.path.join(satellite_path, "json_files")
      if not os.path.exists(meta_path):
        os.mkdir(meta_path)
      meta_coastsat_path = os.path.join(satellite_path, "meta")
      if not os.path.exists(meta_coastsat_path):
        os.mkdir(meta_coastsat_path)
      ms_path = os.path.join(satellite_path, "ms")
      if not os.path.exists(ms_path):
        os.mkdir(ms_path)
      if not os.path.exists(os.path.join(satellite_path, "mask")):
        os.mkdir(os.path.join(satellite_path, "mask"))
      if not os.path.exists(os.path.join(satellite_path, "swir")):
        os.mkdir(os.path.join(satellite_path, "swir"))
    else:
      satellite_path = os.path.join(directory, "S2")
      meta_path = os.path.join(satellite_path, "json_files")
      meta_coastsat_path = os.path.join(satellite_path, "meta")
      ms_path = os.path.join(satellite_path, "ms")
      
    for foldername, subfolders, filenames in os.walk(directory):
      # Check if there is a TIF or JSON file in the current folder
      if any(file.endswith('.json') for file in filenames) or any(file.endswith(('.tif')) for file in filenames) or any(file.endswith(('.txt')) for file in filenames):
          # Move the image and JSON files to their respective folders in the root directory
          for file in filenames:
              if file.endswith('.json'):
                # TODO: rename file here before moving
                if(not(os.path.exists(os.path.join(meta_path, file)))):
                  shutil.copy(os.path.join(foldername, file), meta_path)#shutil.move(os.path.join(foldername, file), meta_path)
              elif file.endswith('.txt'):
                # TODO: rename file here before moving
                if(not(os.path.exists(os.path.join(meta_coastsat_path, file)))):
                  shutil.copy(os.path.join(foldername, file), meta_coastsat_path)#shutil.move(os.path.join(foldername, file), meta_path)
              elif file.endswith(('.tif')):
                if(not(os.path.exists(os.path.join(ms_path, file)))):
                  # TODO: rename file here before moving
                  shutil.copy(os.path.join(foldername, file), ms_path)#shutil.move(os.path.join(foldername, file), ms_path)

############################################################################################################
# adding some additional functions to format the project for coastsat.  Not sure these should reside here...
############################################################################################################

#function to get all of the project paths in the data directory that include a particular string
def get_project_paths(project_name,data_dir='data'):
    project_paths = []
    for root, dirs, files in os.walk(data_dir):
        for name in dirs:
            if project_name in name:
                project_paths.append(os.path.join(root, name))
    return project_paths
  
#Use the list of project paths and the projet name to generate metadata and format downloads
def prep_download_folders(project_paths, project_name):
    for proj_path in project_paths:
        generate_metadata_txt(proj_path,project_name)
        format_downloads(proj_path, True)
        move_folder(project_name, proj_path+'\\S2','S2')

#reformat project downloads
def reformat_downloads(project_lookup, project_name, data_dir='data'):
    project_paths = get_project_paths(project_lookup,data_dir)
    prep_download_folders(project_paths, project_name)

    rename_files(data_dir +'\\'+ project_name,'S2')

#rebuild file name using directory and file name
def rebuild_file_name(file_name):
    name_parts = file_name.split("_")
    type = name_parts[1]
    new_name = name_parts[3]
    #split new name at T character and turn the first part into a date
    date_part = new_name.split("T")[0]
    date_part = datetime.strptime(date_part, '%Y%m%d').strftime('%Y-%m-%d')
    #turn the second part into a time
    time_part = new_name.split("T")[1]
    time_part = datetime.strptime(time_part, '%H%M%S').strftime('%H-%M-%S')
    new_name = date_part + "-" + time_part + "-" + type
    return new_name

#function to rename all tif and txt files in directory and its subdirectories
def rename_files(directory,type):

    sat_file_names = {'S2':[]}

    p_name = directory.split("\\")[-1]
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.tif') or file.endswith('.txt'):
                file_beginning = rebuild_file_name(file)

            if file.endswith('.txt'):
                new_name = file_beginning + "_" + p_name + '.txt'
                print("Renaming " + file + " to " + new_name)
                os.rename(os.path.join(root, file), os.path.join(root, new_name))

            if file.endswith('.tif'):
                new_name = file_beginning + "_" + p_name + '_ms.tif'
                print("Renaming " + file + " to " + new_name) 
                sat_file_names['S2'].append(new_name)              
                os.rename(os.path.join(root, file), os.path.join(root, new_name)) 
    #save sat_file_names to a pickle file
    with open(directory + "\\" + p_name + "_metadata.pkl", 'wb') as f:
        pickle.dump(sat_file_names, f)
      
#function to move newly created folder into a folder with the site name within the data folder
def move_folder(site_name, folder_name,type):  
    #if folder does not exist, create it
    if not os.path.exists("data\\" + site_name + "\\" + type):
        os.makedirs("data\\" + site_name + "\\" + type)
    #os.rename(folder_name, "data\\" + site_name + "\\" + type)
    for root, dirs, files in os.walk(folder_name):
      for dir in dirs:
        #check if the folder already exists in the site folder
        if not os.path.exists("data\\" + site_name + "\\" + type + "\\" + dir):
          shutil.move(os.path.join(root, dir), "data\\" + site_name + "\\" + type + "\\" + dir)
        else:
           #move the files in the folder to the site folder
          for file in os.listdir(os.path.join(root, dir)):
            shutil.move(os.path.join(root, dir, file), "data\\" + site_name + "\\" + type + "\\" + dir + "\\" + file)

#create a function that reads a json metadata file for a satellite image and generates a txt file with the metadata
def generate_metadata_txt(directory,project_name):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_file = os.path.join(root, file)

                with open(json_file) as f:
                    data = json.load(f)

                geo = data['GeoCodings'][0]['Coordinate_Reference_System']['WKT']
                geo = geo.replace("\n","")
                espg = geo.split("AUTHORITY")[-1].split(",")[-1].replace("]","").replace("\"","")

                width = data['ProductInfo']['NCOLS']
                height = data['ProductInfo']['NROWS']

                #rebuild file name
                file_begining = rebuild_file_name(file)

                tif_file = file_begining + "_" + project_name + '_ms.tif'

                #get the bands for this image
                bands = data['Bands']
                bandlist = []
                for band in bands:
                    bandlist.append(band['BAND_NAME'])

                #create a dictionary with the metadata
                metadata = {
                    "filename":tif_file,
                    "epsg":espg,
                    "acc_georef":"PASSED",
                    "image_quality":"PASSED",
                    "im_width":width,
                    "im_height":height,
                    "bands":bandlist
                }

                # #replace tif with txt
                txt_file = json_file.replace("_metadata.json", ".txt")
                #write the metadata to a txt file, line by line, delimited by a tab
                with open(txt_file, 'w') as f:
                    for key in metadata.keys():
                        f.write("%s\t%s\n" % (key, metadata[key])
                        )       
                #print("Metadata saved to: " + txt_file)