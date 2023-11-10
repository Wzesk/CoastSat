"""
This module contains functions to analyze the 2D shorelines along shore-normal
transects
    
Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# load modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import pdb

# other modules
import skimage.transform as transform
from pylab import ginput
from scipy import stats

# CoastSat modules
from coastsat import SDS_tools

#nurbs module
import geomdl
from geomdl import fitting

# Global variables
DAYS_IN_YEAR = 365.2425
SECONDS_IN_DAY = 24*3600

###################################################################################################
# working with polylines
###################################################################################################

def polyline_length(shoreline):
    """
    Calculate the length of a polyline

    Arguments:
    -----------
    shoreline: array
        x and y coordinates of the shoreline

    Returns:
    -----------
    length: float
        length of the polyline
    """
    length = 0
    for i in range(len(shoreline)-1):
        length += np.sqrt((shoreline[i+1][0]-shoreline[i][0])**2 + (shoreline[i+1][1]-shoreline[i][1])**2)

    return length

def divide_polyline(shoreline, length, count):
    """
    Divide a polyline into a given number of equal length segments

    Arguments:
    -----------
    shoreline: array
        x and y coordinates of the shoreline
    length: float
        length of the polyline
    count: integer
        number of segments to divide the polyline into

    Returns:
    -----------
    division_points: array
        x and y coordinates of points along the polyline that divide it into equal length segments
    normal_transects: array
        x and y coordinates of the shore-normal transects at the division points
    """
    division_points = []
    normal_transects = []
    segment_length = length/count
    current_length = 0
    for i in range(len(shoreline)-1):
        current_length += np.sqrt((shoreline[i+1][0]-shoreline[i][0])**2 + (shoreline[i+1][1]-shoreline[i][1])**2)
        if current_length >= segment_length:
            division_points.append(shoreline[i])
            #normal_transects.append(get_normal_transect(shoreline, shoreline[i]))
            normal_transects.append(get_normal_vector(shoreline, shoreline[i]))
            current_length = 0

    return division_points, normal_transects

def get_shortest_shoreline(output):
    """
    Get the shortest shoreline from the output dictionary

    Arguments:
    -----------
    output: dictionary
        output dictionary from the shorelines function

    Returns:
    -----------
    shortest_shoreline: array
        x and y coordinates of the shortest shoreline
    """
    # get the shortest shoreline
    shortest_shoreline = output['shorelines'][0]
    for i in range(1, len(output['shorelines'])):
        if polyline_length(output['shorelines'][i]) < polyline_length(shortest_shoreline):
            shortest_shoreline = output['shorelines'][i]

    return shortest_shoreline

def get_normal_vector(shoreline,division_point):
    """
    Get the normal vector for a point along a polyline

    Arguments:
    -----------
    shoreline: array
        x and y coordinates of the shoreline
    division_point: array
        x and y coordinates of the point along the shoreline

    Returns:
    -----------
    normal_vector: array
        x and y coordinates of the normal vector
    """
    #smooth a set of points along the shoreline
    ss = smooth_polyline(shoreline, 3)

    # get the index of the division point in the shoreline
    index = np.where((ss == division_point).all(axis=1))[0][0]

    # get the two points on either side of the division point
    if index == 0:
        point1 = ss[index]
        point2 = ss[index+1]
    elif index == len(shoreline)-1:
        point1 = ss[index-1]
        point2 = ss[index]
    else:
        point1 = ss[index-1]
        point2 = ss[index+1]

    # get the tangent angle
    angle = np.arctan2(point2[1]-point1[1], point2[0]-point1[0])
    #get the perpendicular angle
    perpendicular_angle = angle + np.pi/2

    #get the perpendicular vector
    perpendicular_vector = np.array([np.cos(perpendicular_angle), np.sin(perpendicular_angle)])

    #normalize vector length
    normal_vector = perpendicular_vector/np.linalg.norm(perpendicular_vector)

    return normal_vector

###################################################################################################
# using nurbs
###################################################################################################

def get_normal_vector_along_nurbs(crv,delta):
    """
    Get the normal vectors along a nurbs curve

    Arguments:
    -----------
    crv: bezier curve
    delta: float
        spacing for the normal vectors

    Returns:
    -----------
    points: list
        x and y coordinates of points along crv
    normal_vectors: list
        x and y coordinates of the normal vector
    """
    crv.delta = delta
    curve_points = crv.evalpts

    #create list numbers from 0 to 1 with 0.01 increments
    t_values = np.arange(0,1,delta)

    normals = []
    #get first ad second derivatives
    for i in range(len(t_values)):
        ders = crv.derivatives(u=t_values[i], order=1)
        #add first dir (position) to second dir (direction) to visualize tangent vector
        tan_vec = np.array([ders[1][0],ders[1][1]])  # tangent vector at u = 0.35
        #normalize vector length
        tan_vector = tan_vec/np.linalg.norm(tan_vec)

        #get angle of tangent vector
        angle = np.arctan2(tan_vec[1], tan_vec[0])
        #get the perpendicular angle
        perpendicular_angle = angle + np.pi/2
        #get the perpendicular vector
        perpendicular_vector = np.array([np.cos(perpendicular_angle), np.sin(perpendicular_angle)])
        #normalize vector length
        normal_vector = perpendicular_vector/np.linalg.norm(perpendicular_vector)

        #append to list
        normals.append(normal_vector)

    return curve_points,normals

def get_planes_along_nurbs(crv,delta):
    """
    Get the normal vectors along a nurbs curve

    Arguments:
    -----------
    crv: bezier curve
    delta: float
        spacing for the normal vectors

    Returns:
    -----------
    points: list
        x and y coordinates of points along crv
    normal_vectors: list
        x and y coordinates of the normal vector
    """
    crv.delta = delta
    curve_points = crv.evalpts

    #create list numbers from 0 to 1 with 0.01 increments
    t_values = np.arange(0,1,delta)

    planes = []
    #get first ad second derivatives
    for i in range(len(t_values)):
        ders = crv.derivatives(u=t_values[i], order=1)
        #add first dir (position) to second dir (direction) to visualize tangent vector
        tan_vec = np.array([ders[1][0],ders[1][1]])  # tangent vector at u = 0.35
        #normalize vector length
        tan_vector = tan_vec/np.linalg.norm(tan_vec)

        #get angle of tangent vector
        angle = np.arctan2(tan_vec[1], tan_vec[0])
        #get the perpendicular angle
        perpendicular_angle = angle + np.pi/2
        #get the perpendicular vector
        perpendicular_vector = np.array([np.cos(perpendicular_angle), np.sin(perpendicular_angle)])
        #normalize vector length
        normal_vector = perpendicular_vector/np.linalg.norm(perpendicular_vector)

        #append to list
        plane = dict([])
        plane['origin'] = curve_points[i]
        plane['xvec'] = tan_vector
        plane['yvec'] = normal_vector
        planes.append(plane)    
    return planes

def get_plane_along_nurbs(crv,t_val):
    """
    Gets a plane along a nurbs curve

    Arguments:
    -----------
    crv: bezier curve
    delta: float
        spacing for the normal vectors

    Returns:
    -----------
    point: array
        x and y coordinates of points along crv
    plane: dict
        origin x and y coordinates, x and y vectors
    """
    curve_pt = crv.evaluate_single(t_val)


    ders = crv.derivatives(u=t_val, order=1)

    tan_vec = np.array([ders[1][0],ders[1][1]])  # tangent vector at u = 0.35
    #normalize vector length
    tan_vector = tan_vec/np.linalg.norm(tan_vec)

    #get angle of tangent vector
    angle = np.arctan2(tan_vec[1], tan_vec[0])
    #get the perpendicular angle
    perpendicular_angle = angle + np.pi/2
    #get the perpendicular vector
    perpendicular_vector = np.array([np.cos(perpendicular_angle), np.sin(perpendicular_angle)])
    #normalize vector length
    normal_vector = perpendicular_vector/np.linalg.norm(perpendicular_vector)

    #create plane
    plane = dict([])
    plane['origin'] = curve_pt
    plane['xvec'] = tan_vector
    plane['yvec'] = normal_vector   
    return plane

def fit_nurbs(shoreline,degree):
    #convert shortest_shoreline into a list of tuples
    pline = [tuple(x) for x in shoreline]

    # Do global curve interpolation
    #crv = fitting.interpolate_curve(pline, degree)
    crv = fitting.approximate_curve(pline, degree, ctrlpts_size=24, centripetal=False)
    return crv

def tangent_dict(curve_points,tangents,extension_length):
    """
    create a dictionary from list of tangents and curve points

    Arguments:
    -----------
    curve_points: list
    tangents: list

    Returns:
    -----------
    tangent_dict: dict
    """
    tangent_dict = dict([])
    for i in range(len(curve_points)):
        t_name = 'NA'+str(i+1)
        tangent_dict[t_name] = np.array([[ curve_points[i][0]-tangents[i][0], curve_points[i][1]-tangents[i][0] ], 
                                         [ curve_points[i][0]+(tangents[i][0]*2), curve_points[i][1]+(tangents[i][0]*2) ]])

    return tangent_dict
 
def closest_point_on_curve(crv,delta,sample_point):
    """
    Get the closest point on a nurbs curve to a sample point

    Arguments:
    -----------
    crv: bezier curve
    delta: float
        spacing to evaluate curvec
    sample_point: array
        x and y coordinates of the sample point

    Returns:
    -----------
    closest_point: array
        x and y coordinates of the closest point on the curve
    """
    crv.delta = delta
    curve_points = crv.evalpts
    #create list numbers from 0 to 1 with 0.01 increments
    t_values = np.arange(0,1,delta)

    #calculate the distance between the sample point and the first point on the curve
    distance = np.sqrt((curve_points[0][0]-sample_point[0])**2 + (curve_points[0][1]-sample_point[1])**2)
    #set the first point as the closest point
    closest_point = curve_points[0]
    closest_point_t = t_values[0]

    #loop through the curve points and check if the distance to the sample point is smaller
    for i in range(1,len(curve_points)):
        new_distance = np.sqrt((curve_points[i][0]-sample_point[0])**2 + (curve_points[i][1]-sample_point[1])**2)
        if new_distance < distance:
            distance = new_distance
            closest_point = curve_points[i]
            closest_point_t = t_values[i]

    return closest_point,closest_point_t

def get_plane_coordinates(point,plane):
    """
    get the coordinates of a point in a plane

    Arguments:
    -----------
    point: list
        x and y coordinates of the point
    plane: dict
        dictionary with the origin, x and y vectors

    Returns:
    -----------
    coordinates: list
        x and y coordinates of the point in the plane
    """
    #get the origin
    origin = plane['origin']
    #get the x and y vectors
    xvec = plane['xvec']
    yvec = plane['yvec']

    #subtract the origin from the point
    tanslated_point = np.array([point[0]-origin[0],point[1]-origin[1]])
    #get angle of xvec
    angle = np.arctan2(xvec[1], xvec[0])

    #rotate translated point by -angle
    rotation_matrix = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
    rotated_point = np.dot(rotation_matrix,tanslated_point)


    # #get the coordinates
    # coordinates = np.array([origin[0]+(point[0]*xvec[0])+(point[1]*yvec[0]),
    #                         origin[1]+(point[0]*xvec[1])+(point[1]*yvec[1])])
    
    return rotated_point