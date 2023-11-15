"""
This module contains functions to analyze the 2D shorelines along shore-normal
transects
    
Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# load modules
import os
import csv
import numpy as np

#nurbs module
import geomdl
from geomdl import fitting

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
    Get planes along a nurbs curve

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

def fit_nurbs(shoreline,degree=3,size=24,periodic=False):
    #if periodic, set the last point equal to the first point
    if periodic:
        shoreline[len(shoreline)-1] = shoreline[0]

    #convert shoreline into a list of tuples
    pline = [tuple(x) for x in shoreline]

    # Do global curve interpolation
    #crv = fitting.interpolate_curve(pline, degree)
    crv = fitting.approximate_curve(pline, degree, ctrlpts_size=size, centripetal=False)
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
 

###################################################################################################
# checking ditances along nurbs plane based transects
###################################################################################################
def closest_point_in_plane_axis(pts,plane,xweight=1):
    """
    Get the closest point from a set of pts to a plane, with a weight on the x-axis

    Arguments:
    -----------
    pts: list
        x and y coordinates of points
    plane: dict
        origin, xaxis and yaxis of the plane

    Returns:
    -----------
    closest_point: array
        x and y coordinates of the closest point on the curve
    """
    #calculate the distance between the sample point and the first point on the curve
    pt1 = get_plane_coordinates(pts[0],plane)
    distance = pt1[1]
    weighted_distance =  np.sqrt((pt1[0]*xweight)**2 + pt1[1]**2)

    #set the first point as the closest point
    closest_point = pts[0]

    #loop through the curve points and check if the distance to the sample point is smaller
    for i in range(1,len(pts)):    
        pt = get_plane_coordinates(pts[i],plane) #set new point to check
        new_distance = pt[1]
        new_weighted_distance = np.sqrt((pt[0]*xweight)**2 + (pt[1])**2)
        if new_weighted_distance < weighted_distance:
            weighted_distance = new_weighted_distance
            distance = new_distance
            closest_point = pts[i]
    return closest_point,distance

def record_shoreline_offsets(output,settings,offsets_path):
    """
    record relative offsets to reference nurbs for all shorelines

    Arguments:
    -----------
    output: dict with all properties for littoral zone
        x and y coordinates of points
    plane: dict
        origin, xaxis and yaxis of the plane

    Returns:
    -----------
    updated reference crv: nurbs curve
    """
    shorelines = output['shorelines']
    dates = output['dates']
    ref_crv = settings['reference_nurbs']
    ref_planes = get_planes_along_nurbs(ref_crv,settings['delta'])
    offsets = []

    #delete file if it exists
    if os.path.isfile(offsets_path):
        os.remove(offsets_path)

    print('calcualting %d shorelines' % len(shorelines))
    pts_out = []
    for i in range(len(shorelines)):
        try:
            #get offsets
            new_pts = []
            distances = [str(dates[i])]
        
            for j in range(len(ref_planes)):
                #get closest point on the reference plane
                closest_point,distance = closest_point_in_plane_axis(shorelines[i],ref_planes[j])
                new_pts.append(closest_point)
                distances.append(str(distance))    
            #print('calculated %d pts' % len(new_pts)) 

            #update ref crv only updated the reference if all points were found
            if len(new_pts) == len(ref_planes): 
                try:

                    ref_crv = fit_nurbs(new_pts,degree=3,size=25,periodic=settings['periodic'])
                    ref_planes = get_planes_along_nurbs(ref_crv,settings['delta'])

                except Exception as e:
                    print("error updating ref curve for shoreline: "+ str(i))
                    print(str(e))
                    continue 
                #print('Updated ref curve at shoreline: '+ str(i))
            else:
                print('Did not updated ref curve at shoreline, was not able to generate enough new points: '+ str(i))
            
            #write to csv
            res = add_line_to_csv(distances,offsets_path)
            if res == False:
                raise Exception('failed to write line: '+ str(i))
            
        except Exception as e:
            print("failed to process shoreline for date: "+ str(dates[i]))
            print(str(e))
            continue  
        pts_out =  new_pts 
    return ref_crv,pts_out

def add_line_to_csv(newline,path):
    """
    add a line to a csv file

    Arguments:
    -----------
    newline: array
        offsets to add as a new row
    path: string
        path to the csv file

    Returns:
    -----------
    """
    try:
        #check if file exists
        if os.path.isfile(path):
            #open file
            with open(path, 'a',newline='') as f:
                #create a writer
                writer = csv.writer(f, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                #write the row
                writer.writerow(newline)
        else:
            #create a new file
            with open(path, 'w',newline='') as f:
                #create a writer
                writer = csv.writer(f, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                #create the first line with column names
                columns = ['time']
                for i in range(len(newline)-1):
                    columns.append('T_'+str(i+1))
                #write the coumn names
                writer.writerow(columns)
                #write the row
                writer.writerow(newline)
        return True
    except Exception as e:
        print("failed to add line: "+ newline)
        print(str(e))
        return False

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
    
    return rotated_point

###################################################################################################
# removing coastlines with large gaps
###################################################################################################

def get_max_distance_between_polyline_points(shoreline):
    """
    Get the maximum distance between two points along a polyline

    Arguments:
    -----------
    shoreline: array
        x and y coordinates of the shoreline

    Returns:
    -----------
    max_distance: float
        maximum distance between two points along the polyline
    """
    max_distance = 0
    for i in range(len(shoreline)-1):
        distance = np.sqrt((shoreline[i][0]-shoreline[i+1][0])**2 + (shoreline[i][1]-shoreline[i+1][1])**2)
        if distance > max_distance:
            max_distance = distance

    return max_distance

def remove_gaps(output,max_dist):
    """
    Function to remove from the output dictionnary entries containing shorelines with gaps.
   Arguments:
    -----------
        output: dict
            contains output dict with shoreline and metadata
        
    Returns:    
    -----------
        output_no_gaps: dict
            contains the updated dict where gaps
        
    """
    gaps = 0
    plines  = output['shorelines'].copy()
    output_no_duplicates = dict([])

    # find indices of shorelines to be removed
    idx = []
    for i in range(len(plines)):
        gap_distance = get_max_distance_between_polyline_points(plines[i])
        if gap_distance < max_dist:
            idx.append(i)

    output_filtered = dict([])
    for key in output.keys():
        output_filtered[key] = [output[key][i] for i in idx]
    print('%d shorelines with gaps removed' % (len(plines)-len(idx)))
    return output_filtered

