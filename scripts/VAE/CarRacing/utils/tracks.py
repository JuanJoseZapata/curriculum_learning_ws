import math
import numpy as np
import cv2
from gym_multi_car_racing import bezier
from gymnasium.utils import seeding
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import geopandas as gpd
import shapely

CHECKPOINTS = 12
SCALE       = 6.0
TRACK_RAD   = 900/SCALE
TRACK_TURN_RATE = 0.31
TRACK_DETAIL_STEP = 21/SCALE
PLAYFIELD   = 2000/SCALE # Game over boundary

IMG_SIZE = 32  # Pixels


# Create a 10x10 grid and set 12 random pixels to 1
def get_random_points(n=12):
    """Get n random points in a 10x10 grid

    Args:
        n (int, optional): Number of points. Defaults to 12.

    Returns:
        np.array: Random points
    """
    grid = np.zeros((10, 10))
    points = []
    for k in range(n):
        while True:
            i, j = np.random.randint(0, grid.shape[0]), np.random.randint(0, grid.shape[1])
            if grid[i, j] == 0:
                grid[i, j] = 1
                points.append([i, j])
                break

    points = np.array(points)

    # Multiply by 40 to get the real coordinates
    points = points

    return points


def get_track(control_points=None, mindst=0.08, np_random=None):
    """Get track from control points or random points

    Args:
        control_points (list, optional): Control points. Defaults to None.
        mindst (float, optional): Minimum distance. Defaults to 0.08.
        np_random (np.random, optional): Numpy random. Defaults to None.
    
    Returns:
        np.array, np.array, np.array: Control points, x, y
    """

    if control_points is not None:
        a = np.array(control_points)
        x, y, _ = bezier.get_bezier_curve(a=a, rad=0.2, edgy=0.2, numpoints=40)
    else:
        a = bezier.get_random_points(n=12, scale=PLAYFIELD, mindst=mindst, np_random=np_random) // 30 * 30
        x, y, _ = bezier.get_bezier_curve(a=a, rad=0.2, edgy=0.2, numpoints=40)

    x, y = np.clip(0, x.max(), x), np.clip(0, y.max(), y)

    return a, x, y


def get_track_default():
    """Get a track using the default algorithm from the original CarRacing environment (i.e. not using Bezier curves)

    Returns:
        np.array, np.array, np.array: Control points, x, y
    """
    # Create checkpoints
    checkpoints = []
    for c in range(CHECKPOINTS):
        alpha = 2*math.pi*c/CHECKPOINTS + np_random.uniform(0, 2*math.pi*1/CHECKPOINTS)
        rad = np_random.uniform(TRACK_RAD/3, TRACK_RAD)
        if c==0:
            alpha = 0
            rad = 1.5*TRACK_RAD
        if c==CHECKPOINTS-1:
            alpha = 2*math.pi*c/CHECKPOINTS
            start_alpha = 2*math.pi*(-0.5)/CHECKPOINTS
            rad = 1.5*TRACK_RAD
        checkpoints.append( (alpha, rad*math.cos(alpha), rad*math.sin(alpha)) )

    # print "\n".join(str(h) for h in checkpoints)
    # self.road_poly = [ (    # uncomment this to see checkpoints
    #    [ (tx,ty) for a,tx,ty in checkpoints ],
    #    (0.7,0.7,0.9) ) ]
    road = []

    # Go from one checkpoint to another to create track
    x, y, beta = 1.5*TRACK_RAD, 0, 0
    dest_i = 0
    laps = 0
    track = []
    no_freeze = 2500
    visited_other_side = False
    while True:
        alpha = math.atan2(y, x)
        if visited_other_side and alpha > 0:
            laps += 1
            visited_other_side = False
        if alpha < 0:
            visited_other_side = True
            alpha += 2*math.pi
        while True: # Find destination from checkpoints
            failed = True
            while True:
                dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                if alpha <= dest_alpha:
                    failed = False
                    break
                dest_i += 1
                if dest_i % len(checkpoints) == 0:
                    break
            if not failed:
                break
            alpha -= 2*math.pi
            continue
        r1x = math.cos(beta)
        r1y = math.sin(beta)
        p1x = -r1y
        p1y = r1x
        dest_dx = dest_x - x  # vector towards destination
        dest_dy = dest_y - y
        proj = r1x*dest_dx + r1y*dest_dy  # destination vector projected on rad
        while beta - alpha >  1.5*math.pi:
                beta -= 2*math.pi
        while beta - alpha < -1.5*math.pi:
                beta += 2*math.pi
        prev_beta = beta
        proj *= SCALE
        if proj >  0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001*proj))
        if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001*proj))
        x += p1x*TRACK_DETAIL_STEP
        y += p1y*TRACK_DETAIL_STEP
        track.append( (alpha,prev_beta*0.5 + beta*0.5,x,y) )
        if laps > 4:
                break
        no_freeze -= 1
        if no_freeze==0:
                break

    # Find closed loop range i1..i2, first loop should be ignored, second is OK
    i1, i2 = -1, -1
    i = len(track)
    while True:
        i -= 1
        if i==0:
            return False  # Failed
        pass_through_start = track[i][0] > start_alpha \
            and track[i-1][0] <= start_alpha
        if pass_through_start and i2==-1:
            i2 = i
        elif pass_through_start and i1==-1:
            i1 = i
            break
    assert i1!=-1
    assert i2!=-1

    track = track[i1:i2-1]
    
    track = np.array(track)
    checkpoints = np.array(checkpoints)

    # Center the track at 165,165
    # Get track center
    track_center = np.mean(track[:,2:], axis=0)
    # Center the track
    track[:,2:] -= track_center
    checkpoints[:,1:] -= track_center
    track += 165
    checkpoints[:,1:] += 165

    return checkpoints[:,1:], track[:, 2], track[:, 3]


def plot_track(track, checkpoints=None, ax=None):
    """Visualize racing track with the same color as the CarRacing environment. Checkpoints are also plotted.

    Args:
        track (np.array): Coordinates of the track
        checkpoints (np.array, optional): Checkpoints. Defaults to None.

    Returns:
        None
    """
    # Plot track
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(track[:,1], track[:,0], linewidth=9, color=[0.4, 0.4, 0.4])

    # Plot checkpoints
    if checkpoints is not None:
        for i in range(len(checkpoints)):
            y, x = checkpoints[i,:]
            ax.plot(x, y, 'o', markersize=2, color="tab:orange")
            ax.text(x, y, str(i), fontsize=10, color="black")
    
    ax.set_xlim(-20,360)
    ax.set_ylim(-20,360)
    ax.set_aspect('equal')

    ax.set_facecolor(np.array([102, 230, 102])/255.)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


def _get_track_image(coordinates, img_size=(32,32)):
    """Get track image from coordinates

    Args:
        coordinates (np.array): Coordinates of the track
        img_size (tuple, optional): Image size. Defaults to (32,32).

    Returns:
        np.array: Track image
    """
    max_value = 330 if coordinates.max() < 330 else coordinates.max()

    if coordinates[:,0].min() < 0:
        coordinates[:,0] = coordinates[:,0] - coordinates[:,0].min()
    if coordinates[:,1].min() < 0:
        coordinates[:,1] = coordinates[:,1] - coordinates[:,1].min()
    coordinates = coordinates // (max_value/(img_size[0]-1))
    coordinates = coordinates.astype(np.int32)

    # Clip to 0-31
    coordinates = np.clip(coordinates, 0, img_size[0]-1)
    
    img = np.zeros(img_size)

    for i in range(coordinates.shape[0]):
        img[coordinates[i,1], coordinates[i,0]] = 1

    return img


def get_track_image(points, img_size=(64, 64), fill=False):
    """
    Create a 2D image of a curve using a set of points.

    Parameters:
    - points: List of tuples representing (x, y) coordinates of the curve.
    - img_size: Tuple representing the size of the image (width, height).
    - fill: Boolean indicating whether to fill the inside of the curve.

    Returns:
    - img_array: NumPy array representing the image.

    Example:
    >>> points = [(0, 0), (1, 1), (2, 4), (3, 9), (4, 16)]
    >>> image_array = get_image(points)
    """
    # Find the bounding box of the points
    #min_x, min_y = np.min(points, axis=0)
    #max_x, max_y = np.max(points, axis=0)
    min_x, min_y = 0, 0
    max_x, max_y = 360, 360

    # Create a blank image with a black background
    width = int(max_x - min_x) + 20
    height = int(max_y - min_y) + 20
    image = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(image)

    # Translate points to the image coordinates
    translated_points = [(x - min_x + 10, y - min_y + 10) for x, y in points]

    # Draw image
    if fill:
        # Fill inside the curve
        draw.polygon(translated_points, fill="white")
    else:
        # Draw the curve on the image
        draw.line(translated_points, fill="white", width=2)

    # Convert the PIL image to a NumPy array
    img_array = np.array(image)

    # Convert to grayscale
    img_array = np.mean(img_array, axis=2)

    # Resize image
    width, height = img_size
    img_array = cv2.resize(img_array, dsize=(width, height), interpolation=cv2.INTER_AREA)

    # Normalize pixel values
    img_array = img_array / img_array.max()

    return img_array


def get_notches(poly):
    """
    Determine the number of notches in a polygon object and calculate 
    normalized notches of polygon
    
    Based on: 
        "Measuring the Complexity of Polygonal Objects" 
        (Thomas Brinkhoff, Hans-Peter Kriegel, Ralf Schneider, Alexander Braun)
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.73.1045&rep=rep1&type=pdf
        
        https://github.com/pondrejk/PolygonComplexity/blob/master/PolygonComplexity.py
        
    @poly (Shapely Polygon object)
    
    Returns normalized notches
    """
    notches = 0 
    coords = list(poly.exterior.coords)
    for i, pt in enumerate(coords[:-1]):
        x_diff = coords[i+1][0] - pt[0]
        y_diff = coords[i+1][1] - pt[1]
        angle = math.atan2(y_diff, x_diff)
        if angle < 0: 
            angle += 2*math.pi
        if angle > math.pi:
            notches += 1
    
    if notches != 0:
        notches_norm = notches / (len(coords)-3)
    else:
        notches_norm = 0 
        
    return notches_norm

def get_stats(gdf, coeff_ampl, coeff_conv):
    """
    Get polygon's amplitude of vibration:
    
    ampl(pol) = (boundary(pol) - boundary(convexhull(pol))) / boundary(pol)
    
    Get deviation from convex hull:
    conv(pol) = (area(convexhull(pol)) - area(pol)) / area(convexhull(pol))
    
    Measure complexity
    
     Based on: 
        "Measuring the Complexity of Polygonal Objects" 
        (Thomas Brinkhoff, Hans-Peter Kriegel, Ralf Schneider, Alexander Braun)
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.73.1045&rep=rep1&type=pdf
        
        https://github.com/pondrejk/PolygonComplexity/blob/master/PolygonComplexity.py
    
    Get area, centroid, distance from each others, boudary, convex hull, 
    perimeter, number of vertices.
    
    Returns tuple with dict of stats values and GeoDataframe with stats
    """
    nb = gdf['geometry'].count()
    gdf['area'] = gdf['geometry'].area
    tot_area = gdf['area'].sum()
    gdf['centroid'] = gdf['geometry'].centroid
    gdf['boundary'] = gdf['geometry'].boundary
    gdf['convex_hull'] = gdf['geometry'].convex_hull
    gdf['convex_boundary'] = gdf['geometry'].convex_hull.boundary
    gdf['convex_area'] = gdf['geometry'].convex_hull.area
    gdf['nbvertices'] = gdf['geometry'].apply(lambda x: len(list(x.exterior.coords)))
    gdf['notches'] = gdf['geometry'].apply(lambda x: get_notches(x))
    
    gdf['amplitude'] = gdf.apply(
            lambda x:(
                    x['boundary'].length - x['convex_boundary'].length
                    ) / (x['boundary'].length + 1e-3), 
                    axis=1)
    gdf['convex'] = gdf.apply(
            lambda x: (
                    x['convex_area'] - x['area']
                    ) / (x['convex_area'] + 1e-3),
                    axis=1)
    gdf['complexity'] = gdf.apply(
            lambda x: coeff_ampl*x['amplitude'] * x['notches'] + coeff_conv * x['convex'],
            axis=1
            )
    
    mean_amplitude = gdf['amplitude'].mean()
    mean_convex = gdf['convex'].mean()
    mean_norm_notches = gdf['notches'].mean()
    mean_complexity = gdf['complexity'].mean()
    
    gdf['perimeter'] = gdf['geometry'].length
    tot_perimeter = gdf['perimeter'].sum()
    
    if ("lat" in gdf.columns) or ("lon" in gdf.columns):
        columns_drop = ["boundary", "convex_hull", "convex_boundary", "convex_area", "centroid", "lat", "lon"]
    else:
        columns_drop = ["boundary", "convex_hull", "convex_boundary", "convex_area", "centroid"]
    gdf = gdf.drop(columns_drop, axis=1)
    
    gdf = gdf.reset_index()
    
    if nb > 1:
        gdf = gdf.sort_values(by='perimeter', ascending=False)
        gdf = gdf.iloc[[0]]
    
    return {
        'area':tot_area,
        'perimeter':tot_perimeter,
        'amplitude': mean_amplitude,
        'convex': mean_convex,
        'notches': mean_norm_notches,
        'complexity': mean_complexity
    }, gdf
            
def complexity(points, coeff_ampl=0.8, coeff_conv=0.2):
    polygon = shapely.geometry.Polygon(points)
    gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries([polygon]))
    dict_complexity, gdf = get_stats(gdf, coeff_ampl, coeff_conv)        

    return dict_complexity