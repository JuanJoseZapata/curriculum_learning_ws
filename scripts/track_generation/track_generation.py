import numpy as np
import cv2

import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

import alphashape

import torch


CHECKPOINTS = 12
SCALE       = 6.0
TRACK_RAD   = 900/SCALE
TRACK_TURN_RATE = 0.31
TRACK_DETAIL_STEP = 21/SCALE
PLAYFIELD   = 2000/SCALE # Game over boundary

IMG_SIZE = 64  # Pixels


def get_control_points_from_image(img):
    """Image must have uint8 values from 0 to 255"""
    
    # Sharpen image
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)
    img = np.where(img < 30, 0, img)

    y, x = np.where(img != 0)
    # Calculate center
    center = [int(x.mean()), int(y.mean())]
    points_2d = np.array([x,y]).T
    # Fill contour
    for point in points_2d:
        cv2.line(img, point, center, color=255, thickness=1)
    points_2d = np.array(np.where(img > 100)).T
    # Get alpha shape
    alpha_shape = alphashape.alphashape(points_2d, alpha=0.6)
    geom_type = alpha_shape.geom_type
    alpha = 0.6
    while geom_type == "MultiPolygon":
        alpha -= 0.01
        alpha_shape = alphashape.alphashape(points_2d, alpha=alpha)
        geom_type = alpha_shape.geom_type

    points = np.array([xy for xy in alpha_shape.exterior.coords])
    points = fit_spline(points, num_points=300)

    return points*350/IMG_SIZE


def fit_spline(points, num_points=200):
    # Linear length along the line:
    distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
    distance = np.insert(distance, 0, 0)/distance[-1]

    # Build a list of the spline function, one for each dimension:
    splines = [UnivariateSpline(distance, coords, k=5, s=10) for coords in points.T]

    # Computed the spline for the asked distances:
    alpha = np.linspace(0, 1, num_points)
    points_fitted = np.vstack([spl(alpha) for spl in splines]).T

    return points_fitted

def plot_track(track, checkpoints=None):
    # Plot track
    plt.plot(track[:,1], track[:,0], linewidth=9, color=[0.4, 0.4, 0.4])
    plt.plot([track[0,1], track[-1,1]], [track[0,0], track[-1,0]], linewidth=9, color=[0.4, 0.4, 0.4])

    # Plot checkpoints
    if checkpoints is not None:
        for i in range(len(checkpoints)):
            y, x = checkpoints[i,:]
            plt.plot(x, y, 'o', markersize=2, color="tab:orange")
            plt.text(x, y, str(i), fontsize=10, color="black")
    
    plt.xlim(-20,360)
    plt.ylim(-20,360)

    ax = plt.gca()
    ax.set_facecolor(np.array([102, 230, 102])/255.)


def generate_track(vae, vector=None):
    """Generate track from a latent space vector"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Reconstruct image from latent space vector
    if vector is None:
        vector = torch.randn(1, 24).to(device)
    recon_x = vae.decode(vector).to('cpu').detach().numpy().squeeze()
    #Get control points from image
    recon_x = (recon_x*255).astype(np.uint8)
    track = get_control_points_from_image(recon_x)

    return track