"""
Generator of simple 3D diagonal dataset with 2 binary, 1 continuous feature
"""

"""
Imports
"""
import numpy as np
import pandas as pd
import os
import csv
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.ticker import LinearLocator
plt.rcParams.update({'font.size': 10})
from true_explanations_search import minimize_closest_point_diagonal_plane, expf_diagonal_plane, plot_values
path = os.path.abspath('')
dataset_dir = str(path)+'/Datasets/'
results_cf_plots_dir = str(path)+'/Results/plots/'

def save_plane_data(plane, name_str):
    """
    Stores diagonal plane dataset
    """
    with open(f'{dataset_dir}{name_str}/{name_str}.csv', 'w', newline='') as f:
        write = csv.writer(f)
        write.writerows(plane)

def labeler(f):
    """
    Function that labels a generated data point
    """
    if f > 0:
        f = 1
    else:
        f = 0
    return f

def plane_function(x):
    """
    Ground truth label function
    """
    if x[0] >= 0.95 or \
        x[0] >= 0.7 and x[0] <= 0.8 or \
        x[0] >= 0.45 and x[0] <= 0.55 or \
        x[0] >= 0.2 and x[0] <= 0.3 or \
        x[0] <= 0.05:
        fx = 2*x[0] + 4*x[2] - 3
    elif x[0] > 0.8 and x[0] < 0.95:
        fx = x[2] - 0.40
    elif x[0] > 0.55 and x[0] < 0.7:
        fx = x[2] - 0.475
    elif x[0] > 0.3 and x[0] < 0.45:
        fx = x[2] - 0.525
    elif x[0] > 0.05 and x[0] < 0.2:
        fx = x[2] - 0.65
    return fx

def diagonal_plane_1ord_2con(N):
    """
    Function that creates data points for the labeler (3D plane dataset with 2 ordinal and 1 binary feature)
    """
    synth_diagonal_plane = []
    binary_range = [0.0, 1.0]
    ordinal_range = [0.0, 0.25, 0.50, 0.75, 1.0]
    for i in range(N):
        x1 = np.random.choice(ordinal_range)
        x2 = np.random.uniform()
        x3 = np.random.uniform()
        x = [x1, x2, x3]
        fx = plane_function(x)
        label_x = labeler(fx)
        x.extend([label_x])
        synth_diagonal_plane.append(x)
    return np.array(synth_diagonal_plane)

def diagonal_plane_2ord_1con(N):
    """
    Function that creates data points for the labeler (3D plane dataset with 2 ordinal and 1 continuous feature)
    """
    synth_diagonal_plane = []
    ordinal_range = [0.0, 0.25, 0.50, 0.75, 1.0]
    for i in range(N):
        x1, x2 = np.random.choice(ordinal_range), np.random.choice(ordinal_range)
        x3 = np.random.uniform()
        x = [x1, x2, x3]
        fx = plane_function(x)
        label_x = labeler(fx)
        x.extend([label_x])
        synth_diagonal_plane.append(x)
    return np.array(synth_diagonal_plane)

def diagonal_plane_1bin_1ord_1con(N):
    """
    Function that creates data points for the labeler (3D plane dataset with 1 binary, 1 ordinal and 1 continuous feature)
    """
    synth_diagonal_plane = []
    binary_range = [0.0, 1.0]
    ordinal_range = [0.0, 0.25, 0.50, 0.75, 1.0]
    for i in range(N):
        x1 = np.random.choice(binary_range)
        x2 = np.random.choice(ordinal_range)
        x3 = np.random.uniform()
        x = [x1, x2, x3]
        fx = plane_function(x)
        label_x = labeler(fx)
        x.extend([label_x])
        synth_diagonal_plane.append(x)
    return np.array(synth_diagonal_plane)

def diagonal_plane_2bin_1con(N):
    """
    Function that creates data points for the labeler (3D plane dataset with 2 binary and 1 continuous feature)
    """
    synth_diagonal_plane = []
    for i in range(N):
        x1, x2 = np.random.randint(0,2), np.random.randint(0,2)
        x3 = np.random.uniform()
        x = [x1, x2, x3]
        fx = plane_function(x)
        label_x = labeler(fx)
        x.extend([label_x])
        synth_diagonal_plane.append(x)
    return np.array(synth_diagonal_plane)

def diagonal_plane_2ord_1bin(N):
    """
    Function that creates data points for the labeler (3D plane dataset with 2 ordinal and 1 binary feature)
    """
    synth_diagonal_plane = []
    binary_range = [0.0, 1.0]
    ordinal_range = [0.0, 0.25, 0.50, 0.75, 1.0]
    for i in range(N):
        x1 = np.random.choice(ordinal_range)
        x2 = np.random.choice(ordinal_range)
        x3 = np.random.choice(binary_range)
        x = [x1, x2, x3]
        fx = plane_function(x)
        label_x = labeler(fx)
        x.extend([label_x])
        synth_diagonal_plane.append(x)
    return np.array(synth_diagonal_plane)

def ground_truth_plane(ax):
    """
    Function that obtains the ground truth plane (the decision boundary has a total of 13 planes)
    """
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_locator(LinearLocator(5))
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.set_zlim(0.0, 1.0)
    ax.zaxis.set_major_locator(LinearLocator(5))
    
    # Plane 1
    x_points = np.array([[ 0.0, 0.0], [ 0.05, 0.05]])
    y_points = np.array([[ 0.0, 1.0], [ 0.0, 1.0]])
    z_points = np.array([[ 0.75, 0.75], [ 0.725, 0.725]])
    ax.plot_surface(x_points, y_points, z_points, color='darkgreen', alpha = 0.8)

    # Plane 2
    x_points = np.array([[ 0.05, 0.05], [ 0.05, 0.05]])
    y_points = np.array([[ 0.0, 1.0], [ 0.0, 1.0]])
    z_points = np.array([[ 0.725, 0.725], [ 0.65, 0.65]])
    ax.plot_surface(x_points, y_points, z_points, color='green', alpha = 0.5)

    # Plane 3
    x_points = np.array([[ 0.05, 0.05], [ 0.2, 0.2]])
    y_points = np.array([[ 0.0, 1.0], [ 0.0, 1.0]])
    z_points = np.array([[ 0.65, 0.65], [ 0.65, 0.65]])
    ax.plot_surface(x_points, y_points, z_points, color='green', alpha = 0.5)

    # Plane 4
    x_points = np.array([[ 0.2, 0.2], [ 0.3, 0.3]])
    y_points = np.array([[ 0.0, 1.0], [ 0.0, 1.0]])
    z_points = np.array([[ 0.65, 0.65], [ 0.6, 0.6]])
    ax.plot_surface(x_points, y_points, z_points, color='darkgreen', alpha = 0.8)

    # Plane 5
    x_points = np.array([[ 0.3, 0.3], [ 0.3, 0.3]])
    y_points = np.array([[ 0.0, 1.0], [ 0.0, 1.0]])
    z_points = np.array([[ 0.6, 0.6], [ 0.525, 0.525]])
    ax.plot_surface(x_points, y_points, z_points, color='green', alpha = 0.5)

    # Plane 6
    x_points = np.array([[ 0.3, 0.3], [ 0.45, 0.45]])
    y_points = np.array([[ 0.0, 1.0], [ 0.0, 1.0]])
    z_points = np.array([[ 0.525, 0.525], [ 0.525, 0.525]])
    ax.plot_surface(x_points, y_points, z_points, color='green', alpha = 0.5)

    # Plane 7
    x_points = np.array([[ 0.45, 0.45], [ 0.55, 0.55]])
    y_points = np.array([[ 0.0, 1.0], [ 0.0, 1.0]])
    z_points = np.array([[ 0.525, 0.525], [ 0.475, 0.475]])
    ax.plot_surface(x_points, y_points, z_points, color='darkgreen', alpha = 0.8)

    # Plane 8
    x_points = np.array([[ 0.55, 0.55], [ 0.7, 0.7]])
    y_points = np.array([[ 0.0, 1.0], [ 0.0, 1.0]])
    z_points = np.array([[ 0.475, 0.475], [ 0.475, 0.475]])
    ax.plot_surface(x_points, y_points, z_points, color='green', alpha = 0.5)

    # Plane 9
    x_points = np.array([[ 0.7, 0.7], [ 0.7, 0.7]])
    y_points = np.array([[ 0.0, 1.0], [ 0.0, 1.0]])
    z_points = np.array([[ 0.475, 0.475], [ 0.4, 0.4]])
    ax.plot_surface(x_points, y_points, z_points, color='green', alpha = 0.5)

    # Plane 10
    x_points = np.array([[ 0.7, 0.7], [ 0.8, 0.8]])
    y_points = np.array([[ 0.0, 1.0], [ 0.0, 1.0]])
    z_points = np.array([[ 0.4, 0.4], [ 0.35, 0.35]])
    ax.plot_surface(x_points, y_points, z_points, color='darkgreen', alpha = 0.8)

    # Plane 11
    x_points = np.array([[ 0.8, 0.8], [ 0.95, 0.95]])
    y_points = np.array([[ 0.0, 1.0], [ 0.0, 1.0]])
    z_points = np.array([[ 0.35, 0.35], [ 0.35, 0.35]])
    ax.plot_surface(x_points, y_points, z_points, color='green', alpha = 0.5)

    # Plane 12
    x_points = np.array([[ 0.95, 0.95], [ 0.95, 0.95]])
    y_points = np.array([[ 0.0, 1.0], [ 0.0, 1.0]])
    z_points = np.array([[ 0.35, 0.35], [ 0.275, 0.275]])
    ax.plot_surface(x_points, y_points, z_points, color='green', alpha = 0.5)

    # Plane 13
    x_points = np.array([[ 0.95, 0.95], [ 1.0, 1.0]])
    y_points = np.array([[ 0.0, 1.0], [ 0.0, 1.0]])
    z_points = np.array([[ 0.275, 0.275], [ 0.25, 0.25]])
    ax.plot_surface(x_points, y_points, z_points, color='darkgreen', alpha = 0.8)
    ax.set(xlabel='x', ylabel='y', zlabel='z')

    return ax

def plot_plane_1ord_2con():
    """
    Plot 2 binary 1 continuous feature plane model
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    x = [0.0, 0.25, 0.50, 0.75, 1.0]
    y0, y1 = 0, 1
    z0, z1 = 0, 1
    for i in x:
        x_points = np.array([[i, i], [i, i]])
        y_points = np.array([[y0, y1], [y0, y1]])
        z_points = np.array([[z0, z0], [z1, z1]])
        ax.plot_surface(x_points, y_points, z_points, color='black', alpha = 0.4)
    ax = ground_truth_plane(ax)
    # x_points = np.array([[x0, x0], [x1, x1]])
    # y_points = np.array([[y0, y1], [y0, y1]])
    # z_points = np.array([[0.75, 0.75], [0.25, 0.25]])
    # ax.set(xlabel='x', ylabel='y', zlabel='z')
    fig.tight_layout()
    plt.savefig(results_cf_plots_dir+'synth_diagonal_plane_1ord_2con_plot.pdf')

def plot_plane_2ord_1con():
    """
    Plot 2 binary 1 continuous feature plane model
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    x = [0.0, 0.25, 0.50, 0.75, 1.0]
    y = [0.0, 0.25, 0.50, 0.75, 1.0]
    z0, z1 = 0, 1
    for i in x:
        for j in y:
            ax.plot([i, i], [j, j], [z0, z1], color='black', marker='.')
    ax = ground_truth_plane(ax)
    # x_points = np.array([[x[0], x[0]], [x[-1], x[-1]]])
    # y_points = np.array([[y[0], y[-1]], [y[0], y[-1]]])
    # z_points = np.array([[0.75, 0.75], [0.25, 0.25]])
    # ax.plot_surface(x_points, y_points, z_points, color='green', alpha = 0.5)
    # ax.set(xlabel='x', ylabel='y', zlabel='z')
    fig.tight_layout()
    plt.savefig(results_cf_plots_dir+'synth_diagonal_plane_2ord_1con_plot.pdf')

def plot_plane_1bin_1ord_1con():
    """
    Plot 1 binary 1 ordinal and 1 continuous feature plane model
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    x = [0.0, 1.0]
    y = [0.0, 0.25, 0.50, 0.75, 1.0]
    z0, z1 = 0, 1
    for i in x:
        for j in y:
            ax.plot([i, i], [j, j], [z0, z1], color='black', marker='.')
    ax = ground_truth_plane(ax)
    # x_points = np.array([[x[0], x[0]], [x[-1], x[-1]]])
    # y_points = np.array([[y[0], y[-1]], [y[0], y[-1]]])
    # z_points = np.array([[0.75, 0.75], [0.25, 0.25]])
    # ax.plot_surface(x_points, y_points, z_points, color='green', alpha = 0.5)
    # ax.set(xlabel='x', ylabel='y', zlabel='z')
    fig.tight_layout()
    plt.savefig(results_cf_plots_dir+'synth_diagonal_plane_1bin_1ord_1con_plot.pdf')

def plot_plane_2bin_1con():
    """
    Plot 2 binary 1 continuous feature plane model
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    x0, x1 = 0, 1
    y0, y1 = 0, 1
    z0, z1 = 0, 1
    ax.plot([x0, x0], [y0, y0], [z0, z1], color='black',marker='.')
    ax.plot([x0, x0], [y1, y1], [z0, z1], color='black',marker='.')
    ax.plot([x1, x1], [y0, y0], [z0, z1], color='black',marker='.')
    ax.plot([x1, x1], [y1, y1], [z0, z1], color='black',marker='.')
    ax = ground_truth_plane(ax)
    # x_points = np.array([[x0, x0], [x1, x1]])
    # y_points = np.array([[y0, y1], [y0, y1]])
    # z_points = np.array([[0.75, 0.75], [0.25, 0.25]])
    # ax.plot_surface(x_points, y_points, z_points, color='green', alpha = 0.5)
    # ax.set(xlabel='x', ylabel='y', zlabel='z')
    fig.tight_layout()
    plt.savefig(results_cf_plots_dir+'synth_diagonal_plane_2bin_1con_plot.pdf')

def plot_plane_2ord_1bin():
    """
    Plot 2 binary 1 continuous feature plane model
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    x = [0.0, 0.25, 0.50, 0.75, 1.0]
    y = [0.0, 0.25, 0.50, 0.75, 1.0]
    z = [0, 1]
    for i in x:
        for j in y:
            for k in z:
                ax.plot([i, i], [j, j], [k, k], color='black', marker='.')
    ax = ground_truth_plane(ax)
    # x_points = np.array([[x[0], x[0]], [x[-1], x[-1]]])
    # y_points = np.array([[y[0], y[-1]], [y[0], y[-1]]])
    # z_points = np.array([[0.75, 0.75], [0.25, 0.25]])
    # ax.plot_surface(x_points, y_points, z_points, color='green', alpha = 0.5)
    # ax.set(xlabel='x', ylabel='y', zlabel='z')
    fig.tight_layout()
    plt.savefig(results_cf_plots_dir+'synth_diagonal_plane_2ord_1bin_plot.pdf')

# Plot decision boundary and feasible spaces
# plot_plane_1ord_2con()
# plot_plane_2ord_1con()
# plot_plane_2bin_1con()
# plot_plane_1bin_1ord_1con()
# plot_plane_2ord_1bin()

N = 2000
sample_diagonal_plane_1ord_2con = diagonal_plane_1ord_2con(N)
sample_diagonal_plane_2bin_1con = diagonal_plane_2bin_1con(N)
sample_diagonal_plane_2ord_1con = diagonal_plane_2ord_1con(N)
sample_diagonal_plane_1bin_1ord_1con = diagonal_plane_1bin_1ord_1con(N)
sample_diagonal_plane_2ord_1bin = diagonal_plane_2ord_1bin(N)
save_plane_data(sample_diagonal_plane_1ord_2con, 'synthetic_diagonal_plane_1ord_2con')
save_plane_data(sample_diagonal_plane_2bin_1con, 'synthetic_diagonal_plane_2bin_1con')
save_plane_data(sample_diagonal_plane_2ord_1con, 'synthetic_diagonal_plane_2ord_1con')
save_plane_data(sample_diagonal_plane_1bin_1ord_1con, 'synthetic_diagonal_plane_1bin_1ord_1con')
save_plane_data(sample_diagonal_plane_2ord_1bin, 'synthetic_diagonal_plane_2ord_1bin')