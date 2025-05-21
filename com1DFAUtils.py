from avaframe.in2Trans.rasterUtils import readRaster
from pathlib import Path
import os
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Polygon
import json
import numpy as np
import hashlib
import subprocess

from matplotlib.path import Path as mplPath
from matplotlib.colors import ListedColormap
from avaframe.in2Trans.shpConversion import SHP2Array
from matplotlib import colors
import pandas as pd
import pickle

def get_dem(path: Path) -> tuple[dict, np.ndarray]:
    files = [file for file in os.listdir(path) if file.endswith(".asc")]
    assert len(files) == 1, "There should be exactly one .asc file with the DEM in the directory"
    dem_file = path / files[0]
    dem = readRaster(dem_file, noDataToNan=True)
    return dem["header"], dem["rasterData"]

def get_release_areas(filename: Path) -> list[Polygon]:
    release = SHP2Array(filename)
    release_areas = []
    for start, length in zip(release["Start"], release["Length"]):
        start = round(start)
        length = round(length)
        x_meters = release["x"][start:start + length]
        y_meters = release["y"][start:start + length]
        # x_pixel = [round((value - dem_header["xllcenter"]) / dem_header["cellsize"]) for value in x_meters]
        # y_pixel = [round((value - dem_header["yllcenter"]) / dem_header["cellsize"]) for value in y_meters]
        release_areas.append(Polygon(list(zip(x_meters, y_meters)), closed=True, edgecolor='red', facecolor='pink', alpha=0.8))
    return release_areas

def dem_np_to_webigeo(path: Path, dem_header: dict, height_data: np.ndarray) -> np.ndarray:
    min_height = height_data.min()
    if min_height < 0:
        height_data = height_data - min_height
    height_data = height_data + 500
    # Normalize the height data to range between 0 and 8191.875
    min_height = 0
    max_height = 8191.875

    # Scale height values to 16-bit unsigned integer range (0 to 65535)
    scaled_heights = np.clip(height_data, min_height, max_height)
    scaled_heights = (scaled_heights / max_height) * 65535
    scaled_heights = scaled_heights.astype(np.uint16)

    # Extract the higher 8 bits for the Red channel and the lower 8 bits for the Green channel
    red_channel = ((scaled_heights >> 8) & 0xFF).astype(np.uint8)   # High 8 bits
    green_channel = (scaled_heights & 0xFF).astype(np.uint8)         # Low 8 bits

    # Create the RGBA image: 
    # Red (R), Green (G), Blue (B), and Alpha (A)
    blue_channel = np.zeros_like(red_channel, dtype=np.uint8)
    alpha_channel = np.full_like(red_channel, 255, dtype=np.uint8)  # Fully opaque alpha

    # Stack channels into a 4-channel image
    rgba_image = np.stack((red_channel, green_channel, blue_channel, alpha_channel), axis=-1)

    # Convert the array to a PIL Image
    image = Image.fromarray(rgba_image).transpose(Image.FLIP_TOP_BOTTOM)

    # Save the image as a PNG
    image.save(path / "texture.png", "PNG")
    # "Bounds file must contain exactly 4 lines: min_x, min_y, max_x, max_y")
    x_min = dem_header["xllcenter"]
    y_min = dem_header["yllcenter"]
    x_max = dem_header["xllcenter"] + dem_header["ncols"] * dem_header["cellsize"]
    y_max = dem_header["yllcenter"] + dem_header["nrows"] * dem_header["cellsize"]

    with open(path / "aabb.txt", "w") as bounds_file:
        bounds_file.write(f"{x_min}\n")
        bounds_file.write(f"{y_min}\n")
        bounds_file.write(f"{x_max}\n")
        bounds_file.write(f"{y_max}\n")
    return height_data

def export_webigeo_release_points(path: Path, xx: np.ndarray, yy: np.ndarray, release_areas: list[Polygon]) -> Image:
    points = np.vstack((xx.ravel(), yy.ravel())).T
    grid = np.zeros(xx.shape, dtype=bool)
    # Combine masks for all polygons
    for polygon in release_areas:  
        mplpath = mplPath(polygon.get_xy())# Create a Path object for the polygon
        mask = mplpath.contains_points(points).reshape(grid.shape)  # Create a mask for the polygon
        grid[mask] = True  # Combine the mask into the grid
    red_channel = np.zeros(xx.shape, dtype=np.uint8)
    red_channel[grid] = 255  # High 8 bits
    green_channel = np.zeros_like(red_channel, dtype=np.uint8) 
    blue_channel = np.zeros_like(red_channel, dtype=np.uint8)
    alpha_channel = red_channel # Fully opaque alpha

    # Stack channels into a 4-channel image
    rgba_image = np.stack((red_channel, green_channel, blue_channel, alpha_channel), axis=-1)
    image = Image.fromarray(rgba_image).transpose(Image.FLIP_TOP_BOTTOM)
    print("DEM:", xx.shape, len(xx.flatten()))
    print("Release pixels:", np.sum(grid))
    image.save(path / "texture.png", "PNG")
    return image

def hash_dict(d):
    # Convert the dictionary to a sorted tuple of key-value pairs
    dict_tuple = tuple(sorted(d.items()))
    
    # Create a hash using hashlib (e.g., SHA-256)
    hash_object = hashlib.sha256(str(dict_tuple).encode())
    return hash_object.hexdigest()

friction_models = {
    "samosat": 3,
    "coulomb": 0,
    "voellmy": 1,
    "voellmyminshear": 2,
    "none": 4,
    }

def export_webigeo_settings(input_dir: Path, output_dir: Path, persistence: float=0.7, random_contribution: float=0.1, 
                            alpha:float =25, num_paths_per_release_cell: int=4096, num_steps: int=2048, 
                            density=200, slab_thickness=.5, friction_coeff=.155, drag_coeff=4000, friction_model="samosat", model=1) -> dict:
    settings = {
        "alpha": alpha,
        "num_paths_per_release_cell": num_paths_per_release_cell,
        "num_steps": num_steps,
        "persistence_contribution": persistence,
        "random_contribution": random_contribution,
        "release_point_interval": 8,
        "source_zoomlevel": 15,
        "tile_source": "dtm",
        "trajectory_resolution_multiplier": 1,
        "trigger_point_max_slope_angle": 45,
        "trigger_point_min_slope_angle": 30,
        "aabb_file_path": str(input_dir / "heights" / "aabb.txt"),
        "heightmap_texture_path": str(input_dir / "heights" / "texture.png"),
        "release_points_texture_path": str(input_dir / "release_points" / "texture.png"),
        "output_dir_path": "output",
        
        "density": density,
        "drag_coeff": drag_coeff,
        "friction_coeff": friction_coeff,
        "friction_model": friction_models[friction_model],
        "model_type": model,
        "slab_thickness": slab_thickness,
    }
    hash = hash_dict(settings)
    if model == 0:
        identifier = f"{hash[:6]}_pers_{persistence:.2f}_rand_{random_contribution:.2f}_n_{num_paths_per_release_cell}_s_{num_steps}_a_{alpha}"
    if model == 1:
        identifier = f"{hash[:6]}_fricmodel_{friction_model}_mu_{friction_coeff}_cd_{drag_coeff}_rho_{density}_h_{slab_thickness}_n_{num_paths_per_release_cell}_s_{num_steps}_a_{alpha}"
    settings["output_dir_path"] = str(output_dir / identifier)
    settings_path = input_dir / f"settings_{identifier}.json"
    json.dump(settings, open(settings_path, "w"), indent=4)
    return settings

def plot_inputs(xx: np.ndarray, yy: np.ndarray, dem: np.ndarray, release_areas: list[Polygon]):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_aspect('equal')
    plot_dem(ax, dem, xx, yy)
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Visualization of DEM and release areas')

    for polygon in release_areas:
        new_polygon = Polygon(polygon.get_xy(), closed=True, edgecolor='red', facecolor='pink', alpha=0.5)
        ax.add_patch(new_polygon)
    return fig, ax

def plot_dem_3d(xx, yy, dem):# Plot the DEM in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(xx, yy, dem, cmap='Reds', edgecolor='none')

    # Add a colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Elevation (m)')

    # Set labels and title
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('Elevation (m)')
    ax.set_title('3D Visualization of DEM')
    ax.set_aspect('equal')
    return fig, ax

def plot_dem(ax, dem, xx, yy, dark=True):
    # DEM
    cmap_contours = "Greys_r" if dark else "Greys"
    color_lines = "white" if dark else "black"
    levels_dem = np.arange(0, 4000, 200)
    ax.contourf(xx, yy, dem, levels=levels_dem, cmap=cmap_contours)
    CS = ax.contour(xx, yy, dem, levels=levels_dem, linewidths=.5, colors=color_lines)
    ax.clabel(CS, fontsize=10)

def get_levels(data, step=10):
    return np.arange(0.01, (int(data[np.isfinite(data)].max()/10) + 2) * 10, step)

def plot_flow_velocity(flow_velocity, dem, xx, yy, title="Flow Velocity", step=10):
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    plot_dem(ax, dem, xx, yy)
    surf = ax.contourf(xx, yy, flow_velocity, cmap='viridis', levels=get_levels(flow_velocity, step), vmin=0.01)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust size and padding
    cbar = fig.colorbar(surf, ax=ax, ticks=[round(tick) for tick in get_levels(flow_velocity, step)], cax=cax)
    cbar.set_label("Flow Velocity (m/s)")
    ax.set(title=title)
    return fig, ax

def create_file_hash(file_path, hash_length=10):
    # Open the file in binary mode and read its contents
    with open(file_path, "rb") as file:
        file_data = file.read()

    # Create a hash of the file's contents using SHA-256
    hash_object = hashlib.sha256(file_data)
    hash_hex = hash_object.hexdigest()  # Get the hexadecimal representation of the hash

    # Truncate the hash to the desired length
    return hash_hex[:hash_length]

def read_webigeo_flow_velocity(filename):
    FLOAT_MIN_ENCODING = -10000.0
    FLOAT_MAX_ENCODING = 10000.0
    UINT32_MAX = np.iinfo(np.uint32).max

    # Load the image
    img = Image.open(filename).convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
    rgba = np.array(img).astype(np.uint32)

    # Reconstruct the packed 32-bit values
    mapped_pixel = (
        (rgba[..., 0] << 24) |
        (rgba[..., 1] << 16) |
        (rgba[..., 2] << 8) |
        rgba[..., 3]
    )

    # Convert back to float using inverse of the encoding formula
    normalized_pixel = mapped_pixel / UINT32_MAX
    clamped_pixel = normalized_pixel * (FLOAT_MAX_ENCODING - FLOAT_MIN_ENCODING) + FLOAT_MIN_ENCODING

    return np.nan_to_num(clamped_pixel)

def read_webigeo_flow_velocity_from_z_delta(filename):
    FLOAT_MIN_ENCODING = -10000.0
    FLOAT_MAX_ENCODING = 10000.0
    UINT32_MAX = np.iinfo(np.uint32).max

    # Load the image
    img = Image.open(filename).convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
    rgba = np.array(img).astype(np.uint32)

    # Reconstruct the packed 32-bit values
    mapped_pixel = (
        (rgba[..., 0] << 24) |
        (rgba[..., 1] << 16) |
        (rgba[..., 2] << 8) |
        rgba[..., 3]
    )

    # Convert back to float using inverse of the encoding formula
    normalized_pixel = mapped_pixel / UINT32_MAX
    clamped_pixel = normalized_pixel * (FLOAT_MAX_ENCODING - FLOAT_MIN_ENCODING) + FLOAT_MIN_ENCODING

    return np.nan_to_num(np.sqrt(clamped_pixel * 2 * 9.81))

    
def read_webigeo_flow_velocity_float_from_z_delta(filename):
    # Load the image
    img = Image.open(filename).convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
    
    # Convert the image to a NumPy array
    rgba_data = np.array(img, dtype=np.uint8)
    
    # Extract RGBA channels
    r = rgba_data[:, :, 0].astype(np.uint32)  # Red channel
    g = rgba_data[:, :, 1].astype(np.uint32)  # Green channel
    b = rgba_data[:, :, 2].astype(np.uint32)  # Blue channel
    a = rgba_data[:, :, 3].astype(np.uint32)  # Alpha channel
    
    # Combine RGBA channels into a single 32-bit integer
    combined = (a << 24) | (b << 16) | (g << 8) | r
    
    # Convert the 32-bit integer to float
    float_data = combined.view(np.float32)


    return np.nan_to_num(np.sqrt(float_data * 2 * 9.81))

def read_webigeo_flow_velocity_float(filename):
    # Load the image
    img = Image.open(filename).convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
    
    # Convert the image to a NumPy array
    rgba_data = np.array(img, dtype=np.uint8)
    
    # Extract RGBA channels
    r = rgba_data[:, :, 0].astype(np.uint32)  # Red channel
    g = rgba_data[:, :, 1].astype(np.uint32)  # Green channel
    b = rgba_data[:, :, 2].astype(np.uint32)  # Blue channel
    a = rgba_data[:, :, 3].astype(np.uint32)  # Alpha channel
    
    # Combine RGBA channels into a single 32-bit integer
    combined = (a << 24) | (b << 16) | (g << 8) | r
    
    # Convert the 32-bit integer to float
    float_data = combined.view(np.float32)


    return np.nan_to_num(float_data)

def plot_flow_velocity_diff(flow_velocity, dfa_flow_velocity, dem, xx, yy):
    velocity_diff = flow_velocity - dfa_flow_velocity
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_dem(ax, dem, xx, yy)
    cont = ax.contourf(xx, yy, velocity_diff, cmap="bwr", alpha=0.7, levels=100, norm=colors.CenteredNorm())
    fig.colorbar(cont, ax=ax)
    ax.set_aspect('equal')
    ax.set(title="Difference between webigeo and dfa flow velocity")
    return fig, ax