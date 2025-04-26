# adapted from geotiff_heights_to_png_heights.py

import argparse
import numpy as np
import os
import rasterio
from rasterio.enums import Resampling
from PIL import Image

# Function to convert GeoTIFF release cell data to RGBA PNG
def geotiff_to_rgba_png(input_geotiff, output_png):
    # Open the GeoTIFF file
    with rasterio.open(input_geotiff) as src:
        # Read the relese cell data from the first band
        release_cell_data = src.read(1, resampling=Resampling.nearest)
        is_release_cell = release_cell_data > 0

        # set red and alpha channels to 255 if release cells
        red_channel = np.zeros_like(is_release_cell, dtype=np.uint8)
        red_channel[is_release_cell] = 255
        red_channel = red_channel.astype(np.uint8)
        green_channel = np.zeros_like(red_channel, dtype=np.uint8)
        blue_channel = np.zeros_like(red_channel, dtype=np.uint8)
        alpha_channel = red_channel

        # Stack channels into a 4-channel image
        rgba_image = np.stack((red_channel, green_channel, blue_channel, alpha_channel), axis=-1)

        # Convert the array to a PIL Image
        image = Image.fromarray(rgba_image)

        # Save the image as a PNG
        image.save(output_png)


# Set up command-line argument parsing
def main():
    parser = argparse.ArgumentParser(description="Convert GeoTIFF release cell data to RGBA PNG.")
    parser.add_argument("input_geotiff", type=str, help="Path to the input GeoTIFF file, containing release cell data")
    parser.add_argument("output_png", type=str, nargs="?", default=None, help="Path to the output PNG file (optional)")

    args = parser.parse_args()

    # If output_png is not provided, create a default based on input_geotiff
    if args.output_png is None:
        # Change extension from .tif or .tiff to .png
        input_base = os.path.splitext(args.input_geotiff)[0]  # Remove the extension
        args.output_png = input_base + ".png"

    # Call the conversion function with the provided paths
    geotiff_to_rgba_png(args.input_geotiff, args.output_png)


if __name__ == "__main__":
    main()