# generated using chatgpt, i didnt safe the prompt unfortunately
# slightly modified

import argparse
import numpy as np
import os
import rasterio
from rasterio.enums import Resampling
from PIL import Image


# Function to convert GeoTIFF height data to RGBA PNG
def geotiff_to_rgba_png(input_geotiff, output_png):
    # Open the GeoTIFF file
    with rasterio.open(input_geotiff) as src:
        # Read the height data from the first band
        height_data = src.read(1, resampling=Resampling.nearest)

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
        image = Image.fromarray(rgba_image)

        # Save the image as a PNG
        image.save(output_png)


# Set up command-line argument parsing
def main():
    parser = argparse.ArgumentParser(description="Convert GeoTIFF height data to RGBA PNG.")
    parser.add_argument("input_geotiff", type=str, help="Path to the input GeoTIFF file, containing height data")
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