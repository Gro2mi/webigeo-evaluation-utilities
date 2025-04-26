# generated using chatgpt

import rasterio
from pyproj import CRS, Transformer
import argparse
import os

# Function to read the extent of a TIFF file
def get_extent(tiff_file):
    with rasterio.open(tiff_file) as src:
        # Get the bounds of the TIFF file (minx, miny, maxx, maxy)
        bounds = src.bounds
        crs = src.crs  # Get the CRS of the TIFF file
    return bounds, crs

# Function to convert a bounding box to Web Mercator (EPSG:3857)
def convert_to_web_mercator(bounds, src_crs):
    # Define the target CRS (Web Mercator)
    target_crs = CRS.from_epsg(3857)  # Web Mercator (EPSG:3857)

    # Initialize transformer to convert from source CRS to target CRS
    transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)

    # Extract the coordinates (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = bounds

    # Transform the coordinates to Web Mercator
    minx_web, miny_web = transformer.transform(minx, miny)
    maxx_web, maxy_web = transformer.transform(maxx, maxy)

    return (minx_web, miny_web, maxx_web, maxy_web)

# Function to save the extent as a text file
def save_extent_to_file(extent, output_file):
    with open(output_file, 'w') as aabb_file:
        aabb_file.write(f"{extent[0]}\n{extent[1]}\n{extent[2]}\n{extent[3]}\n")
    print(f"Extent saved to {output_file}")

def main(tiff_file, output_file):
    # Get the extent and CRS from the TIFF
    extent, crs = get_extent(tiff_file)

    # Convert the extent to Web Mercator
    web_mercator_extent = convert_to_web_mercator(extent, crs)

    # Save the Web Mercator extent to a file
    save_extent_to_file(web_mercator_extent, output_file)

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Process a TIFF file to get its extent in Web Mercator")
    
    # Add arguments for input TIFF file and output text file
    parser.add_argument('tiff_file', type=str, help="Path to the input TIFF file")
    parser.add_argument('output_file', type=str, nargs='?', help="Path to the output text file")
    
    # Parse the arguments
    args = parser.parse_args()

    # If output file is not provided, create one by replacing the extension of the input TIFF path
    if not args.output_file:
        base, _ = os.path.splitext(args.tiff_file)
        args.output_file = base + '.txt'

    # Run the script
    main(args.tiff_file, args.output_file)
