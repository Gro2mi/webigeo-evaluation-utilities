from typing import Callable
import math
import os

import numpy as np
from PIL import Image


def generate_heightmap(f: Callable[[float, float], float], width: int, height: int) -> np.ndarray:
    assert(width > 0)
    assert(height > 0)

    xs = np.linspace(0, 1, width)
    ys = np.linspace(0, 1, height)
    zs = np.array([f(x, y) for y in ys for x in xs]).reshape((height, width))
    return zs


def encode_heightmap(heights: np.ndarray) -> np.ndarray:
    normalized_heights = np.uint16(heights / 8191.875 * 65535.0) # maps [0, 8191.875] (in R) to [0, 2^16-1] (in N)
    zeros = np.zeros(heights.shape, dtype="uint8")
    ones = np.ones(heights.shape, dtype="uint8") * 255
    color_data = np.stack([np.uint8(normalized_heights >> 8), np.uint8(normalized_heights & 255), zeros, ones], axis=-1)
    return color_data


def write_image(color_data: np.ndarray, file_path: str):
    image = Image.fromarray(color_data, mode="RGBA")
    image.save(file_path)
    #image.show()


def generate_heightmap_file(f: Callable[[float, float], float], width: int, height: int, file_path: str):
    """
    Takes a function that generates a height field from a two-dimensional function
    and writes it to an image file with the given size to the given path.
    
    The texture contains four 8-bit channels (RGBA). Heights are encoded in the first two channels (RG) as 16-bit unsigned integers.
    The lowest possible valule (0) means a height of 0 meters. The heighest possible value (2^16-1=65535) means 8191.875 meters.
    Channel B is always 0 and channel A is always 255.
    
    :param f: function, taking two floats x, y (both in [0,1]) and returns float z (meters height, in [0,8191.875])
              called for each pixel of the texture to generate
    :param width: width of image in pixels
    :param height: height of image in pixels
    :param file_path: path to write image file to
    """

    heights = generate_heightmap(f, width, height)
    texture_data = encode_heightmap(heights)
    write_image(texture_data, file_path)


def generate_releasepoints_file(coords: list[tuple[int, int]], width: int, height: int, file_path: str):
    """
    Writes a release point image file with the given size to the given path.
    
    The texture contains four 8-bit channels (RGBA). If a given
    Pixels with the given coordinates are colored red.

    :param coords: list of integer xy pairs (x in [0,width) and y in [0,height) respectively)
    :param width: width of image in pixels
    :param height: height of image in pixels
    :param file_path: path to write image file to
    """
    assert(width > 0)
    assert(height > 0)
    
    image = Image.new(mode="RGBA", size=(width, height), color=(0, 0, 0, 0))
    for coord in coords:
        assert(coord[0] >= 0 and coord[0] < width)
        assert(coord[1] >= 0 and coord[1] < height)
        image.putpixel(xy=coord, value=(255, 0, 0, 255))
    image.save(file_path)


def write_aabb_file(min_x: float, min_y: float, max_x: float, max_y: float, file_path: str):
    """
    Writes aabb file with the given extent to the given location.
    All coordinates are in web-mercator projected space.
    """
    
    with open(file_path, "w") as aabb_file:
        aabb_file.write(f"{min_x}\r\n{min_y}\r\n{max_x}\r\n{max_y}\r\n")


def generate_example_heightmap():
    """Example of how to generate a heightmap using the provided functions."""

    def f(x: float, y: float) -> float:
        """
        Two-dimensional function defining a height map.
        When passed to generate_heightmap_file(...), is used to generate height value per pixel.

        :param x: x coord in [0,1]
        :param y: y coord in [0,1]
        :returns: height in meters [0,8191.875]
        """
        return 2000 * x + 2000 # linear slope from 2000 to 4000 meters
    
    # other examples for generator functions, just pass them into generate_heightmap
    #f = lambda x, y: (x - 0.5) ** 2 * 500
    #f = lambda x, y: math.sin(x * 2 * math.pi) * 400 + 2000

    generate_heightmap_file(f, width=512, height=256, file_path="example_heights.png")


def generate_flowpy_parabolic_open_slope():
    """
    Generates height map, release points and aabb file for FlowPy's "parabolic, open slope" example (section 3.1)
    """

    # two-dim function for generating the "parabolic, open slope" surface from flowpy paper, section 3.1
    def flowpy_parabolic_open_slope(x: float, y: float) -> float:
        x_meters = x * 5000 # transform from [0,1] to [0,5000]
        if x_meters > 2250:
            return 0
        else:
            return 2/10125 * x_meters ** 2 - 8/9 * x_meters + 1000

    width = 500
    height = 150
    base_file_path = "example_flowpy_parabolic_open_slope"
    
    os.makedirs(base_file_path, exist_ok=True)

    # generate heightmap
    generate_heightmap_file(flowpy_parabolic_open_slope,
                            width, height,
                            file_path=f"{base_file_path}/heights.png")
    
    # generate release points
    generate_releasepoints_file([(0, int(height / 2) - 1), (0, int(height / 2)), (0, int(height / 2) + 1)],
                                width, height,
                                file_path=f"{base_file_path}/rp.png")
    
    # generate aabb file
    #  for slope calculation, heights are corrected according to web mercator projection (i.e. multiplied by 1/cos(lat(y)))
    #  therefore, use region that is in austria (in regions of small scales such as ours it is basically be a constant factor anyway)
    write_aabb_file(min_x=1760000.0, min_y=6060000.0, max_x=1765000.0, max_y=6071500.0, file_path=f"{base_file_path}/aabb.txt")


def main():
    generate_example_heightmap()
    generate_flowpy_parabolic_open_slope()


if __name__ == "__main__":
    main()