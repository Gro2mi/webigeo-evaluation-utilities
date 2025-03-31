# Generation and conversion utilities for evaluating weBIGeo

We use Python scripts for

- generating synthetic data (in our internal format, as input to our simulations) as well as 
- convert between our internal format and GeoTIFF

## Setup
Only requirement is Python 3.11.

1. Navigate to the root directory of this repository
1. Create a new virtual environment in the current directory named `venv`: `python -m venv venv`
1. Install required packages into our virtual environment: `./venv/Scripts/pip install -r requirements.txt`
1. Done

Of course, you also need a working build of the weBIGeo native version, see [our setup instructions](https://github.com/weBIGeo/webigeo/blob/develop/docs/Setup.md#building-the-native-version) (use `develop` branch).

## Generating synthetic data

Generating synthetic test data can be done using `generate_heightmap.py`.

Execute the script using our local virtual environment

```
./venv/Scripts/python generate_heightmap.py
```

This generates an example height map file `example_heights.png` as well as a directory `example_flowpy_parabolic_open_slope`.
You can see how the example file is generated in function `generate_example_heightmap`.
First, define a function that maps from $[0,1]^2$ (texture coordinates) to $[0,8191.875]$ (height in meters).
Then pass this function to `generate_heightmap_file`.

The directory `example_flowpy_parabolic_open_slope` contains generated heights, release points and
aabb region file for the parabolic, open slope example from the FlowPy paper (section 3.1).
Have a look at function `generate_flowpy_parabolic_open_slope` to see how these are generated.
Generally, for generating AABB files or release points you can also use other tools.
AABB files are just text files and can be written with any text editor.
Release point textures can be created using any image manipulation tool (e.g. GIMP).
All pixels that are non-transparent are treated as release points.

## Converting between GeoTIFF and our format (.png + AABB)

TODO

## Using weBIGeo client for evaluation

We added a new pipeline called `Avalanche trajectories (eval)` that allows users to specify files for heights, release points and region aabb (via upload buttons).
You can try it out using the files generated in `example_flowpy_parabolic_open_slope` (see Generating synthetic data).
I suggest to use resolution `1x` for quicker results / better comparability to FlowPy.

Also for both pipelines `Avalanche trajectories` and `Avalanche trajectories (eval)`, heights, release points, simulation results and region aabb are written to files.
These files can be found in the subdirectory `export` of the build directory.
Usually the build directory is located in the repository root and called something like `build-renderer-desktop_qt_6_7_2_msvc2022-Release`.