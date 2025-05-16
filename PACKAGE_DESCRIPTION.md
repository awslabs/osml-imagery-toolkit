The OversightML Imagery Toolkit is a Python package that contains image processing and photogrammetry routines commonly
used during the analysis of imagery collected by satellites and unmanned aerial vehicles (UAVs). It builds upon GDAL
by providing additional support for images compliant with the National Imagery Transmission Format (NITF), Sensor
Independent Complex Data (SICD), and Sensor Independent Derived Data (SIDD) standards.

This library contains four core packages under the `aws.osml` namespace:
* **photogrammetry**: convert locations between the image (x, y) and geodetic (lon, lat, elev) coordinate systems
* **gdal**: utilities to work with datasets loaded by GDAL
* **image_processing**: common image manipulation routines
* **features**: common geospatial feature manipulation routines

## Documentation

* **APIs**: You can find the latest API documentation for the OSML Imagery Toolkit hosted [here](https://awslabs.github.io/osml-imagery-toolkit/).
  If you are working from the source code running `tox -e docs` will trigger the Sphinx documentation build.
* **Example Notebooks**: Example notebooks for some operations are in the `examples` directory

## Installation

This software is available through a Python Package Index.
If your environment has a distribution, you should be able to install it using pip:
```shell
pip install osml-imagery-toolkit[gdal]
```

If you are working from a source code, you can build and install the package from the root directory of the
distribution.
```shell
pip install .[gdal]
```
Note that GDAL is listed as an extra dependency for this package. This is done to facilitate environments that either
don't want to use GDAL or those that have their own custom installation steps for that library. Future versions of
this package will include image IO backbones that have fewer dependencies.

## Contributing
We welcome contributions and suggestions. If you would like to submit a pull request please do so using a GitHub pull
request [here](https://github.com/awslabs/osml-imagery-toolkit/pulls).
