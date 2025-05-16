# OversightML Imagery Toolkit
![Build Badge](https://github.com/awslabs/osml-imagery-toolkit/actions/workflows/build.yml/badge.svg)
![Python Badge](https://img.shields.io/badge/python-3.9%2C%203.10%2C%203.11%2C%203.12%2C%203.13-blue)
![GDAL Badge](https://img.shields.io/badge/gdal-3.7%2C%203.8%2C%203.9-blue)
![GitHub License](https://img.shields.io/github/license/awslabs/osml-imagery-toolkit?color=blue)
![PyPI - Version](https://img.shields.io/pypi/v/osml-imagery-toolkit)


The OversightML Imagery Toolkit is a Python package that contains image processing and photogrammetry routines commonly
used during the analysis of imagery collected by satellites and unmanned aerial vehicles (UAVs). It builds upon GDAL
by providing additional support for images compliant with the National Imagery Transmission Format (NITF), Sensor
Independent Complex Data (SICD), and Sensor Independent Derived Data (SIDD) standards.

This library contains four core packages under the `aws.osml` namespace:
* **photogrammetry**: convert locations between the image (x, y) and geodetic (lon, lat, elev) coordinate systems
* **gdal**: utilities to work with datasets loaded by GDAL
* **image_processing**: image manipulation routines including tiling, orthoimage projections, and range adjustments
* **features**: geospatial feature indexing and property management routines

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
Note that GDAL is currently required but it is listed as an extra dependency for this package. This is done to facilitate
environments that either don't want to use GDAL or those that have their own custom installation steps for that library.
Future versions of this package will include image IO backbones that have fewer dependencies. Beware that GDAL has been
known to introduce breaking changes on minor version numbers so testing for specific version compatability is
recommended. The tox build has been setup to test multiple gdal/proj/python combinations and the versions
checked by automated testing can be seen in the environment-*.yml files.

## Contributing

This project welcomes contributions and suggestions. If you would like to submit a pull request, see our
[Contribution Guide](CONTRIBUTING.md) for more information.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file.
