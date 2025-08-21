#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

import numpy as np
import numpy.typing as npt
import pyproj
from pyproj.enums import TransformDirection


class WorldCoordinate:
    """
    A world coordinate is a vector representing a position in 3D space. Note that this type is a simple value with
    3 components that does not distinguish between geodetic or other cartesian coordinate systems (e.g. geocentric
    Earth-Centered Earth-Fixed or coordinates based on a local tangent plane).
    """

    def __init__(self, coordinate: npt.ArrayLike = None) -> None:
        """
        Constructs a world coordinate from an x, y, z triple. The triple can be expressed as a List or any other
        structure that can be used to construct a Numpy array.

        :param coordinate: the x,y,z components

        :return: None
        """
        if coordinate is None:
            coordinate = [0.0, 0.0, 0.0]

        if len(coordinate) != 3:
            raise ValueError("WorldCoordinates must have 3 components (x,y,z)")

        self.coordinate = np.array(coordinate, dtype=np.float64)

    @property
    def x(self) -> float:
        return self.coordinate[0]

    @x.setter
    def x(self, value: float) -> None:
        self.coordinate[0] = value

    @property
    def y(self) -> float:
        return self.coordinate[1]

    @y.setter
    def y(self, value: float) -> None:
        self.coordinate[1] = value

    @property
    def z(self) -> float:
        return self.coordinate[2]

    @z.setter
    def z(self, value: float) -> None:
        self.coordinate[2] = value

    def __repr__(self):
        return f"WorldCoordinate(coordinate={np.array_repr(self.coordinate)})"


class GeodeticWorldCoordinate(WorldCoordinate):
    """
    A GeodeticWorldCoordinate is an WorldCoordinate where the x,y,z components can be interpreted as longitude,
    latitude, and elevation. It is important to note that longitude, and latitude are in radians while elevation
    is meters above the ellipsoid.

    This class uses a custom format specification for a geodetic coordinate uses % directives similar to datetime.
    These custom directives can be combined as needed with literal values to produce a wide
    range of output formats. For example, '%ld%lm%ls%lH%od%om%os%oH' will produce a ddmmssXdddmmssY formatted
    coordinate. The first half, ddmmssX, represents degrees, minutes, and seconds of latitude with X representing
    North or South (N for North, S for South). The second half, dddmmssY, represents degrees, minutes, and seconds
    of longitude with Y representing East or West (E for East, W for West), respectively.


    ========= ================================================ =====
    Directive Meaning                                          Notes
    ========= ================================================ =====
    %L        latitude in decimal radians                       1
    %l        latitude in decimal degrees                       1
    %ld       latitute degrees                                  2
    %lD       latitude degrees rounded down                     2
    %lm       latitude minutes
    %ls       latitude seconds
    %lh       latitude hemisphere (n or s)
    %lH       latitude hemisphere uppercase (N or S)
    %O        longitude in decimal radians                      1
    %o        longitude in decimal degrees                      1
    %od       longitude degrees                                 2
    %oD       longitude degrees rounded down                    2
    %om       longitude minutes
    %os       longitude seconds
    %oh       longitude hemisphere (e or w)
    %oH       longitude hemisphere uppercase (E or W)
    %E        elevation in meters
    %%        used to represent a literal % in the output
    ========= ================================================ =====

    Notes:

    #. Formatting in decimal degrees or radians will be signed values
    #. Formatting for the degrees, minutes, seconds will always be unsigned assuming hemisphere will be included
    #. Any unknown directives will be ignored
    """

    def __init__(self, coordinate: npt.ArrayLike = None) -> None:
        """
        Constructs a geodetic world coordinate from a longitude, latitude, elevation triple. The longitude and
        latitude components are in radians. The triple can be expressed as a List or any other structure that can
        be used to construct a Numpy array.

        :param coordinate: the longitude, latitude, elevation components

        :return: None
        """
        super().__init__(coordinate)

    @property
    def longitude(self) -> float:
        return self.x

    @longitude.setter
    def longitude(self, value: float) -> None:
        self.x = value

    @property
    def latitude(self) -> float:
        return self.y

    @latitude.setter
    def latitude(self, value: float) -> None:
        self.y = value

    @property
    def elevation(self) -> float:
        return self.z

    @elevation.setter
    def elevation(self, value: float) -> None:
        self.z = value

    def to_dms_string(self) -> str:
        """
        Outputs this coordinate in a format ddmmssXdddmmssY. The first half, ddmmssX, represents degrees, minutes, and
        seconds of latitude with X representing North or South (N for North, S for South). The second half, dddmmssY,
        represents degrees, minutes, and seconds of longitude with Y representing East or West (E for East, W for West),
        respectively.

        :return: the formatted coordinate string
        """
        return f"{self:%ld%lm%ls%lH%od%om%os%oH}"

    def __repr__(self):
        return f"GeodeticWorldCoordinate(coordinate={np.array_repr(self.coordinate)})"

    def __format__(self, format_spec: str) -> str:
        if format_spec is None or format_spec == "":
            format_spec = "%ld%lm%ls%lH %od%om%os%oH %E"

        lat_degrees = np.degrees(self.latitude)
        lD = int(np.abs(np.floor(lat_degrees)))
        lh = "N"
        if lat_degrees < 0:
            lat_degrees *= -1.0
            lh = "S"
        ld = int(lat_degrees)
        lm = int(round(lat_degrees - ld, 6) * 60)
        ls = int(round(lat_degrees - ld - lm / 60, 6) * 3600)

        lon_degrees = np.degrees(self.longitude)
        oD = int(np.abs(np.floor(lon_degrees)))
        oh = "E"
        if lon_degrees < 0:
            lon_degrees *= -1.0
            oh = "W"
        od = int(lon_degrees)
        om = int(round(lon_degrees - od, 6) * 60)
        os = int(round(lon_degrees - od - om / 60, 6) * 3600)

        result = []
        i = 0
        while i < len(format_spec):
            if format_spec[i] == "%" and (i + 1) < len(format_spec):
                i += 1
                directive = format_spec[i]
                if directive == "L":
                    result.append(str(self.latitude))
                elif directive == "O":
                    result.append(str(self.longitude))
                elif directive == "l":
                    if (i + 1) < len(format_spec) and format_spec[i + 1] in ["d", "D", "m", "s", "h", "H"]:
                        i += 1
                        part = format_spec[i]
                        if part == "d":
                            result.append(format(ld, "02d"))
                        elif part == "D":
                            result.append(format(lD, "02d"))
                        elif part == "m":
                            result.append(format(lm, "02d"))
                        elif part == "s":
                            result.append(format(ls, "02d"))
                        elif part == "h":
                            result.append(lh.lower())
                        else:
                            # part must equal 'H'
                            result.append(lh)
                    else:
                        result.append(str(lat_degrees))
                elif directive == "o":
                    if (i + 1) < len(format_spec) and format_spec[i + 1] in ["d", "D", "m", "s", "h", "H"]:
                        i += 1
                        part = format_spec[i]
                        if part == "d":
                            result.append(format(od, "03d"))
                        elif part == "D":
                            result.append(format(oD, "03d"))
                        elif part == "m":
                            result.append(format(om, "02d"))
                        elif part == "s":
                            result.append(format(os, "02d"))
                        elif part == "h":
                            result.append(oh.lower())
                        else:
                            # part must equal 'H'
                            result.append(oh)
                    else:
                        result.append(str(lon_degrees))
                elif directive == "E":
                    result.append(str(self.elevation))
                elif directive == "%":
                    result.append("%")
            else:
                result.append(format_spec[i])
            i += 1
        return "".join(result)


# These are common definitions of projections used by Pyproj. They are used when converting between an Earth Centered
# Earth Fixed (ECEF or geocentric) coordinate system that uses cartesian coordinates and a longitude, latitude based
# geographic coordinate system. Both of these systems use the WGS84 datum which is a widely used standard among our
# customers
ECEF_PROJ = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")
LLA_PROJ = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")
GEODETIC_TO_GEOCENTRIC_TRANSFORM = pyproj.Transformer.from_proj(LLA_PROJ, ECEF_PROJ)


def geocentric_to_geodetic(ecef_world_coordinate: WorldCoordinate) -> GeodeticWorldCoordinate:
    """
    Converts a ECEF world coordinate (x, y, z) in meters into a (longitude, latitude, elevation) geodetic coordinate
    with units of radians, radians, meters.

    :param ecef_world_coordinate: the geocentric coordinate

    :return: the geodetic coordinate
    """
    return GeodeticWorldCoordinate(
        GEODETIC_TO_GEOCENTRIC_TRANSFORM.transform(
            ecef_world_coordinate.x,
            ecef_world_coordinate.y,
            ecef_world_coordinate.z,
            radians=True,
            direction=TransformDirection.INVERSE,
        )
    )


def geodetic_to_geocentric(geodetic_coordinate: GeodeticWorldCoordinate) -> WorldCoordinate:
    """
    Converts a geodetic world coordinate (longitude, latitude, elevation) with units of radians, radians, meters into
    a ECEF / geocentric world coordinate (x, y, z) in meters.

    :param geodetic_coordinate: the geodetic coordinate

    :return: the geocentric coordinate
    """
    return WorldCoordinate(
        GEODETIC_TO_GEOCENTRIC_TRANSFORM.transform(
            geodetic_coordinate.longitude,
            geodetic_coordinate.latitude,
            geodetic_coordinate.elevation,
            radians=True,
            direction=TransformDirection.FORWARD,
        )
    )


class ImageCoordinate:
    """
    This image coordinate system convention is defined as follows. The upper left corner of the upper left pixel
    of the original full image has continuous image coordinates (pixel position) (r, c) = (0.0,0.0) , and the center
    of the upper left pixel has continuous image coordinates (r, c) = (0.5,0.5) . The first row of the original full
    image has discrete image row coordinate R = 0 , and corresponds to a range of continuous image row coordinates of
    r = [0,1) . The first column of the original full image has discrete image column coordinate C = 0 , and
    corresponds to a range of continuous image column coordinates of c = [0,1) . Thus, for example, continuous image
    coordinates (r, c) = (5.6,8.3) correspond to the sixth row and ninth column of the original full image, and
    discrete image coordinates (R,C) = (5,8).
    """

    def __init__(self, coordinate: npt.ArrayLike = None) -> None:
        """
        Constructs an image coordinate from an x, y tuple. The tuple can be expressed as a List or any other
        structure that can be used to construct a Numpy array.

        :param coordinate: the x, y components

        :return: None
        """
        if coordinate is None:
            coordinate = [0.0, 0.0]

        if len(coordinate) != 2:
            raise ValueError("ImageCoordinates must have 2 components (x,y)")

        self.coordinate = np.array(coordinate, dtype=np.float64)

    @property
    def c(self) -> float:
        return self.coordinate[0]

    @c.setter
    def c(self, value: float) -> None:
        self.coordinate[0] = value

    @property
    def r(self) -> float:
        return self.coordinate[1]

    @r.setter
    def r(self, value: float) -> None:
        self.coordinate[1] = value

    @property
    def x(self) -> float:
        return self.c

    @x.setter
    def x(self, value: float) -> None:
        self.c = value

    @property
    def y(self) -> float:
        return self.r

    @y.setter
    def y(self, value: float) -> None:
        self.r = value

    def __repr__(self):
        return f"ImageCoordinate(coordinate={np.array_repr(self.coordinate)})"
