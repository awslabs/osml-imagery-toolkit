#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
#  Copyright 2025-2025 General Atomics Integrated Intelligence, Inc.

import logging
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from . import ElevationModel
from .coordinates import (
    GeodeticWorldCoordinate,
    ImageCoordinate,
    WorldCoordinate,
    geocentric_to_geodetic,
    geodetic_to_geocentric,
)
from .sensor_model import SensorModel

logger = logging.getLogger(__name__)


class Polynomial2D:
    """
    This class contains coefficients for a two-dimensional polynomial.
    """

    def __init__(self, coef: npt.ArrayLike):
        """
        Constructor that takes the coefficients of the polynomial. The coefficients should be ordered so that the
        coefficient of the term of multi-degree i,j is contained in coef[i,j].

        :param coef: array-like structure of coefficients
        """
        self.coef = np.array(coef)
        if len(self.coef.shape) != 2:
            raise ValueError(
                f"Coefficients for class Polynomial2D must be two-dimensional. "
                f"Received numpy.ndarray of shape {self.coef.shape}"
            )

    def __call__(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> np.ndarray:
        """
        Invoke NumPy's polyval2d given the inputs and the coefficients of the polynomial.

        :param x: the first input parameter
        :param y: the second input parameter
        :return: the values of the 2-d polynomial at points formed with pairs of corresponding values from x and y.
        """
        return np.polynomial.polynomial.polyval2d(x, y, self.coef)


class PolynomialXYZ:
    """
    This class is an aggregation 3 one-dimensional polynomials all with the same input variable. The result of
    evaluating this class on the input variable is an [x, y, z] vector.
    """

    def __init__(
        self,
        x_polynomial: np.polynomial.Polynomial,
        y_polynomial: np.polynomial.Polynomial,
        z_polynomial: np.polynomial.Polynomial,
    ):
        """
        Constructor that accepts the 3 NumPy 1-d polynomials one for each component.

        :param x_polynomial: polynomial for the x component
        :param y_polynomial: polynomial for the y component
        :param z_polynomial: polynomial for the z component
        """
        self.x_polynomial = x_polynomial
        self.y_polynomial = y_polynomial
        self.z_polynomial = z_polynomial

    def __call__(self, t: float) -> np.ndarray:
        """
        Evaluate the x, y, and z polynomials at t and return the result as a vector.

        :param t: the value
        :return: the polynomial result
        """
        x = self.x_polynomial(t)
        y = self.y_polynomial(t)
        z = self.z_polynomial(t)

        return np.array([x, y, z], dtype=x.dtype)

    def deriv(self, m: int = 1):
        """
        Create a new PolynomialXYZ that is the derivative of the current PolynomialXYZ.

        :param m: find the derivative of order m
        :return: the new polynomial derivative
        """
        x_derivative = self.x_polynomial.deriv(m=m)
        y_derivative = self.y_polynomial.deriv(m=m)
        z_derivative = self.z_polynomial.deriv(m=m)

        return PolynomialXYZ(x_polynomial=x_derivative, y_polynomial=y_derivative, z_polynomial=z_derivative)


class SARImageCoordConverter:
    """
    This class contains image grid and image plane coordinate conversions for a provided set of SICD parameters. The
    equations are mostly defined in Section 2 of the SICD Standard Volume 3.
    """

    def __init__(
        self,
        scp_pixel: ImageCoordinate,
        scp_ecf: WorldCoordinate,
        u_row: np.ndarray,
        u_col: np.ndarray,
        row_ss: float,
        col_ss: float,
        first_pixel: ImageCoordinate = ImageCoordinate([0.0, 0.0]),
    ):
        """
        Construct the coordinate converter given parameters from the metadata. The names of these parameters have been
        chosen to align with the names in the specification.

        :param scp_pixel: location of the scene center point (SCP) in the global pixel grid
        :param scp_ecf: location of the scene center point (SCP) in earth centered fixed (ECF) coordinates
        :param u_row: unit vector in the increasing row direction in ECF coordinates.
        :param u_col: unit vector in the increasing column direction in ECF coordinates.
        :param row_ss: sample spacing in the row direction
        :param col_ss: sample spacing in the column direction
        :param first_pixel: location of the first row/column of the pixel array. For a full image array [0, 0]
        """
        self.scp_pixel = scp_pixel
        self.scp_ecf = scp_ecf
        self.row_ss = row_ss
        self.col_ss = col_ss
        self.u_row = u_row
        self.u_col = u_col
        self.first_pixel = first_pixel
        # Section 2.4 calculation of the image plane unit normal vector
        ipn = np.cross(self.u_row, self.u_col)
        self.uvect_ipn = ipn / np.linalg.norm(ipn)

        # Section 2.4 calculation of transform from ipp to xrow, ycol
        cos_theta = np.dot(self.u_row, self.u_col)
        sin_theta = np.sqrt(1 - cos_theta * cos_theta)
        ipp_transform = np.array([[1, -cos_theta], [-cos_theta, 1]], dtype="float64") / (sin_theta * sin_theta)
        row_col_transform = np.zeros((3, 2), dtype="float64")
        row_col_transform[:, 0] = self.u_row
        row_col_transform[:, 1] = self.u_col
        self.matrix_transform = np.dot(row_col_transform, ipp_transform)

    def rowcol_to_xrowycol(self, row_col: np.ndarray) -> np.ndarray:
        """
        This function converts the row and column indexes (row, col) in the global image grid to SCP centered
        image coordinates (xrow, ycol) using equations (2) (3) in Section 2.2 of the SICD Specification
        Volume 3.

        :param row_col: the [row, col] location as an array
        :return: the [xrow, ycol] location as an array
        """
        xrow_ycol = np.zeros(2, dtype="float64")
        xrow_ycol[0] = (row_col[0] - self.scp_pixel.r) * self.row_ss
        xrow_ycol[1] = (row_col[1] - self.scp_pixel.c) * self.col_ss
        return xrow_ycol

    def xrowycol_to_rowcol(self, xrow_ycol: np.ndarray) -> np.ndarray:
        """
        This function converts the SCP centered image coordinates (xrow, ycol) to row and column indexes (row, col)
        in the global image grid using equations (2) (3) in Section 2.2 of the SICD Specification Volume 3.

        :param xrow_ycol: the [xrow, ycol] location as an array
        :return: the [row, col] location as an array
        """
        row_col = np.zeros(2, dtype="float64")
        row_col[0] = xrow_ycol[0] / self.row_ss + self.scp_pixel.r
        row_col[1] = xrow_ycol[1] / self.col_ss + self.scp_pixel.c
        return row_col

    def xrowycol_to_ipp(self, xrow_ycol: np.ndarray) -> np.ndarray:
        """
        This function converts SCP centered image coordinates (xrow, ycol) to a ECF coordinate, image plane point (IPP),
        on the image plane using equations in Section 2.4 of the SICD Specification Volume 3.

        :param xrow_ycol: the [xrow, ycol] location as an array
        :return: the image plane point [x, y, z] ECF location on the image plane
        """
        delta_ipp = xrow_ycol[0] * self.u_row + xrow_ycol[1] * self.u_col
        return self.scp_ecf.coordinate + delta_ipp

    def ipp_to_xrowycol(self, ipp: np.ndarray) -> np.ndarray:
        """
        This function converts an ECF location on the image plane into SCP centered image coordinates (xrow, ycol)
        using equations in Section 2.4 of the SICD Specification volume 3.

        :param ipp: the image plane point [x, y, z] ECF location on the image plane
        :return: the [xrow, ycol] location as an array
        """
        delta_ipp = ipp - self.scp_ecf.coordinate
        xrow_ycol = np.dot(delta_ipp, self.matrix_transform)
        return xrow_ycol


class COAProjectionSet:
    """
    This is an abstract base class for R/Rdot projection contour computations described in Section 4 of the SICD
    Standard Volume 3.
    """

    def __init__(
        self,
        coa_time_poly: Polynomial2D,
        arp_poly: PolynomialXYZ,
        delta_arp: np.ndarray = np.array([0.0, 0.0, 0.0], dtype="float64"),
        delta_varp: np.ndarray = np.array([0.0, 0.0, 0.0], dtype="float64"),
        range_bias: float = 0.0,
    ):
        """
        Constructor with parameters supporting the calculations common to all R/Rdot projections (i.e. the
        calculations that do not depend on grid type and image formation algorithm).

        :param coa_time_poly: Center Of Aperture (COA) time polynomial.
        :param arp_poly: Aperture Reference Point (ARP) position polynomial coefficients.
        :param delta_arp: the ARP position offset
        :param delta_varp: the ARP velocity offset
        :param range_bias: the range bias offset
        """
        self.coa_time_poly = coa_time_poly
        self.arp_poly = arp_poly
        self.varp_poly = self.arp_poly.deriv(m=1)

        self.delta_arp = delta_arp
        self.delta_varp = delta_varp
        self.range_bias = float(range_bias)

    def precise_rrdot_computation(
        self, xrow_ycol: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This executes the precise image pixel grid location to R/Rdot projection. This function invokes
        the _grid_specific_projection() function implemented by subclasses which should handle the portions
        of the calculation that are dependent on the image grid and image formation algorithm.

        :param xrow_ycol: the [xrow, ycol] location as an array
        :return: the COA projection set { Rcoa, Rdotcoa, tcoa, arpcoa, varpcoa }
        """
        # These are the common calculations for image COA time (coa_time), COA ARP position and velocity
        # (arp_position and arp_velocity) as described in Section 2 of the SICD specification Volume 3.
        coa_time = self.coa_time_poly(xrow_ycol[0], xrow_ycol[1])
        arp_position = self.arp_poly(coa_time)
        arp_velocity = self.varp_poly(coa_time)

        # These are the image grid and image formation algorithm dependent calculations for the precise
        # computation of the R/Rdot contour. Each subclass should implement an approach as described in
        # sections 4.1 through 4.6 of the SICD specification Volume 3.
        r_tgt_coa, r_dot_tgt_coa = self._grid_specific_projection(xrow_ycol, coa_time, arp_position, arp_velocity)

        # If provided the Adjustable Parameter Offsets are incorporated into the computation from
        # image pixel location to COA projection parameters. See Section 8.1 of the SICD specification Volume 3.
        # TODO: Check this. This is the same approach as SarPy but I'm not 100% sure it is correct
        arp_position += self.delta_arp
        arp_velocity += self.delta_varp
        r_tgt_coa += self.range_bias

        return r_tgt_coa, r_dot_tgt_coa, coa_time, arp_position, arp_velocity

    @abstractmethod
    def _grid_specific_projection(self, xrow_ycol, coa_time, arp_position, arp_velocity) -> Tuple[np.ndarray, np.ndarray]:
        """
        The precise computation of the R/Rdot contour is dependent upon the image grid type and the image
        formation algorithm that produced the image but the computation of the COA time, ARP position, and
        velocity is the same for all image products.

        This abstract method should be overriden by subclasses to perform the R/Rdot calculations for a
        specific image grid and formation algorithm.

        :param xrow_ycol: the [xrow, ycol] location as an array
        :param coa_time: Center Of Aperture (COA) time
        :param arp_position: Aperture Reference Point (ARP) position
        :param arp_velocity: Aperture Reference Point (ARP) velocity
        :return: the tuple containing range and range rate relative to the ARP at COA time
        """


class PFAProjectionSet(COAProjectionSet):
    """
    This Center Of Aperture (COA) Projection set is to be used with a range azimuth image grid (RGAZIM) and polar
    formatted (PFA) phase history data. See section 4.1 of the SICD Specification Volume 3.
    """

    def __init__(
        self,
        scp_ecf: WorldCoordinate,
        polar_ang_poly,
        spatial_freq_sf_poly,
        coa_time_poly: Polynomial2D,
        arp_poly: PolynomialXYZ,
        delta_arp: np.ndarray = np.array([0.0, 0.0, 0.0], dtype="float64"),
        delta_varp: np.ndarray = np.array([0.0, 0.0, 0.0], dtype="float64"),
        range_bias: float = 0.0,
    ):
        """
        Constructor for this projection set.

        :param scp_ecf: Scene Center Point position in ECF coordinates
        :param polar_ang_poly: Polar Angle polynomial coefficients
        :param spatial_freq_sf_poly: Spatial Frequency Scale Factor polynomial coefficients
        :param coa_time_poly: Center Of Aperture (COA) time polynomial
        :param arp_poly: Aperture Reference Point (ARP) position polynomial coefficients
        :param delta_arp: the ARP position offset
        :param delta_varp: the ARP velocity offset
        :param range_bias: the range bias offset
        """
        super().__init__(coa_time_poly, arp_poly, delta_arp, delta_varp, range_bias)
        self.scp_ecf = scp_ecf
        self.polar_ang_poly = polar_ang_poly
        self.spatial_freq_sf_poly = spatial_freq_sf_poly
        self.polar_ang_poly_der = polar_ang_poly.deriv(m=1)
        self.spatial_freq_sf_poly_der = spatial_freq_sf_poly.deriv(m=1)

    def _grid_specific_projection(self, xrow_ycol, coa_time, arp_position, arp_velocity) -> Tuple[np.ndarray, np.ndarray]:
        """
        These are the calculations for the precise computation of the R/Rdot contour unique to these grid and
        image formation algorithm types. See SICD Volume 3 Section 4.1

        :param xrow_ycol: the [xrow, ycol] location as an array
        :param coa_time: Center Of Aperture (COA) time
        :param arp_position: Aperture Reference Point (ARP) position
        :param arp_velocity: Aperture Reference Point (ARP) velocity
        :return: the tuple containing range and range rate relative to the ARP at COA time
        """
        # For the RGAZIM grid, the image coordinates are range and azimuth. The row coordinate is the range
        # coordinate, xrow = rg. The column coordinate is the azimuth coordinate, ycol = az.
        rg = xrow_ycol[0]
        az = xrow_ycol[1]

        # (2) Compute the range and range rate to the SCP at the pixel COA time
        arp_minus_scp = arp_position - self.scp_ecf.coordinate
        range_to_scp = np.linalg.norm(arp_minus_scp, axis=-1)
        rdot_to_scp = np.sum(arp_velocity * arp_minus_scp, axis=-1) / range_to_scp

        # (3) Compute the polar angle (theta) and its derivative with respect to time (d_theta_d_time)
        # at the pixel COA time.
        theta = self.polar_ang_poly(coa_time)
        d_theta_d_time = self.polar_ang_poly_der(coa_time)

        # (4) Compute the polar aperture scale factor (KSF) and its derivative with respect to polar angle
        # (d_ksf_d_theta) at the pixel COA time
        ksf = self.spatial_freq_sf_poly(theta)
        d_ksf_d_theta = self.spatial_freq_sf_poly_der(theta)

        # (5) Compute the spatial frequency domain phase slopes in the radial (ka) and cross radial
        # (kc) directions (d_phi_d_ka and d_phi_d_kc) for the radial direction at theta. Note: The sign COA
        # parameter (SGN) for the phase may be ignored as it is cancelled in a subsequent computation.
        d_phi_d_ka = rg * np.cos(theta) + az * np.sin(theta)
        d_phi_d_kc = -rg * np.sin(theta) + az * np.cos(theta)

        # (6) Compute range relative to the SCP (delta_range) at the COA.
        delta_range = ksf * d_phi_d_ka

        # (7) Compute the derivative of the range relative to the SCP with respect to polar angle
        # (d_delta_range_d_theta) at the COA. Scale by the derivative of the polar angle with respect
        # to time to yield the derivative with respect to time (delta_r_dot_tgt_coa).
        d_delta_range_d_theta = d_ksf_d_theta * d_phi_d_ka + ksf * d_phi_d_kc
        delta_r_dot_tgt_coa = d_delta_range_d_theta * d_theta_d_time

        # (8) Compute the range and range rate relative to the ARP at COA ( r_tgt_coa and rdot_tgt_coa).
        # The projection to three-dimensional scene point for grid location (rgTGT, azTGT) is along this
        # R/Rdot contour.
        r_tgt_coa = range_to_scp + delta_range
        rdot_tgt_coa = rdot_to_scp + delta_r_dot_tgt_coa

        return r_tgt_coa, rdot_tgt_coa


class RGAZCOMPProjectionSet(COAProjectionSet):
    def __init__(
        self,
        scp_ecf: WorldCoordinate,
        az_scale_factor: float,
        coa_time_poly: Polynomial2D,
        arp_poly: PolynomialXYZ,
        delta_arp: np.ndarray = np.array([0.0, 0.0, 0.0], dtype="float64"),
        delta_varp: np.ndarray = np.array([0.0, 0.0, 0.0], dtype="float64"),
        range_bias: float = 0.0,
    ):
        """
        Constructor for this projection set.

        :param scp_ecf: Scene Center Point position in ECF coordinates
        :param az_scale_factor: Scale factor that converts azimuth coordinate to an increment in cosine of the DCA at COA
        :param coa_time_poly: Center Of Aperture (COA) time polynomial
        :param arp_poly: Aperture Reference Point (ARP) position polynomial coefficients
        :param delta_arp: the ARP position offset
        :param delta_varp: the ARP velocity offset
        :param range_bias: the range bias offset
        """
        super().__init__(coa_time_poly, arp_poly, delta_arp, delta_varp, range_bias)
        self.scp_ecf = scp_ecf
        self.az_scale_factor = az_scale_factor

    def _grid_specific_projection(self, xrow_ycol, coa_time, arp_position, arp_velocity) -> Tuple[np.ndarray, np.ndarray]:
        """
        These are the calculations for the precise computation of the R/Rdot contour unique to these grid and
        image formation algorithm types. See SICD Volume 3 Section 4.2

        :param xrow_ycol: the [xrow, ycol] location as an array
        :param coa_time: Center Of Aperture (COA) time
        :param arp_position: Aperture Reference Point (ARP) position
        :param arp_velocity: Aperture Reference Point (ARP) velocity
        :return: the tuple containing range and range rate relative to the ARP at COA time
        """
        # For the RGAZIM grid, the image coordinates are range and azimuth. The row coordinate is the range
        # coordinate, xrow = rg. The column coordinate is the azimuth coordinate, ycol = az.
        rg = xrow_ycol[0]
        az = xrow_ycol[1]

        # (2) Compute the range and range rate to the SCP at COA.
        arp_minus_scp = arp_position - self.scp_ecf.coordinate
        range_to_scp = np.linalg.norm(arp_minus_scp, axis=-1)
        rdot_to_scp = np.sum(arp_velocity * arp_minus_scp, axis=-1) / range_to_scp

        # (3) Compute the increment in cosine of the DCA at COA of the target (delta_cos_dca_tgt_coa) by
        # scaling the azimuth coordinate by the azimuth to DCA scale factor. Compute the increment
        # in range rate (delta_rdot_tgt_coa) by scaling by the magnitude of the velocity vector at COA.
        delta_cos_dca_tgt_coa = self.az_scale_factor * az
        delta_r_dot_tgt_coa = -np.linalg.norm(arp_velocity, axis=-1) * delta_cos_dca_tgt_coa

        # (4) Compute the range and range rate to the target at COA as follows.
        r_tgt_coa = range_to_scp + rg
        rdot_tgt_coa = rdot_to_scp + delta_r_dot_tgt_coa

        return r_tgt_coa, rdot_tgt_coa


class INCAProjectionSet(COAProjectionSet):
    def __init__(
        self,
        r_ca_scp: float,
        inca_time_coa_poly: np.polynomial.Polynomial,
        drate_sf_poly: Polynomial2D,
        coa_time_poly: Polynomial2D,
        arp_poly: PolynomialXYZ,
        delta_arp: np.ndarray = np.array([0.0, 0.0, 0.0], dtype="float64"),
        delta_varp: np.ndarray = np.array([0.0, 0.0, 0.0], dtype="float64"),
        range_bias: float = 0.0,
    ):
        """
        Constructor for this projection set.

        :param r_ca_scp: Range at Closest Approach for the SCP (m)
        :param inca_time_coa_poly: Time of Closest Approach polynomial coefficients
        :param drate_sf_poly: Doppler Rate Scale Factor polynomial coefficients
        :param coa_time_poly: Center Of Aperture (COA) time polynomial
        :param arp_poly: Aperture Reference Point (ARP) position polynomial coefficients
        :param delta_arp: the ARP position offset
        :param delta_varp: the ARP velocity offset
        :param range_bias: the range bias offset
        """
        super().__init__(coa_time_poly, arp_poly, delta_arp, delta_varp, range_bias)
        self.r_ca_scp = r_ca_scp
        self.inca_time_coa_poly = inca_time_coa_poly
        self.drate_sf_poly = drate_sf_poly

    def _grid_specific_projection(self, xrow_ycol, coa_time, arp_position, arp_velocity) -> Tuple[np.ndarray, np.ndarray]:
        """
        These are the calculations for the precise computation of the R/Rdot contour unique to these grid and
        image formation algorithm types. See SICD Volume 3 Section 4.3

        :param xrow_ycol: the [xrow, ycol] location as an array
        :param coa_time: Center Of Aperture (COA) time
        :param arp_position: Aperture Reference Point (ARP) position
        :param arp_velocity: Aperture Reference Point (ARP) velocity
        :return: the tuple containing range and range rate relative to the ARP at COA time
        """
        # For the RGZERO grid, the image coordinates are range and azimuth. The row coordinate is the range
        # coordinate, xrow = rg. The column coordinates is the azimuth coordinate, ycol = az.
        rg = xrow_ycol[0]
        az = xrow_ycol[1]

        # (2) Compute the range at closest approach and the time of closest approach for the image
        # grid location. The range at closest approach, R TGT , is computed from the range coordinate.
        # The time of closest approach, tTGT , is computed from the azimuth coordinate. CA
        range_ca_tgt = self.r_ca_scp + rg
        time_ca_tgt = self.inca_time_coa_poly(az)

        # (2 repeated in v1.3.0 of the spec) Compute the ARP velocity at the time of closest approach
        # and the magnitude of the vector.
        arp_velocity_ca_tgt = self.varp_poly(time_ca_tgt)
        mag_arp_velocity_ca_tgt = np.linalg.norm(arp_velocity_ca_tgt)

        # (3) Compute the Doppler Rate Scale Factor (drsf_tgt) for image grid location (rg, az).
        drsf_tgt = self.drate_sf_poly(rg, az)

        # (4) Compute the time difference between the COA time and the CA time (delta_coa_tgt).
        delta_coa_tgt = coa_time - time_ca_tgt

        # (5) Compute the range and range rate relative to the ARP at COA ( RTGT and RdotTGT ).
        r_tgt_coa = np.sqrt(range_ca_tgt**2 + drsf_tgt * mag_arp_velocity_ca_tgt**2 * delta_coa_tgt**2)
        r_dot_tgt_coa = (drsf_tgt / r_tgt_coa) * mag_arp_velocity_ca_tgt**2 * delta_coa_tgt

        return r_tgt_coa, r_dot_tgt_coa


class PlaneProjectionSet(COAProjectionSet):
    def __init__(
        self,
        scp_ecf: WorldCoordinate,
        image_plane_urow: np.ndarray,
        image_plane_ucol: np.ndarray,
        coa_time_poly: Polynomial2D,
        arp_poly: PolynomialXYZ,
        delta_arp: np.ndarray = np.array([0.0, 0.0, 0.0], dtype="float64"),
        delta_varp: np.ndarray = np.array([0.0, 0.0, 0.0], dtype="float64"),
        range_bias: float = 0.0,
    ):
        """
        Constructor for this projection set.

        :param scp_ecf: Scene Center Point position in ECF coordinates
        :param image_plane_urow: Unit vector in the increasing row direction in ECF coordinates (uRow)
        :param image_plane_ucol: Unit vector in the increasing column direction in ECF coordinates (uCol)
        :param coa_time_poly: Center Of Aperture (COA) time polynomial
        :param arp_poly: Aperture Reference Point (ARP) position polynomial coefficients
        :param delta_arp: the ARP position offset
        :param delta_varp: the ARP velocity offset
        :param range_bias: the range bias offset
        """
        super().__init__(coa_time_poly, arp_poly, delta_arp, delta_varp, range_bias)
        self.scp_ecf = scp_ecf
        self.image_plane_urow = image_plane_urow
        self.image_plane_ucol = image_plane_ucol

    def _grid_specific_projection(self, xrow_ycol, coa_time, arp_position, arp_velocity) -> Tuple[np.ndarray, np.ndarray]:
        """
        These are the calculations for the precise computation of the R/Rdot contour unique to these grid and
        image formation algorithm types. See SICD Volume 3 Sections 4.4, 4.5, and 4.6.

        Note that the calculations in sections 4.4, 4.5, and 4.6 are the same with the only difference being the
        interpretation of the xrow and ycol gird positions. To share this one implementation for all three grid
        planes assume: xrow = xrg = xct and ycol = ycr = yat

        :param xrow_ycol: the [xrow, ycol] location as an array
        :param coa_time: Center Of Aperture (COA) time
        :param arp_position: Aperture Reference Point (ARP) position
        :param arp_velocity: Aperture Reference Point (ARP) velocity
        :return: the tuple containing range and range rate relative to the ARP at COA time
        """

        # xrow = xrg = xct and ycol = ycr = yat

        # (2) The samples of an XRGYCR, XCTYAT, or PLANE grid are uniformly spaced locations in the image plane
        # formed by the SCP, and image plane vectors uRow and uCol. Vectors uRow and uCol are orthogonal. Compute
        # the point the image plane point for image grid location (xrgTGT, ycrTGT).
        image_plane_point = (
            self.scp_ecf.coordinate + xrow_ycol[0] * self.image_plane_urow + xrow_ycol[1] * self.image_plane_ucol
        )

        # (3) Compute the range and range rate relative to the ARP at COA (r_tgt_coa and rdot_tgt_coa) for image plane
        # point (image_plane_point).
        arp_minus_ipp = arp_position - image_plane_point
        r_tgt_coa = np.linalg.norm(arp_minus_ipp, axis=-1)
        rdot_tgt_coa = np.sum(arp_velocity * arp_minus_ipp, axis=-1) / r_tgt_coa

        return r_tgt_coa, rdot_tgt_coa


class RRDotSurfaceProjection:
    """
    This is the base class for calculations that project the R/RDot contour onto a surface model. The SICD specification
    defines a way to do this for planes, a surface of constant height above an ellipsoid, or a digital elevation model.
    """

    @abstractmethod
    def rrdot_to_ground(self, r_tgt_coa, r_dot_tgt_coa, arp_position, arp_velocity) -> np.ndarray:
        """
        Subclasses should implement this method to compute the R/RDot Contour Ground Plane intersection with a
        specific surface type (e.g. planar, HAE, DEM)

        :param r_tgt_coa: target COA range
        :param r_dot_tgt_coa: target COA range rate
        :param arp_position: ARP position
        :param arp_velocity: ARP velocity
        :return: the intersection between the R/Rdot Contour and the ground plane
        """


class GroundPlaneRRDotSurfaceProjection(RRDotSurfaceProjection):
    """
    This class implements the Precise R/RDot Ground Plane Projection described in Section 5 of the SICD Specification
    Volume 3 (v1.3.0).
    """

    class GroundPlaneNormalType(Enum):
        SPHERICAL_EARTH = "SPHERICAL_EARTH"
        GEODETIC_EARTH = "GEODETIC_EARTH"

    def __init__(
        self,
        ref_ecf: WorldCoordinate,
        gpn: Optional[np.ndarray],
        gpn_type: GroundPlaneNormalType = GroundPlaneNormalType.GEODETIC_EARTH,
    ):
        """
        The ground plane is defined by a reference point in the plane (ref_ect) and the vector normal to the plane
        (gpn). The reference point and plane orientation may be based upon specific terrain height and slope
        information for the imaged area. When only a reference point is specified, a ground plane normal may be
        derived assuming the plane is tangent to a spherical earth model or a surface of constant geodetic height
        above the WGS-84 reference ellipsoid passing through (ref_ect).

        :param ref_ecf: reference point in the plane, GREF in the specification
        :param gpn: optional vector normal to the ground plane; if missing it will be computed using gpn_type
        :param gpn_type: method to derive the ground plan normal
        """
        self.ref_ecf = ref_ecf

        if gpn is not None:
            self.u_gpn = gpn / np.linalg.norm(gpn)
        elif gpn_type == self.GroundPlaneNormalType.SPHERICAL_EARTH:
            self.u_gpn = ref_ecf.coordinate / np.linalg.norm(ref_ecf.coordinate)
        elif gpn_type == self.GroundPlaneNormalType.GEODETIC_EARTH:
            ref_lle = geocentric_to_geodetic(ref_ecf)
            self.u_gpn = np.array(
                [
                    np.cos(ref_lle.latitude) * np.cos(ref_lle.longitude),
                    np.cos(ref_lle.latitude) * np.sin(ref_lle.longitude),
                    np.sin(ref_lle.latitude),
                ]
            )
        else:
            raise ValueError(f"Provided gpn_type, {gpn_type}, is invalid.")

    def rrdot_to_ground(self, r_tgt_coa, r_dot_tgt_coa, arp_position, arp_velocity) -> np.ndarray:
        """
        This method implements the R/RDot Contour Ground Plane Intersection described in section 5.2

        :param r_tgt_coa: target COA range
        :param r_dot_tgt_coa: target COA range rate
        :param arp_position: ARP position
        :param arp_velocity: ARP velocity
        :return: the intersection between the R/Rdot Contour and the ground plane
        """
        # (1) Compute the unit vector in the +Z direction (normal to the ground plane).
        uvect_z = self.u_gpn / np.linalg.norm(self.u_gpn)

        # (2) Compute the ARP distance from the plane (arp_z). Also compute the ARP ground plane nadir (agpn).
        arp_z = np.sum((arp_position - self.ref_ecf.coordinate) * uvect_z, axis=-1)
        if arp_z > r_tgt_coa:
            raise ValueError("No solution exists. Distance between ARP and the plane is greater than range.")

        agpn = arp_position - np.outer(arp_z, uvect_z)

        # (3) Compute the ground plane distance (gp_distance) from the ARP nadir to the circle of constant range. Also
        # compute the sine and cosine of the grazing angle (sin_graz and cos_graz).
        gp_distance = np.sqrt(r_tgt_coa * r_tgt_coa - arp_z * arp_z)
        sin_graz = arp_z / r_tgt_coa
        cos_graz = gp_distance / r_tgt_coa

        # (4) Compute velocity components normal to the ground plane (v_z) and parallel to the ground plane (v_x).
        v_z = np.dot(arp_velocity, uvect_z)
        v_mag = np.linalg.norm(arp_velocity, axis=-1)
        v_x = np.sqrt(v_mag * v_mag - v_z * v_z)

        # (5) Orient the +X direction in the ground plane such that the v_x > 0. Compute unit vectors uvect_x
        # and uvect_y.
        uvect_x = (arp_velocity - (v_z * uvect_z)) / v_x
        uvect_y = np.cross(uvect_z, uvect_x)

        # (6) Compute the cosine of the azimuth angle to the ground plane point.
        cos_az = (-r_dot_tgt_coa + v_z * sin_graz) / (v_x * cos_graz)
        if np.abs(cos_az) > 1:
            raise ValueError("No solution exists. cos_az < -1 or cos_az > 1.")

        # (7) Compute the sine of the azimuth angle. Use parameter LOOK to establish the correct sign corresponding
        # to the correct Side of Track.
        look = np.sign(np.dot(np.cross(arp_position - self.ref_ecf.coordinate, arp_velocity), uvect_z))
        sin_az = look * np.sqrt(1 - cos_az * cos_az)

        # (8) Compute GPPTGT at distance G from the AGPN and at the correct azimuth angle.
        return agpn + uvect_x * (gp_distance * cos_az) + uvect_y * (gp_distance * sin_az)


class HAERRDotSurfaceProjection(RRDotSurfaceProjection):
    """
    This class implements the Precise R/RDot Height Above Ellipsioid (HAE) Projection described in Section 9 of the
    SICD Specification Volume 3 (v1.3.0).
    """

    def __init__(
        self,
        scp_ecf: WorldCoordinate,
        side_of_track: str,
        hae: float,
        height_threshold: float = 1.0,
        max_number_iterations: int = 3,
    ):
        """
        Constructor for the projection that takes the image parameters and a height above theWGS-84 ellipsoid.
        The parameter defaults are the recommended values for the user selectable parameters in section 9.2.

        :param scp_ecf: Scene Center Point position in ECF coordinates
        :param side_of_track: side of track imaged
        :param hae: the surface height (m) above the WGS-84 reference ellipsoid
        :param height_threshold: the height threshold for convergence of iterative projection sequence
        :param max_number_iterations: maximum number of iterations allowed
        """
        self.scp_ecf = scp_ecf
        self.scp_lle = geocentric_to_geodetic(scp_ecf)
        self.side_of_track = side_of_track
        self.hae = hae

        # Integer based on side of track parameter
        self.look = 1.0
        if side_of_track == "R":
            self.look = -1.0

        self.delta_hae_max = height_threshold
        self.nlim = max_number_iterations

    def rrdot_to_ground(self, r_tgt_coa, r_dot_tgt_coa, arp_position, arp_velocity) -> np.ndarray:
        """
        This method implements the R/RDot Contour Ground Plane Intersection described in section 9.2.

        The precise projection to a surface of constant height above the WGS-84 reference ellipsoid along an
        R/Rdot contour. The R/Rdot contour is relative to an ARP Center Of Aperture position and velocity. The
        algorithm computes the R/Rdot projection to one or more ground planes that are tangent to the constant
        height surface. Each ground plane projection point computed is slightly above the constant HAE surface.
        The final surface position is computed by projecting from the final ground plane projection point down
        to the HAE surface.

        :param r_tgt_coa: target COA range
        :param r_dot_tgt_coa: target COA range rate
        :param arp_position: ARP position
        :param arp_velocity: ARP velocity
        :return: the intersection between the R/Rdot Contour and the ground plane
        """
        # (1) Compute the geodetic ground plane normal at the SCP. Compute the parameters for the initial ground plane.
        # The reference point position is gref and the unit normal is u_gpn.
        u_gpn = np.array(
            [
                np.cos(self.scp_lle.latitude) * np.cos(self.scp_lle.longitude),
                np.cos(self.scp_lle.latitude) * np.sin(self.scp_lle.longitude),
                np.sin(self.scp_lle.latitude),
            ]
        )
        gref = self.scp_ecf.coordinate + (self.hae - self.scp_lle.z) * u_gpn

        cont = True
        n = 1
        while cont:
            # (2) Compute the precise projection along the R/Rdot contour to Ground Plane. The result is ground plane
            # point position gpp_ecf. Convert from ECF coordinates to geodetic coordinates (gpp_lle).
            gp_surface_projection = GroundPlaneRRDotSurfaceProjection(ref_ecf=WorldCoordinate(gref), gpn=u_gpn)
            gpp_ecf = gp_surface_projection.rrdot_to_ground(r_tgt_coa, r_dot_tgt_coa, arp_position, arp_velocity)[0]
            gpp_lle = geocentric_to_geodetic(WorldCoordinate(coordinate=gpp_ecf))

            # (3) Compute the unit vector in the increasing height direction at point gpp_lle, (u_up). Also
            # compute the height difference at point gpp_lle relative to the desired surface height (delta_hae).
            u_up = np.array(
                [
                    np.cos(gpp_lle.latitude) * np.cos(gpp_lle.longitude),
                    np.cos(gpp_lle.latitude) * np.sin(gpp_lle.longitude),
                    np.sin(gpp_lle.latitude),
                ]
            )
            delta_hae = gpp_lle.elevation - self.hae

            # (4) Test to see if the point is sufficiently close the surface or if the maximum number of iterations
            # has been reached.  Otherwise, compute a new ground reference point (gref) and unit normal (u_up); repeat
            # Steps 2, 3 and 4.
            if delta_hae <= self.delta_hae_max or n >= self.nlim:
                cont = False
            else:
                gref = gpp_ecf - delta_hae * u_up
                u_gpn = u_up
                n += 1

        # (5) Compute the unit slant plane normal vector, u_spn, that is tangent to the R/Rdot contour at point gpp.
        # Unit vector u_spn points away from the center of the earth and in a direction of increasing HAE at gpp.
        spn = np.cross(self.look * arp_velocity, gpp_ecf - arp_position)
        u_spn = spn / np.linalg.norm(spn)

        # (6) Compute the straight line projection from point gpp_ecf along the slant plane normal to point slp.
        # Point slp is very close to the precise R/Rdot contour intersection with the constant height surface.
        # Convert the position of point slp from ECF coordinates to geodetic coordinates (slp_lle).
        sf = np.dot(u_up, u_spn)
        slp = gpp_ecf - (delta_hae / sf) * u_spn
        slp_lle = geocentric_to_geodetic(WorldCoordinate(slp))

        # (7) Assign surface point spp position by adjusting the HAE to be on the desired surface. Convert from
        # geodetic coordinates to ECF coordinates.
        spp_lle = GeodeticWorldCoordinate([slp_lle.longitude, slp_lle.latitude, self.hae])
        spp_ecf = geodetic_to_geocentric(spp_lle)

        return np.array([spp_ecf.coordinate])


class DEMRRDotSurfaceProjection(RRDotSurfaceProjection):
    """
    This class implements the Precise R/RDot Height Above a Digital Elevation Model (DEM) Projection described in
    Section 10 of the SICD Specification Volume 3 (v1.3.0).
    """

    def __init__(
        self,
        scp_ecf: WorldCoordinate,
        side_of_track: str,
        elevation_model: ElevationModel,
        max_adjacent_point_distance: float = 10.0,
        height_threshold: float = 0.001,
        min_height: float = -4000.0,
        max_height: float = 9000.0,
        max_horizontal_distance: float = 15.0,
    ):
        """
        Constructor for the projection that takes the image parameters and a digital elevation model (DEM).
        The parameter defaults are the recommended values for the user selectable parameters in section 10.3.

        :param scp_ecf: Scene Center Point position in ECF coordinates
        :param side_of_track: side of track imaged
        :param elevation_model: the digital elevation model
        :param max_adjacent_point_distance: Maximum distance between adjacent points along the R/Rdot contour
        :param height_threshold: threshold for determining if a R/Rdot contour point is on the DEM surface (m)
        :param min_height: Minimum height above ellipsoid in meters.
        :param max_height: Maximum height above ellipsoid in meters.
        :param max_horizontal_distance: Maximum horizontal distance between adjacent points along the R/Rdot contour
        """
        self.scp_ecf = scp_ecf
        self.scp_lle = geocentric_to_geodetic(scp_ecf)
        self.side_of_track = side_of_track
        self.elevation_model = elevation_model

        # Integer based on Side of Track parameter.
        self.look = 1.0
        if side_of_track == "R":
            self.look = -1.0

        self.hae_min = min_height
        self.hae_max = max_height
        self.hae_max_surface_projection = HAERRDotSurfaceProjection(
            scp_ecf=scp_ecf, side_of_track=side_of_track, hae=self.hae_max
        )
        self.hae_min_surface_projection = HAERRDotSurfaceProjection(
            scp_ecf=scp_ecf, side_of_track=side_of_track, hae=self.hae_min
        )
        self.delta_dist_dem = max_horizontal_distance

        self.delta_dist_rrc = max_adjacent_point_distance
        self.delta_hd_lim = height_threshold

    def rrdot_to_ground(self, r_tgt_coa, r_dot_tgt_coa, arp_position, arp_velocity) -> np.ndarray:
        """
        This method implements the R/RDot Contour Ground Plane Intersection described in section 10.3

        The R/Rdot contour is relative to an ARP Center Of Aperture position and velocity. The earth surface is
        described by a Digital Elevation Model (DEM) that defines a unique surface height as a function of two
        horizontal coordinates. The projection computation may yield one or more surface points that lie along
        the R/Rdot contour.

        :param r_tgt_coa: target COA range
        :param r_dot_tgt_coa: target COA range rate
        :param arp_position: ARP position
        :param arp_velocity: ARP velocity
        :return: the intersection between the R/Rdot Contour and the DEM, if multiple intersections occur they will
                 be returned in order of increasing height above the WGS-84 ellipsoid.
        """

        # (1) Compute the center point (ctr) and the radius of the R/Rdot projection contour (rrrc).
        v_mag = np.linalg.norm(arp_velocity)
        u_vel = arp_velocity / v_mag
        cos_dca = -1.0 * r_dot_tgt_coa / v_mag
        sin_dca = np.sqrt(1 - cos_dca * cos_dca)
        ctr = arp_position + r_tgt_coa * cos_dca * u_vel
        rrrc = r_tgt_coa * sin_dca

        # (2) Compute the unit vectors u_rrx and u_rry to be used to compute points located on the R/Rdot contour.
        dec_arp = np.linalg.norm(arp_position)
        u_up = arp_position / dec_arp
        rry = np.cross(u_up, u_vel)
        u_rry = rry / np.linalg.norm(rry)
        u_rrx = np.cross(u_rry, u_vel)

        # (3) Compute the projection along the R/Rdot contour to the surface of constant HAE at height hae_max.
        # The projection point at height hae_max is point_a. Also compute the cosine and sine of the contour angle
        # to point_a, cos_caa and sin_caa.
        point_a = self.hae_max_surface_projection.rrdot_to_ground(r_tgt_coa, r_dot_tgt_coa, arp_position, arp_velocity)[0]
        cos_caa = np.dot(point_a - ctr, u_rrx) / rrrc
        # This variable is defined in the specification but it does not appear to be used anywhere
        # sin_caa = self.look * np.sqrt(1 - cos_caa * cos_caa)

        # (4) Compute the projection along the R/Rdot contour to the surface of constant HAE at height hae_min.
        # The projection point at height hae_min is point_b. Also compute the cosine and sine of the contour angle
        # to point_b, cos_cab and sin_cab.
        point_b = self.hae_min_surface_projection.rrdot_to_ground(r_tgt_coa, r_dot_tgt_coa, arp_position, arp_velocity)[0]
        cos_cab = np.dot(point_b - ctr, u_rrx) / rrrc
        sin_cab = self.look * np.sqrt(1 - cos_cab * cos_cab)

        # (5) A set of points along the R/Rdot contour are to be computed. The points will be spaced in equal
        # increments of the cosine of the contour angle. Compute the step size, delta_cos_ca.

        # (5.1) Step size delta_cos_rrc is computed such that the distance between adjacent points on the R/Rdot
        # contour is approximately equal to delta_dist_rrc.
        delta_cos_rrc = self.delta_dist_rrc * np.abs(sin_cab) / rrrc

        # (5.2) Step size delta_cos_dem is computed such that the horizontal distance between adjacent points on
        # the R/Rdot contour is approximately equal to delta_dist_dem.
        delta_cos_dem = self.delta_dist_dem * (np.abs(sin_cab) / cos_cab) / rrrc

        # (5.3) Set delta_cos_ca (Note the value of delta_cos_ca is < 0)
        delta_cos_ca = -1.0 * min(delta_cos_rrc, delta_cos_dem)

        # (6) Determine the number of points along the R/Rdot contour to be computed, npts.
        npts = int(np.floor((cos_caa - cos_cab) / delta_cos_ca)) + 2

        # (7) Compute the set of points along the R/Rdot contour, {Pn} for n =0, 2, ..., npts-1. Initial point P1 is
        # located on the hae_min surface. The final point is located above the hae_max surface. Point Pn is computed
        # in ECF coordinates. Note that here n ranges from [0, npts-1] while in the specification n is [1, npts].
        # Equations have been modified accordingly.
        points_ecf = []
        for n in range(0, npts):
            cos_can = cos_cab + n * delta_cos_ca  # n-1 is unnecessary since n is zero based here
            sin_can = self.look * np.sqrt(1 - cos_can * cos_can)
            pn = ctr + rrrc * (cos_can * u_rrx + sin_can * u_rry)
            points_ecf.append(pn)

        # (8 - 10) For each of the NPTS points, convert from ECF coordinates to DEM coordinates (lon, lat, ele). Also
        # compute the DEM surface height for the point with DEM horizontal coordinates (lon, lat). Compute the
        # difference in height (delta_height) between the point on the contour and DEM surface point. Also,
        # set an indicator to track if that point is ABOVE, ON, or BELOW the DEM surface.
        #
        # Contour points that are within delta_hd_lim of the surface point are considered to be “on” the surface and
        # will be added to the result set. Also compute a result point when points n and n+1 when both are “off”
        # the surface and the R/Rdot contour intersects the surface between them (i.e. indicator n-1 x indicator n = -1)
        #
        # All height coordinates are in meters.
        intersection_points = []
        prev_indicator = None
        prev_delta_height = None
        for n in range(0, npts):
            point_lle = geocentric_to_geodetic(WorldCoordinate(points_ecf[n]))
            point_dem = GeodeticWorldCoordinate(point_lle.coordinate.copy())
            self.elevation_model.set_elevation(point_dem)
            delta_height = point_lle.elevation - point_dem.elevation

            # Determine if the contour point is ABOVE (indicator = 1), ON (indicator = 0), or BELOW (indicator = -1)
            if np.abs(delta_height) < self.delta_hd_lim:
                # Contour point is on the DEM surface, add it to the result set
                indicator = 0
                intersection_points.append(points_ecf[n])
            elif delta_height > self.delta_hd_lim:
                # Contour point is above the DEM surface
                indicator = 1
            else:
                # Contour point is below the DEM surface
                indicator = -1

            # If in two adjacent points one is ABOVE and the other BELOW then the contour intersected the DEM
            # surface between the two points. Interpolate an intersection point and add it to the result.
            if prev_indicator and prev_indicator * indicator == -1:
                # contour crossed between two points; compute the surface point between
                frac = prev_delta_height / (prev_delta_height - delta_height)
                cos_cas = cos_cab + (n - 1 + frac) * delta_cos_ca  # here the -1 is necessary for previous point
                sin_cas = self.look * np.sqrt(1 - cos_cas * cos_cas)
                sm = ctr + rrrc * (cos_cas * u_rrx + sin_cas * u_rry)
                intersection_points.append(sm)

            # Keep track of the current indicator and delta height for comparison to the next point
            prev_indicator = indicator
            prev_delta_height = delta_height

        # Return the set of surface points found. Points will be ordered of increasing height above the WGS-84
        # ellipsoid.
        return np.array(intersection_points)


class SICDSensorModel(SensorModel):
    """
    This is an implementation of the SICD sensor model as described by SICD Volume 3 Image Projections Description
    NGA.STND.0024-3_1.3.0 (2021-11-30)
    """

    def __init__(
        self,
        coord_converter: SARImageCoordConverter,
        coa_projection_set: COAProjectionSet,
        u_spn: np.ndarray,
        side_of_track: str = "L",
        u_gpn: Optional[np.ndarray] = None,
    ):
        """
        Constructs a SICD sensor model from the information derived from the XML metadata.

        :param coord_converter: converts coordinates between image grid and image plane
        :param coa_projection_set: projects image locations to the r/rdot contour
        :param u_spn: slant plane normal
        :param side_of_track: side of track imaged, "L" or "R", default "L"
        :param u_gpn: optional unit normal for ground plane
        """
        super().__init__()
        self.coa_projection_set = coa_projection_set
        self.coord_converter = coord_converter
        self.uvect_gpn = u_gpn
        self.uvect_spn = u_spn
        self.side_of_track = side_of_track

        self.default_surface_projection = GroundPlaneRRDotSurfaceProjection(self.coord_converter.scp_ecf, self.uvect_gpn)

    @staticmethod
    def compute_u_spn(scp_ecf: WorldCoordinate, scp_arp: np.ndarray, scp_varp: np.ndarray, side_of_track: str) -> np.ndarray:
        """
        This helper function computes the slant plane normal.

        :param scp_ecf: Scene Center Point position in ECF coordinates
        :param scp_arp: aperture reference point position
        :param scp_varp: aperture reference point velocity
        :param side_of_track: side of track imaged
        :return: unit vector for the slant plane normal
        """
        u_spn = np.cross(scp_varp, scp_ecf.coordinate - scp_arp)
        if side_of_track == "R":
            u_spn *= -1.0
        u_spn /= np.linalg.norm(u_spn)
        return u_spn

    @staticmethod
    def compute_u_gpn(scp_ecf: WorldCoordinate, u_row: np.ndarray, u_col: np.ndarray) -> np.ndarray:
        u_gpn = np.cross(u_row, u_col)
        u_gpn /= np.linalg.norm(u_gpn)
        if np.dot(u_gpn, scp_ecf.coordinate) < 0:
            u_gpn *= -1
        return u_gpn

    def image_to_world(
        self,
        image_coordinate: ImageCoordinate,
        elevation_model: Optional[ElevationModel] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> GeodeticWorldCoordinate:
        """
        This is an implementation of an Image Grid to Scene point projection that first projects the image
        location to the R/RDot contour and then intersects the R/RDot contour with the elevation model. If
        an elevation model is provided then this routine intersects the R/Rdot contour with the DEM surface which
        may result in multiple solutions. In that case the solution with the lowest HAE is returned.

        :param image_coordinate: the x,y image coordinate
        :param elevation_model: the optional elevation model, if none supplied a plane tangent to SCP is assumed
        :param options: no additional options are supported at this time
        :return: the lon, lat, elev geodetic coordinate of the surface matching the image coordinate
        """
        row_col = np.array(
            [
                image_coordinate.r + self.coord_converter.first_pixel.r,
                image_coordinate.c + self.coord_converter.first_pixel.c,
            ]
        )
        xrow_ycol = self.coord_converter.rowcol_to_xrowycol(row_col=row_col)
        r_tgt_coa, r_dot_tgt_coa, time_coa, arp_coa, varp_coa = self.coa_projection_set.precise_rrdot_computation(xrow_ycol)

        if elevation_model is not None:
            scp_lle = geocentric_to_geodetic(self.coord_converter.scp_ecf)
            elevation_summary = elevation_model.describe_region(scp_lle)
            if elevation_summary is not None:
                extra_kwargs = {
                    "min_height": elevation_summary.min_elevation,
                    "max_height": elevation_summary.max_elevation,
                    "max_horizontal_distance": 0.5 * elevation_summary.post_spacing,
                }
            else:
                extra_kwargs = {}
            surface_projection = DEMRRDotSurfaceProjection(
                self.coord_converter.scp_ecf,
                self.side_of_track,
                elevation_model=elevation_model,
                **extra_kwargs,
            )
        else:
            surface_projection = self.default_surface_projection

        # Note that for a DEM the r/rdot contour may intersect the surface at multiple locations
        # resulting in an ambiguous location. Here we are arbitrarily selecting the first result.
        # TODO: Is there a better way to handle multiple DEM intersections?
        coords_ecf = surface_projection.rrdot_to_ground(r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa)

        return geocentric_to_geodetic(WorldCoordinate(coords_ecf[0]))

    def world_to_image(self, world_coordinate: GeodeticWorldCoordinate) -> ImageCoordinate:
        """
        This is an implementation of Section 6.1 Scene To Image Grid Projection for a single point.

        :param world_coordinate: lon, lat, elevation coordinate of the scene point
        :return: the x,y pixel location in this image
        """
        ecf_world_coordinate = geodetic_to_geocentric(world_coordinate)

        # TODO: Consider making these options like we have for image_to_world
        tolerance = 1e-2
        max_iterations = 10

        # (2) Ground plane points are projected along straight lines to the image plane to establish points.
        # The GP to IP projection direction is along the SCP COA slant plane normal. Also, compute the image
        # plane unit normal, uIPN. Compute projection scale factor SF as shown.
        uvect_proj = self.uvect_spn
        scale_factor = float(np.dot(uvect_proj, self.coord_converter.uvect_ipn))

        # (3) Set initial ground plane position G1 to the scene point position S.
        scene_point = np.array([ecf_world_coordinate.x, ecf_world_coordinate.y, ecf_world_coordinate.z])
        g_n = scene_point.copy()

        xrow_ycol_n = None
        cont = True
        iteration = 0
        while cont:
            iteration += 1

            # (4) Project ground plane point g_n to image plane point i_n. The projection distance is dist_n. Compute
            # image coordinates xrown and ycoln.
            dist_n = np.dot(self.coord_converter.scp_ecf.coordinate - g_n, self.coord_converter.uvect_ipn) / scale_factor
            i_n = g_n + dist_n * uvect_proj
            xrow_ycol_n = self.coord_converter.ipp_to_xrowycol(i_n)

            # (5) Compute the precise projection for image grid location (xrown, ycoln) to the ground plane containing
            # the scene point S. The result is point p_n. For image grid location (xrown, ycoln), compute COA
            # parameters per Section 2. Compute the precise R/Rdot projection contour per Section 4. Compute the
            # R/Rdot intersection with the ground plane per Section 5.
            r_tgt_coa, r_dot_tgt_coa, time_coa, arp_coa, varp_coa = self.coa_projection_set.precise_rrdot_computation(
                xrow_ycol_n
            )
            p_n = self.default_surface_projection.rrdot_to_ground(r_tgt_coa, r_dot_tgt_coa, arp_coa, varp_coa)

            # (6) Compute the displacement between ground plane point Pn and the scene point S.
            diff_n = scene_point - p_n[0]
            delta_gpn = np.linalg.norm(diff_n)
            g_n += diff_n

            # If the displacement is greater than the threshold (GP_MAX), compute point Gn+1 and repeat the
            # projections in steps (4) and (5) above. If the displacement is less than the threshold, accept image
            # grid location (xrown, ycoln) as the precise image grid location for scene point S.
            cont = delta_gpn > tolerance and (iteration < max_iterations)

        row_col = self.coord_converter.xrowycol_to_rowcol(xrow_ycol_n)

        # Convert the row_col image grid location to an x,y image coordinate. Note that row_col is in reference
        # to the full image, so we subtract off the first_pixel offset to make the image coordinate correct if this
        # is a chip.
        return ImageCoordinate(
            [row_col[1] - self.coord_converter.first_pixel.x, row_col[0] - self.coord_converter.first_pixel.y]
        )
