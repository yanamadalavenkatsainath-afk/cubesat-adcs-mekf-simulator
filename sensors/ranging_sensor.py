"""
Relative Navigation Sensor — Range + Bearing (Optical / RF)
===========================================================
Simulates a proximity sensor that measures:
    - Range     : scalar distance to deputy [m]
    - Azimuth   : bearing angle in LVLH x-y plane [rad]
    - Elevation : out-of-plane bearing angle [rad]

This measurement set is representative of:
    - Monocular camera with known target geometry (bearing) + RF ranging
    - LIDAR (both range and bearing from point cloud centroid)
    - VBS (Vision-Based Sensor) as used on PRISMA, ATV, Dragon

Noise model:
    σ_range     = max(σ_range_abs, σ_range_frac * range)
                  Absolute floor + fractional component (laser noise model)
    σ_az, σ_el  = σ_angle  (white noise on angular measurements)

FOV constraint:
    Sensor has a ±FOV/2 half-angle cone. Returns None if deputy outside FOV.

Reference:
    D'Amico & Montenbruck, "Proximity Operations of Formation-Flying
    Spacecraft Using an Eccentricity/Inclination Vector Separation",
    JGCD 29(3), 2006.
"""

import numpy as np


class RangingBearingSensor:
    """
    Range + bearing sensor for relative navigation.

    Measurement vector z = [range, azimuth, elevation]

    Parameters
    ----------
    sigma_range_m    : absolute range noise 1-sigma [m]. Default 2m.
    sigma_range_frac : fractional range noise (e.g. 0.005 = 0.5%). Default 0.5%.
    sigma_angle_rad  : angular noise 1-sigma [rad]. Default 0.5°.
    fov_half_deg     : sensor half-angle FOV [deg]. Default 45°.
    min_range_m      : minimum detectable range [m]. Default 1m.
    max_range_m      : maximum detectable range [m]. Default 10 km.
    """

    def __init__(self,
                 sigma_range_m:    float = 2.0,
                 sigma_range_frac: float = 0.005,
                 sigma_angle_rad:  float = np.radians(0.5),
                 fov_half_deg:     float = 45.0,
                 min_range_m:      float = 1.0,
                 max_range_m:      float = 10_000.0):

        self.sigma_range_abs  = sigma_range_m
        self.sigma_range_frac = sigma_range_frac
        self.sigma_angle      = sigma_angle_rad
        self.fov_half         = np.radians(fov_half_deg)
        self.min_range        = min_range_m
        self.max_range        = max_range_m

    def measure(self,
                dr_lvlh: np.ndarray,
                sensor_pointing_lvlh: np.ndarray = None
                ) -> tuple[np.ndarray | None, np.ndarray]:
        """
        Generate noisy range + bearing measurement.

        Parameters
        ----------
        dr_lvlh              : true relative position in LVLH [m]
        sensor_pointing_lvlh : unit vector of sensor boresight in LVLH.
                               Default: [0, 1, 0] (along-track, toward deputy
                               in a typical trailing formation).

        Returns
        -------
        z    : measurement [range_m, az_rad, el_rad] or None if outside FOV/range
        R    : 3×3 measurement noise covariance
        """
        if sensor_pointing_lvlh is None:
            sensor_pointing_lvlh = np.array([0., 1., 0.])

        true_range = np.linalg.norm(dr_lvlh)

        # Range validity check
        if true_range < self.min_range or true_range > self.max_range:
            return None, self._noise_cov(true_range)

        dr_hat = dr_lvlh / true_range

        # FOV check — angle between sensor boresight and deputy direction
        cos_ang = np.dot(sensor_pointing_lvlh, dr_hat)
        if cos_ang < np.cos(self.fov_half):
            return None, self._noise_cov(true_range)    # outside FOV

        # True azimuth and elevation from LVLH x-y-z
        # Azimuth   : angle in x-y plane from +x axis
        # Elevation : angle above x-y plane
        az_true = np.arctan2(dr_lvlh[1], dr_lvlh[0])
        el_true = np.arctan2(dr_lvlh[2],
                             np.sqrt(dr_lvlh[0]**2 + dr_lvlh[1]**2))

        # Range noise: absolute floor + fractional
        sigma_r = max(self.sigma_range_abs,
                      self.sigma_range_frac * true_range)
        noise_r  = np.random.normal(0, sigma_r)
        noise_az = np.random.normal(0, self.sigma_angle)
        noise_el = np.random.normal(0, self.sigma_angle)

        z = np.array([true_range + noise_r,
                      az_true    + noise_az,
                      el_true    + noise_el])

        R = self._noise_cov(true_range)
        return z, R

    def _noise_cov(self, range_m: float) -> np.ndarray:
        """Build 3×3 diagonal measurement noise covariance."""
        sigma_r = max(self.sigma_range_abs,
                      self.sigma_range_frac * max(range_m, self.min_range))
        return np.diag([sigma_r**2,
                        self.sigma_angle**2,
                        self.sigma_angle**2])

    @staticmethod
    def invert(z: np.ndarray) -> np.ndarray:
        """
        Convert [range, azimuth, elevation] measurement back to
        Cartesian LVLH estimate (noiseless inversion for filter init).
        """
        r, az, el = z
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z_c = r * np.sin(el)
        return np.array([x, y, z_c])