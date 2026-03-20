"""
Relative Orbital Elements (ROE) Dynamics — J2-Perturbed
=========================================================
Implements the quasi-nonsingular ROE state vector and its J2-perturbed
dynamics following D'Amico & Montenbruck (2006).

State vector (6 quasi-nonsingular ROE):
    δα = [δa, δλ, δe_x, δe_y, δi_x, δi_y]

Where:
    δa   = (a_d - a_c) / a_c          — relative semi-major axis
    δλ   = δM + δω + δΩ cos(i)        — relative mean longitude
    δe_x = e_d cos(ω_d) - e_c cos(ω_c) — relative eccentricity x
    δe_y = e_d sin(ω_d) - e_c sin(ω_c) — relative eccentricity y
    δi_x = i_d - i_c                   — relative inclination x
    δi_y = Ω_d sin(i_c) - Ω_c sin(i_c) — relative inclination y

J2 secular drift rates (D'Amico 2006, Eq. 10):
    δȧ   = 0
    δλ̇   = -3/2 n δa + 7/2 kappa η δe_x (+ mean motion difference)
    δė_x =  κ δe_y
    δė_y = -κ δe_x
    δi̇_x =  0
    δi̇_y =  κ' δi_x

where κ, κ' are J2-dependent drift coefficients.

The key insight: J2 causes the eccentricity vector (δe_x, δe_y) to precess
at a rate that depends only on the chief orbit — so a J2-invariant formation
simply needs δe aligned with this precession. This is the D'Amico
eccentricity/inclination vector separation condition.

LVLH conversion:
    Uses the Gauss variational equations (GVE) linearised about the chief
    to map ROE → Cartesian LVLH for sensor simulation and handoff to CW.

Reference:
    D'Amico & Montenbruck, "Proximity Operations of Formation-Flying
    Spacecraft Using an Eccentricity/Inclination Vector Separation",
    JGCD 29(3), 2006.

    Schaub & Junkins, "Analytical Mechanics of Space Systems", §14.6.
    
    Gaias & D'Amico, "Impulsive Maneuvers for Formation Reconfiguration
    Using Relative Orbital Elements", JGCD 38(6), 2015.
"""

import numpy as np


class ROEDynamics:
    """
    J2-perturbed ROE propagator for the deputy spacecraft.

    Parameters
    ----------
    a_chief_m : chief semi-major axis [m]
    e_chief   : chief eccentricity (near-circular, < 0.01)
    i_chief   : chief inclination [rad]
    """

    MU    = 3.986004418e14   # m³/s²
    J2    = 1.08263e-3       # Earth oblateness
    RE    = 6.3781e6         # Earth radius [m]

    def __init__(self,
                 a_chief_m: float,
                 e_chief:   float = 0.0001,
                 i_chief:   float = np.radians(51.6)):

        self.a  = float(a_chief_m)
        self.e  = float(e_chief)
        self.i  = float(i_chief)

        # Derived chief quantities
        self.n   = np.sqrt(self.MU / self.a**3)      # mean motion [rad/s]
        self.T   = 2.0 * np.pi / self.n              # orbital period [s]
        self.eta = np.sqrt(1.0 - self.e**2)          # eccentricity function

        # J2 secular drift coefficients
        # Reference: D'Amico (2006) Eq. 9-11, Schaub & Junkins Eq. 14.85
        kappa = (3.0/2.0) * self.J2 * (self.RE/self.a)**2 * self.n / self.eta**4

        # ω̇  — perigee precession rate
        self.omega_dot = kappa * (2.0 - (5.0/2.0) * np.sin(self.i)**2)

        # Ω̇  — RAAN drift rate
        self.RAAN_dot = -kappa * np.cos(self.i)

        # Mean motion corrected for J2 secular effect on mean longitude
        # ṅ_J2 = n * (1 + 3/2 J2 (Re/a)^2 / eta^3 * (1 - 3/2 sin^2 i))
        self._n_eff = self.n * (1.0
                      + 1.5 * self.J2 * (self.RE/self.a)**2
                      / self.eta**3 * (1.0 - 1.5 * np.sin(self.i)**2))

        # Relative eccentricity/inclination precession rate (κ in D'Amico)
        # Both e-vector and i-vector precess at same rate from J2
        self._kappa_ei = self.omega_dot  # rad/s

        # ROE state [δa, δλ, δe_x, δe_y, δi_x, δi_y]  (dimensionless)
        self.roe = np.zeros(6)

        # Mean anomaly of chief [rad] — tracked for LVLH conversion
        self.mean_anomaly = 0.0

        # Delta-V accumulator [m/s]
        self.dv_total = np.zeros(3)

        # Time elapsed [s]
        self.t = 0.0

    # ─────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────

    def set_from_lvlh(self,
                      dr_lvlh: np.ndarray,
                      dv_lvlh: np.ndarray):
        """
        Convert initial LVLH Cartesian offset to ROE.

        Uses the inverse of the linearised GVE mapping (D'Amico Eq. 5).
        Valid for small relative motion (|δr| << a).

        Parameters
        ----------
        dr_lvlh : relative position [δx, δy, δz] in LVLH [m]
        dv_lvlh : relative velocity [δẋ, δẏ, δż] in LVLH [m/s]
        """
        a, n, e, i, eta = self.a, self.n, self.e, self.i, self.eta

        dx, dy, dz   = dr_lvlh
        dvx, dvy, dvz = dv_lvlh

        # Near-circular approximation (e << 1): ω ≈ 0, M ≈ u (argument of lat)
        # ROE from GVE linearisation:
        # δa = 2/n * dvy   (Gauss VE in-plane normal)
        # δλ = dy/a - 2*dx/a * e  (≈ dy/a for e~0)
        # δe_x = dvx/n/a + 2*dvy/n/a  -- from in-plane motion
        # δe_y = ... 
        # See: Gaias&D'Amico 2015 Eq. 2-4 for exact form

        # For near-circular (e << 1):
        da   = (2.0 / (n * a)) * dvy
        dlam = dy / a - 2.0 * dx / a * e   # ≈ dy/a

        # Relative eccentricity vector components
        # From radial/tangential velocity perturbations
        dex  =  1.0/(n*a) * dvx + 2.0*e*da/2.0
        dey  =  2.0/(n*a) * dvy - e*(dy/a)   # leading term

        # Relative inclination vector
        dix  = dz / a
        diy  = dvz / (n * a)

        self.roe = np.array([da, dlam, dex, dey, dix, diy])

    def set_roe(self, roe: np.ndarray):
        """Directly set ROE state vector."""
        self.roe = np.array(roe, dtype=float)

    # ─────────────────────────────────────────────────────────────────
    # Propagation — J2-perturbed secular drift
    # ─────────────────────────────────────────────────────────────────

    def step(self, dt: float):
        """
        Propagate ROE by dt seconds under J2 secular drift.

        J2 secular dynamics (D'Amico 2006, linearised about chief):
            δȧ   = 0                          (conserved)
            δλ̇   = -3/2 n δa                  (differential mean motion)
            δė_x = +κ δe_y                    (eccentricity vector precession)
            δė_y = -κ δe_x
            δi̇_x = 0                          (inclination vector conserved)
            δi̇_y = +κ' δi_x                  (RAAN differential drift)

        Where κ = ω̇_J2 (perigee precession rate).
        """
        da, dlam, dex, dey, dix, diy = self.roe
        k = self._kappa_ei

        # Secular rates
        da_dot   = 0.0
        dlam_dot = -1.5 * self.n * da
        dex_dot  = +k * dey
        dey_dot  = -k * dex
        dix_dot  = 0.0
        diy_dot  = -self.RAAN_dot * dix   # differential RAAN drift

        # Euler step (ROE evolve slowly — first order adequate for dt=0.1s)
        self.roe += dt * np.array([da_dot, dlam_dot, dex_dot, dey_dot,
                                   dix_dot, diy_dot])

        # Advance chief mean anomaly
        self.mean_anomaly = (self.mean_anomaly + self.n * dt) % (2*np.pi)
        self.t += dt

    # ─────────────────────────────────────────────────────────────────
    # LVLH conversion — ROE → Cartesian
    # ─────────────────────────────────────────────────────────────────

    def to_lvlh(self, mean_anomaly: float = None) -> np.ndarray:
        """
        Convert current ROE to LVLH Cartesian state [δr, δv] in [m, m/s].

        Uses the linearised GVE mapping (D'Amico 2006, Eq. 4):
            δx = a * (δe_x cos(u) + δe_y sin(u)) - a/2 δa
            δy = a * δλ - 2a * δe_x sin(u) + 2a * δe_y cos(u)
            δz = a * (δi_x sin(u) - δi_y cos(u))

        Velocity from time derivative of position along orbit.

        Parameters
        ----------
        mean_anomaly : chief mean anomaly [rad]. Uses self.mean_anomaly if None.

        Returns
        -------
        state : [δx, δy, δz, δẋ, δẏ, δż] in LVLH [m, m/s]
        """
        if mean_anomaly is None:
            mean_anomaly = self.mean_anomaly

        a, n = self.a, self.n
        da, dlam, dex, dey, dix, diy = self.roe
        u = mean_anomaly   # argument of latitude ≈ M + ω (circular orbit)

        su, cu = np.sin(u), np.cos(u)

        # Position (D'Amico Eq. 4, near-circular form)
        dx = a * (dex * cu + dey * su) - 0.5 * a * da
        dy = a * dlam + 2.0*a * (-dex * su + dey * cu)
        dz = a * (dix * su - diy * cu)

        # Velocity (time derivative, using du/dt = n)
        dvx = a * n * (-dex * su + dey * cu)
        dvy = 2.0 * a * n * (-dex * cu - dey * su) - 1.5 * a * n * da
        dvz = a * n * (dix * cu + diy * su)

        return np.array([dx, dy, dz, dvx, dvy, dvz])

    # ─────────────────────────────────────────────────────────────────
    # Maneuver application
    # ─────────────────────────────────────────────────────────────────

    def apply_impulse_lvlh(self,
                           dv_lvlh: np.ndarray,
                           mean_anomaly: float = None):
        """
        Apply an instantaneous delta-V in LVLH and update ROE accordingly.

        Uses Gauss variational equations (GVE) linearised for near-circular orbit.
        Δδα = Γ(u) · Δv   (D'Amico 2006, Eq. 6)

        Parameters
        ----------
        dv_lvlh      : [Δvx, Δvy, Δvz] impulse in LVLH [m/s]
        mean_anomaly : argument of latitude at burn [rad]
        """
        if mean_anomaly is None:
            mean_anomaly = self.mean_anomaly

        a, n = self.a, self.n
        u  = mean_anomaly
        su, cu = np.sin(u), np.cos(u)

        dvr, dvt, dvn = dv_lvlh   # radial, tangential, normal

        # GVE mapping Γ: LVLH ΔV → ΔROE  (D'Amico Eq. 6, near-circular)
        # Rows: δa, δλ, δe_x, δe_y, δi_x, δi_y
        inv_na = 1.0 / (n * a)

        d_da   = 2.0 * inv_na * dvt
        d_dlam = -2.0 * inv_na * dvr  + 0.0 * dvt    # leading order
        d_dex  = inv_na * (dvr * su  + 2.0 * dvt * cu)
        d_dey  = inv_na * (-dvr * cu + 2.0 * dvt * su)
        d_dix  = inv_na * dvn * cu
        d_diy  = inv_na * dvn * su

        self.roe += np.array([d_da, d_dlam, d_dex, d_dey, d_dix, d_diy])
        self.dv_total += np.abs(dv_lvlh)

    # ─────────────────────────────────────────────────────────────────
    # Convenience
    # ─────────────────────────────────────────────────────────────────

    @property
    def range_m(self) -> float:
        """Current range to chief [m]."""
        return float(np.linalg.norm(self.to_lvlh()[0:3]))