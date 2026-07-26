from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable
import numpy as np

@dataclass(frozen=True)
class OpticalConfig:
    led_positions: np.ndarray
    receiver_height: float = 0.85
    optical_power_w: float = 5.0
    lambertian_order: float = 1.0
    pd_area_m2: float = 1e-4
    responsivity: float = 0.53
    fov_deg: float = 85.0
    refractive_index: float = 1.5
    filter_gain: float = 1.0


def unit_normal(tilt_deg: float, azimuth_deg: float) -> np.ndarray:
    tilt = np.deg2rad(tilt_deg)
    azi = np.deg2rad(azimuth_deg)
    return np.array([np.sin(tilt)*np.cos(azi), np.sin(tilt)*np.sin(azi), np.cos(tilt)], dtype=float)


def default_pd_normals() -> np.ndarray:
    orientations = [(0,0),(45,0),(45,180),(45,90),(45,-90),(45,45),(45,135),(45,-45)]
    return np.stack([unit_normal(t,a) for t,a in orientations])


def los_gains(xy: np.ndarray, cfg: OpticalConfig, pd_normals: np.ndarray | None = None) -> np.ndarray:
    """Return received-current proxy with shape [N, LED, PD]."""
    xy = np.atleast_2d(xy).astype(float)
    pd_normals = default_pd_normals() if pd_normals is None else np.asarray(pd_normals)
    rx = np.column_stack([xy, np.full(len(xy), cfg.receiver_height)])
    leds = np.asarray(cfg.led_positions, dtype=float)
    # Direction from receiver to LED.
    v = leds[None, :, :] - rx[:, None, :]
    d = np.linalg.norm(v, axis=-1)
    u_rx_to_led = v / np.maximum(d[..., None], 1e-12)
    # LED points down. Irradiance angle is measured against downward normal.
    cos_phi = np.clip(u_rx_to_led[..., 2], 0.0, 1.0)
    # Incidence angle against each PD normal.
    cos_psi = np.einsum('nld,pd->nlp', u_rx_to_led, pd_normals)
    fov_cos = math.cos(math.radians(cfg.fov_deg))
    valid = cos_psi >= fov_cos
    g_con = cfg.refractive_index**2 / (math.sin(math.radians(cfg.fov_deg))**2)
    base = ((cfg.lambertian_order+1)*cfg.pd_area_m2/(2*math.pi)) * (cos_phi**cfg.lambertian_order) / np.maximum(d**2, 1e-12)
    h = base[..., None] * np.clip(cos_psi, 0, None) * cfg.filter_gain * g_con
    h *= valid
    return cfg.responsivity * cfg.optical_power_w * h


def apply_gain_mismatch(gains: np.ndarray, rng: np.random.Generator, std: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    factors = np.exp(rng.normal(0.0, std, size=gains.shape[-1]))
    return gains * factors[None, None, :], factors


def calibrate_relative_gains(gains: np.ndarray, factors: np.ndarray) -> np.ndarray:
    return gains / factors[None, None, :]


def add_awgn_for_snr(gains: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_rms = np.sqrt(np.mean(gains**2, axis=(-2,-1), keepdims=True))
    noise_rms = signal_rms / (10.0**(snr_db/20.0))
    return gains + rng.normal(0.0, 1.0, size=gains.shape)*noise_rms


def scale_free_features(gains: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(gains, axis=-1, keepdims=True)
    return gains / np.maximum(denom, eps)


def generate_smooth_trajectory(rng: np.random.Generator, steps: int, dt: float = 0.1,
                               room_xy=(5.0,6.0), speed_range=(0.2,1.2)) -> np.ndarray:
    pos = np.array([rng.uniform(0.4, room_xy[0]-0.4), rng.uniform(0.4, room_xy[1]-0.4)])
    theta = rng.uniform(-math.pi, math.pi)
    speed = rng.uniform(*speed_range)
    out=[]
    for _ in range(steps):
        theta += rng.normal(0,0.10)
        speed = np.clip(speed+rng.normal(0,0.03), *speed_range)
        pos = pos + dt*speed*np.array([math.cos(theta),math.sin(theta)])
        for k,lim in enumerate(room_xy):
            if pos[k] < 0.15 or pos[k] > lim-0.15:
                theta = math.pi-theta if k==0 else -theta
                pos[k] = np.clip(pos[k],0.15,lim-0.15)
        out.append(pos.copy())
    return np.asarray(out)


def missing_mask(rng: np.random.Generator, steps: int, anchors: int=6, p=0.15,
                 independent_fraction=0.60, burst_lengths=(2,4)) -> np.ndarray:
    mask=np.ones((steps,anchors),dtype=np.float32)
    independent_p=p*independent_fraction
    mask[rng.random(mask.shape)<independent_p]=0
    target=int(round(steps*anchors*p*(1-independent_fraction)))
    missing=0
    while missing<target:
        a=int(rng.integers(anchors)); start=int(rng.integers(steps)); length=int(rng.integers(burst_lengths[0],burst_lengths[1]+1))
        end=min(steps,start+length)
        before=np.sum(mask[start:end,a]==0)
        mask[start:end,a]=0
        missing += (end-start)-before
    return mask
