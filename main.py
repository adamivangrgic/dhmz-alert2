#!/usr/bin/env python3
"""
DHMZ radar + pljusak station tracker with satellite basemap & storm frame archiver.

Endpoints:
  /                    HTML dashboard
  /threat.png          Current frame: satellite/radar composite + overlays
  /threat.gif          Animated composite
  /status              JSON: summary + local_weather + sensors + threats + archive
  /settings            GET/POST hot-swappable runtime settings
  /config              GET full JSON config (POST replace, DELETE reset)
  /archive             GET archived storm episodes & frame counts
  /refresh             POST force refresh
  /accuracy            GET stats, /accuracy/predictions, /accuracy/events
  /accuracy/reset      POST clear scores
  /accuracy/replay     POST re-score

Config persists to <data-dir>/config.json. Default <data-dir> is /data.
Basemap cache at <data-dir>/basemap_cache/.
Storm frame archive at <data-dir>/archive/<YYYY-MM-DD_HHMMZ>/.
"""

import argparse
import io
import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import numpy as np
import requests
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageSequence
from scipy import ndimage
from scipy.spatial import cKDTree

try:
    import pytesseract
    HAVE_TESSERACT = True
except ImportError:
    HAVE_TESSERACT = False
try:
    import paho.mqtt.client as mqtt
    HAVE_MQTT = True
except ImportError:
    HAVE_MQTT = False
try:
    from zoneinfo import ZoneInfo
    HAVE_ZONEINFO = True
except ImportError:
    HAVE_ZONEINFO = False

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
GIF_URL = "https://vrijeme.hr/anim_kompozit.gif"
RADAR_HEADERS = {"User-Agent": "Mozilla/5.0 (dhmz-radar-tracker/1.0)"}
PLJUSAK_URL = "https://pljusak.com/"
PLJUSAK_HEADERS = {
    "User-Agent": "Mozilla/5.0 (pljusak-station-puller/1.0)",
    "Referer": "https://pljusak.com/",
    "Accept": "text/html,application/xhtml+xml",
}

MIN_SAT = 0.45
MIN_VAL = 0.25
MIN_CLUSTER_PIXELS = 15
MAX_STORM_SPEED_KMH = 140.0
VELOCITY_WINDOW_MIN = 25.0
MIN_SPEED_KMH = 2.0

STALE_MIN = 20
VERY_STALE_MIN = 40

CLUSTER_DIGITS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"

DEFAULT_SRI_THRESHOLDS = (2.5, 10.0, 30.0)

# Official DHMZ SRI (Surface Rainfall Intensity) color scale:
# (min_mmh, max_mmh, (R, G, B), label)
DHMZ_SRI_SCALE = [
    (0.2,   0.5,   (20,  50,  210), "0.2"),
    (0.5,   1.0,   (0,   110, 255), "0.5"),
    (1.0,   2.0,   (0,   185, 255), "1"),
    (2.0,   3.0,   (0,   230, 200), "2"),
    (3.0,   5.0,   (0,   225, 110), "3"),
    (5.0,   7.0,   (0,   195, 0),   "5"),
    (7.0,   10.0,  (140, 230, 0),   "7"),
    (10.0,  15.0,  (255, 255, 0),   "10"),
    (15.0,  20.0,  (255, 210, 0),   "15"),
    (20.0,  30.0,  (255, 155, 0),   "20"),
    (30.0,  40.0,  (255, 95,  0),   "30"),
    (40.0,  50.0,  (240, 40,  0),   "40"),
    (50.0,  70.0,  (215, 0,   0),   "50"),
    (70.0,  100.0, (170, 0,   35),  "70"),
    (100.0, 150.0, (200, 0,   180), "100"),
    (150.0, 999.0, (240, 130, 255), "150+"),
]

_DHMZ_SRI_TREE = cKDTree(np.array([c[2] for c in DHMZ_SRI_SCALE], dtype=np.float64))
_DHMZ_SRI_MIDPOINTS = np.array(
    [math.sqrt(c[0] * min(c[1], 200.0)) for c in DHMZ_SRI_SCALE], dtype=np.float32
)

COMPASS_16_DEG = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}

STALE_PRED_EXTRA_MIN = 30.0
MATCH_TOLERANCE_FACTOR = 1.5
MATCH_TOLERANCE_MIN = 60.0

DEFAULT_DISPLAY_TZ = "Europe/Zagreb"
DEFAULT_TIMESTAMP_REGION_STR = "0,0,260,22"

CITIES = [
    ("zg", "Zagreb",         45.8146, 15.9722),
    ("ri", "Rijeka",         45.3271, 14.4422),
    ("st", "Split",          43.5081, 16.4402),
    ("os", "Osijek",         45.5550, 18.6955),
    ("zd", "Zadar",          44.1194, 15.2314),
    ("db", "Dubrovnik",      42.6507, 18.0944),
    ("pu", "Pula",           44.8666, 13.8496),
    ("ka", "Karlovac",       45.4929, 15.5551),
    ("va", "Varaždin",       46.3057, 16.3366),
    ("lj", "Ljubljana",      46.0569, 14.5058),
    ("tb", "Sarajevo",       43.8633, 18.4131),
    ("sk", "Sisak",          45.4667, 16.3789),
    ("mk", "Makarska",       43.2969, 17.0178),
    ("šb", "Šibenik",        43.7350, 15.8956),
    ("vk", "Vukovar",        45.2551, 19.1122),
    ("sb", "Slavonski Brod", 45.1603, 18.0156),
    ("po", "Požega",         45.3403, 17.6853),
    ("bj", "Bjelovar",       45.8986, 16.8422),
    ("kt", "Kutina",         45.4772, 16.7783),
    ("bk", "Bjelolasica",    45.2540, 15.0160),
    ("og", "Ogulin",         45.2631, 15.2281),
    ("gs", "Gospić",         44.5461, 15.3747),
    ("kn", "Knin",           44.0409, 16.1969),
]

ADRIATIC_COAST = [
    (45.52, 13.57), (45.47, 13.58), (45.40, 13.55), (45.30, 13.58),
    (45.20, 13.60), (45.10, 13.62), (44.97, 13.65), (44.87, 13.83),
    (44.80, 13.92), (44.70, 14.00), (44.60, 14.10), (44.75, 14.20),
    (44.75, 14.30), (44.75, 14.40), (44.63, 14.50), (44.55, 14.45),
    (44.60, 14.60), (44.70, 14.70), (44.80, 14.75), (45.00, 14.75),
    (45.10, 14.70), (45.15, 14.65), (45.30, 14.45), (45.30, 14.35),
    (45.20, 14.30), (45.10, 14.20), (45.05, 14.15), (44.95, 14.15),
    (44.88, 14.20), (44.85, 14.30), (44.75, 14.55), (44.65, 14.75),
    (44.55, 14.80), (44.78, 14.78), (44.75, 14.85), (44.60, 14.95),
    (44.55, 14.90), (44.50, 14.85), (44.45, 15.05), (44.30, 15.10),
    (44.20, 15.10), (44.15, 15.20), (44.10, 15.23), (44.00, 15.20),
    (43.95, 15.30), (43.85, 15.50), (43.75, 15.65), (43.73, 15.89),
    (43.65, 15.95), (43.50, 16.10), (43.50, 16.44), (43.45, 16.55),
    (43.40, 16.65), (43.30, 16.90), (43.20, 17.10), (43.05, 17.45),
    (42.90, 17.55), (42.85, 17.70), (42.75, 18.00), (42.65, 18.10),
    (42.55, 18.25), (42.48, 18.35), (42.40, 18.52),
]

DEFAULT_CALIBRATION_JSON = (
    '{"map":{"x0":0,"y0":0,"x1":720,"y1":720},'
    '"points":['
    '{"pixel":[343,226],"city":"Zagreb"},'
    '{"pixel":[385,490],"city":"Split"},'
    '{"pixel":[566,258],"city":"Osijek"}'
    ']}'
)

# ----------------------------------------------------------------------------
# Runtime (hot-swappable)
# ----------------------------------------------------------------------------
_RUNTIME = {
    "approaching_radius_km": 20.0,
    "cluster_radius_weight": 1.0,
    "only_approach_deg": 180.0,
    "threat_radius_km": 15.0,
    "threat_angle_deg": 45.0,
    "threat_min_duration": 3,
    "threat_min_sri_mmh": 0.0,
    "min_duration_frames": 2,
    "min_size_px": 30,
    "radius_km": 150.0,
    "weather_radius_km": 5.0,
    "archive_enabled": True,
    "archive_radius_km": 200.0,
    "archive_min_pixels": 40,
    "poll_s": 300,
    "view_rotation_auto": True,
    "view_rotation": 0.0,
    "basemap": "auto",
    "basemap_saturation": 1.0,
    "basemap_brightness": 1.0,
    "basemap_contrast": 1.0,
    "show_cities": True,
    "web_scale": 1.0,
    "settings_tag": "",
}
_RUNTIME_TYPES = {
    "approaching_radius_km": (float, 1.0, 500.0),
    "cluster_radius_weight": (float, 0.0, 5.0),
    "only_approach_deg": (float, 0.0, 180.0),
    "threat_radius_km": (float, 1.0, 500.0),
    "threat_angle_deg": (float, 0.0, 180.0),
    "threat_min_duration": (int, 1, 50),
    "threat_min_sri_mmh": (float, 0.0, 500.0),
    "min_duration_frames": (int, 1, 50),
    "min_size_px": (int, 5, 5000),
    "radius_km": (float, 20.0, 1000.0),
    "weather_radius_km": (float, 0.5, 100.0),
    "archive_enabled": (bool, None, None),
    "archive_radius_km": (float, 20.0, 500.0),
    "archive_min_pixels": (int, 10, 5000),
    "poll_s": (int, 30, 7200),
    "view_rotation_auto": (bool, None, None),
    "view_rotation": (float, 0.0, 360.0),
    "basemap": (str, None, None),
    "basemap_saturation": (float, 0.0, 2.0),
    "basemap_brightness": (float, 0.2, 2.0),
    "basemap_contrast": (float, 0.2, 2.0),
    "show_cities": (bool, None, None),
    "web_scale": (float, 0.1, 4.0),
    "settings_tag": (str, None, None),
}
_RUNTIME_LOCK = threading.Lock()
_FORCE_REFRESH = threading.Event()

# ----------------------------------------------------------------------------
# Static (restart-required)
# ----------------------------------------------------------------------------
_STATIC_DEFAULTS = {
    "location": None,
    "lat": None,
    "lon": None,
    "calibration_inline": DEFAULT_CALIBRATION_JSON,
    "calibration": None,
    "radar_intensity": "simple",
    "sri_legend_region": None,
    "sri_legend_orientation": "horizontal",
    "sri_legend_padding": 1,
    "sri_breaks": None,
    "sri_thresholds": "2.5,10,30",
    "frame_interval_min": 5.0,
    "auto_frame_interval": False,
    "timestamp_region": DEFAULT_TIMESTAMP_REGION_STR,
    "timestamp_enabled": True,
    "timestamp_tz": None,
    "timestamp_source": "last",
    "display_tz": DEFAULT_DISPLAY_TZ,
    "pljusak_enabled": True,
    "pljusak_url": PLJUSAK_URL,
    "pljusak_interval": 60,
    "station_confirm_radius": 15.0,
    "station_confirm_min_mm": 0.2,
    "station_confirm_max_age_min": 15.0,
    "station_confirm_fraction": 0.5,
    "station_confirm_min_count": 1,
    "web_enabled": True,
    "web_port": 8080,
    "mqtt_enabled": True,
    "mqtt_host": None,
    "mqtt_port": 1883,
    "mqtt_tls": False,
    "mqtt_discovery_prefix": "homeassistant",
    "mqtt_state_prefix": "dhmz_radar",
    "mqtt_device_id": None,
    "mqtt_device_name": "DHMZ Radar",
    "mqtt_sensors": "all",
    "mqtt_mark_offline": None,
    "mqtt_no_retain": False,
    "mqtt_threat_topic": None,
    "jsonl_dump": None,
    "accuracy_enabled": True,
    "mapbox_token": None,
    "mapbox_style": "mapbox/satellite-v9",
    "basemap_size": 640,
}
_STATIC = dict(_STATIC_DEFAULTS)
_STATIC_LOCK = threading.Lock()

_CLI_TO_RUNTIME = {
    "approaching_radius": "approaching_radius_km",
    "cluster_radius_weight": "cluster_radius_weight",
    "only_approach": "only_approach_deg",
    "threat_radius": "threat_radius_km",
    "threat_angle": "threat_angle_deg",
    "threat_min_duration": "threat_min_duration",
    "threat_min_sri": "threat_min_sri_mmh",
    "min_duration": "min_duration_frames",
    "min_size": "min_size_px",
    "radius": "radius_km",
    "weather_radius": "weather_radius_km",
    "archive_radius": "archive_radius_km",
    "archive_min_pixels": "archive_min_pixels",
    "poll": "poll_s",
    "view_rotation": "view_rotation",
    "basemap": "basemap",
    "basemap_saturation": "basemap_saturation",
    "basemap_brightness": "basemap_brightness",
    "basemap_contrast": "basemap_contrast",
    "web_scale": "web_scale",
    "settings_tag": "settings_tag",
}
_CLI_TO_STATIC = {
    "location": "location",
    "lat": "lat",
    "lon": "lon",
    "calibration": "calibration",
    "calibration_inline": "calibration_inline",
    "radar_intensity": "radar_intensity",
    "sri_legend_region": "sri_legend_region",
    "sri_legend_orientation": "sri_legend_orientation",
    "sri_legend_padding": "sri_legend_padding",
    "sri_breaks": "sri_breaks",
    "sri_thresholds": "sri_thresholds",
    "frame_interval_min": "frame_interval_min",
    "timestamp_region": "timestamp_region",
    "timestamp_tz": "timestamp_tz",
    "timestamp_source": "timestamp_source",
    "display_tz": "display_tz",
    "pljusak_url": "pljusak_url",
    "pljusak_interval": "pljusak_interval",
    "station_confirm_radius": "station_confirm_radius",
    "station_confirm_min_mm": "station_confirm_min_mm",
    "station_confirm_max_age_min": "station_confirm_max_age_min",
    "station_confirm_fraction": "station_confirm_fraction",
    "station_confirm_min_count": "station_confirm_min_count",
    "web_port": "web_port",
    "mqtt_host": "mqtt_host",
    "mqtt_port": "mqtt_port",
    "mqtt_tls": "mqtt_tls",
    "ha_discovery_prefix": "mqtt_discovery_prefix",
    "ha_state_prefix": "mqtt_state_prefix",
    "ha_device_id": "mqtt_device_id",
    "ha_device_name": "mqtt_device_name",
    "ha_sensors": "mqtt_sensors",
    "mqtt_no_retain": "mqtt_no_retain",
    "ha_threat_topic": "mqtt_threat_topic",
    "jsonl_dump": "jsonl_dump",
    "mapbox_token": "mapbox_token",
    "mapbox_style": "mapbox_style",
    "basemap_size": "basemap_size",
}
_CLI_TO_STATIC_BOOL = {
    "web_enabled": "web_enabled",
    "pljusak_enabled": "pljusak_enabled",
    "timestamp_enabled": "timestamp_enabled",
    "mqtt_enabled": "mqtt_enabled",
    "accuracy_enabled": "accuracy_enabled",
}
_CLI_TO_RUNTIME_BOOL = {
    "cities": "show_cities",
    "view_rotation_auto": "view_rotation_auto",
    "archive_enabled": "archive_enabled",
}

_VIZ_STATE = {
    "png": None, "gif": None, "updated": None, "status": {},
    "lock": threading.Lock(),
}


# ----------------------------------------------------------------------------
# Config helpers
# ----------------------------------------------------------------------------
def _load_json_file(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"config: load error {e}", file=sys.stderr)
        return None


def _save_json_file(path, data):
    if not path:
        return False
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        if os.path.exists(path):
            try:
                os.replace(path, path + ".bak")
            except OSError:
                pass
        os.replace(tmp, path)
        return True
    except Exception as e:
        print(f"config: save error {e}", file=sys.stderr)
        return False


def _coerce(key, value):
    if key == "basemap":
        v = str(value).lower().strip()
        return v if v in ("auto", "mapbox", "greyscale", "none") else "auto"
    t, lo, hi = _RUNTIME_TYPES[key]
    if t is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).lower() in ("1", "true", "yes", "on", "checked")
    if t is str:
        return str(value)[:64]
    if t is int:
        v = int(round(float(value)))
    elif t is float:
        v = float(value)
    else:
        return value
    if lo is not None:
        v = max(lo, min(hi, v))
    return v


def _snapshot():
    with _RUNTIME_LOCK:
        return dict(_RUNTIME)


def _static_snapshot():
    with _STATIC_LOCK:
        return dict(_STATIC)


def _apply_static(d):
    with _STATIC_LOCK:
        _STATIC.update(d)


def _merge_config(cli_args, existing_cfg):
    existing_cfg = existing_cfg or {}
    runtime = dict(_RUNTIME)
    runtime.update(existing_cfg.get("runtime") or {})
    static = dict(_STATIC_DEFAULTS)
    static.update(existing_cfg.get("static") or {})

    for cli_key, cfg_key in _CLI_TO_RUNTIME.items():
        v = getattr(cli_args, cli_key, None)
        if v is not None:
            runtime[cfg_key] = _coerce(cfg_key, v)
    for cli_key, cfg_key in _CLI_TO_RUNTIME_BOOL.items():
        v = getattr(cli_args, cli_key, None)
        if v is not None:
            runtime[cfg_key] = bool(v)
    for cli_key, cfg_key in _CLI_TO_STATIC.items():
        v = getattr(cli_args, cli_key, None)
        if v is not None:
            static[cfg_key] = v
    for cli_key, cfg_key in _CLI_TO_STATIC_BOOL.items():
        v = getattr(cli_args, cli_key, None)
        if v is not None:
            static[cfg_key] = bool(v)

    if cli_args.location is not None:
        static["location"] = cli_args.location
        static["lat"] = None
        static["lon"] = None
    elif cli_args.lat is not None or cli_args.lon is not None:
        static["location"] = None
        if cli_args.lat is not None:
            static["lat"] = cli_args.lat
        if cli_args.lon is not None:
            static["lon"] = cli_args.lon

    if static.get("mqtt_host") is None:
        env_val = os.environ.get("MQTT_HOST")
        if env_val:
            static["mqtt_host"] = env_val
    if static.get("mqtt_port") in (None, 1883):
        env_val = os.environ.get("MQTT_PORT")
        if env_val:
            try:
                static["mqtt_port"] = int(env_val)
            except ValueError:
                pass
    if static.get("mapbox_token") is None:
        env_val = os.environ.get("MAPBOX_TOKEN")
        if env_val:
            static["mapbox_token"] = env_val

    return runtime, static


# ----------------------------------------------------------------------------
# Cities + timezone
# ----------------------------------------------------------------------------
def _city_lookup(query):
    import unicodedata
    import difflib

    def norm(s):
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return s.lower().strip()

    q = norm(query)
    for label, name, lat, lon in CITIES:
        if q == norm(label) or q == norm(name):
            return name, lat, lon
    d = {norm(name): (name, lat, lon) for _, name, lat, lon in CITIES}
    d.update({norm(label): (name, lat, lon) for label, name, lat, lon in CITIES})
    m = difflib.get_close_matches(q, list(d.keys()), n=1, cutoff=0.6)
    return d[m[0]] if m else None


def resolve_city(query):
    return _city_lookup(query)


def city_label(name):
    for label, nm, _, _ in CITIES:
        if nm == name:
            return label
    return name[:2].lower()


def get_zoneinfo(name):
    if not HAVE_ZONEINFO:
        return timezone.utc, "UTC"
    cand = name or os.environ.get("TZ") or DEFAULT_DISPLAY_TZ
    try:
        return ZoneInfo(cand), cand
    except Exception:
        return timezone.utc, "UTC"


def tz_abbrev(dt):
    return dt.strftime("%Z") or dt.strftime("%z")


def cluster_label(idx):
    return CLUSTER_DIGITS[idx] if 0 <= idx < len(CLUSTER_DIGITS) else f"#{idx + 1}"


# ----------------------------------------------------------------------------
# Coast rotation (satellite edge detection + polyline fallback)
# ----------------------------------------------------------------------------
def detect_coast_rotation_from_satellite(basemap_north_up):
    if basemap_north_up is None:
        return None
    arr = np.asarray(basemap_north_up.convert("RGB"), dtype=np.float32)
    H, W = arr.shape[:2]
    if H < 32 or W < 32:
        return None

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    water = (lum < 95.0) & (b >= r - 2.0) & (b >= g - 6.0)
    water |= (lum < 55.0) & (b >= r)

    struct = np.ones((5, 5), dtype=bool)
    water = ndimage.binary_closing(water, structure=struct)
    water = ndimage.binary_opening(water, structure=struct)
    water = ndimage.binary_fill_holes(water)

    margin = max(12, int(min(H, W) * 0.04))
    inner = water[margin:H - margin, margin:W - margin]
    water_frac = float(inner.mean())
    if water_frac < 0.04 or water_frac > 0.96:
        return None

    dil = ndimage.binary_dilation(water, structure=np.ones((7, 7), dtype=bool))
    ero = ndimage.binary_erosion(water, structure=np.ones((7, 7), dtype=bool))
    coast_band = dil & (~ero)
    coast_band[:margin, :] = False
    coast_band[H - margin:, :] = False
    coast_band[:, :margin] = False
    coast_band[:, W - margin:] = False

    if int(coast_band.sum()) < 25:
        return None

    smooth_water = ndimage.gaussian_filter(water.astype(np.float32), sigma=3.0)
    gx = ndimage.sobel(smooth_water, axis=1)
    gy = ndimage.sobel(smooth_water, axis=0)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = H / 2.0, W / 2.0
    sigma_pos = min(H, W) * 0.35
    spatial_w = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma_pos ** 2))

    w = spatial_w * coast_band.astype(np.float32)
    sx = float(np.sum(gx * w))
    sy = float(np.sum(gy * w))
    mag = math.hypot(sx, sy)
    total_grad = float(np.sum(np.hypot(gx, gy) * w))
    if total_grad <= 1e-6 or (mag / total_grad) < 0.15:
        return None

    seaward_bearing = math.degrees(math.atan2(sx, -sy)) % 360.0
    return seaward_bearing


def compute_coast_rotation(target_lat, target_lon, max_dist_km=100.0):
    best_dist = float("inf")
    best_bearing = None
    for (lat1, lon1), (lat2, lon2) in zip(ADRIATIC_COAST, ADRIATIC_COAST[1:]):
        coslat = math.cos(math.radians((lat1 + target_lat) / 2))
        sx = (lon2 - lon1) * 111.32 * coslat
        sy = (lat2 - lat1) * 111.32
        tx = (target_lon - lon1) * 111.32 * coslat
        ty = (target_lat - lat1) * 111.32
        seg_sq = sx * sx + sy * sy
        if seg_sq < 1e-9:
            continue
        t = (tx * sx + ty * sy) / seg_sq
        t = max(0.0, min(1.0, t))
        px, py = sx * t, sy * t
        dist = math.hypot(tx - px, ty - py)
        if dist < best_dist:
            best_dist = dist
            best_bearing = math.degrees(math.atan2(sx, sy)) % 360
    if best_dist > max_dist_km or best_bearing is None:
        return None, best_dist
    return (best_bearing + 90.0) % 360.0, best_dist


# ----------------------------------------------------------------------------
# Timestamp OCR
# ----------------------------------------------------------------------------
TIMESTAMP_PATTERNS = [
    (re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?"),
     ("y", "mo", "d", "h", "mi", "s")),
    (re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})\.?\s+(\d{1,2}):(\d{2})(?::(\d{2}))?"),
     ("d", "mo", "y", "h", "mi", "s")),
    (re.compile(r"(\d{1,2})/(\d{1,2})/(\d{2,4})\s+(\d{1,2}):(\d{2})"),
     ("d", "mo", "y", "h", "mi", None)),
]


def parse_timestamp_text(text, tz):
    if not text:
        return None
    t = re.sub(r"[^\d.:/\- ]", " ", text.strip())
    for pat, keys in TIMESTAMP_PATTERNS:
        m = pat.search(t)
        if not m:
            continue
        parts = {}
        for k, v in zip(keys, m.groups()):
            if k is None:
                continue
            parts[k] = int(v) if v is not None else 0
        if "y" not in parts:
            continue
        y = parts["y"] + (2000 if parts["y"] < 100 else 0)
        try:
            return datetime(y, parts["mo"], parts["d"], parts["h"],
                            parts["mi"], parts.get("s", 0),
                            tzinfo=tz).astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _otsu(arr):
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b, w_b = 0.0, 0
    best_t, best_var = 128, -1.0
    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, i
    return best_t


def read_frame_timestamp(img, region, tz, source="last"):
    if not HAVE_TESSERACT or region is None:
        return None, None, 0
    try:
        nf = getattr(img, "n_frames", 1)
        idx = 0 if source == "first" else max(0, nf - 1)
        try:
            img.seek(idx)
        except (EOFError, OSError, ValueError):
            idx = 0
            img.seek(0)
        frame = img.convert("RGB")
        crop = frame.crop(region)
        w, h = crop.size
        if w < 10 or h < 5:
            return None, None, idx
        crop = crop.resize((w * 3, h * 3), Image.LANCZOS)
        gray = np.array(crop.convert("L"))
        t = _otsu(gray)
        binary = ((gray > t) * 255).astype(np.uint8)
        if binary.mean() < 128:
            binary = 255 - binary
        text = pytesseract.image_to_string(
            Image.fromarray(binary), config="--psm 7").strip()
        return parse_timestamp_text(text, tz), text, idx
    except Exception:
        return None, None, 0
    finally:
        try:
            img.seek(0)
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Geometry + calibration
# ----------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_label(deg):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[int((deg + 11.25) % 360 // 22.5)]


@dataclass
class MapRegion:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self): return self.x1 - self.x0
    @property
    def height(self): return self.y1 - self.y0

    def crop_mask(self, full_h, full_w):
        m = np.zeros((full_h, full_w), dtype=bool)
        y0, y1 = max(0, self.y0), min(full_h, self.y1)
        x0, x1 = max(0, self.x0), min(full_w, self.x1)
        m[y0:y1, x0:x1] = True
        return m


def validate_map_against_image(mr, size):
    w, h = size
    probs = []
    if mr.x0 < 0 or mr.y0 < 0:
        probs.append(f"origin ({mr.x0},{mr.y0}) negative")
    if mr.x1 > w: probs.append(f"x1={mr.x1} > width {w}")
    if mr.y1 > h: probs.append(f"y1={mr.y1} > height {h}")
    if mr.x1 <= mr.x0 or mr.y1 <= mr.y0: probs.append("empty region")
    if probs:
        raise ValueError(f"map region invalid for {w}x{h}: " + "; ".join(probs))


def parse_map_region(raw, image_size=None):
    if raw and "map" in raw:
        m = raw["map"]
        return MapRegion(int(m["x0"]), int(m["y0"]), int(m["x1"]), int(m["y1"]))
    if image_size:
        return MapRegion(0, 0, image_size[0], image_size[1])
    return None


VALID_EDGES = {"nw", "ne", "sw", "se"}


def _resolve_edge(name, mr):
    name = name.lower()
    if name not in VALID_EDGES:
        raise ValueError(f"Unknown edge '{name}'")
    return (0 if name in ("nw", "sw") else mr.width - 1,
            0 if name in ("nw", "ne") else mr.height - 1)


class Calibration:
    def __init__(self, points, map_region, anchor_cities=None):
        if len(points) < 3:
            raise ValueError(f"Need >=3 anchors, got {len(points)}")
        A, bl, bo = [], [], []
        for x, y, lat, lon in points:
            A.append([x, y, 1.0]); bl.append(lat); bo.append(lon)
        A = np.asarray(A, dtype=float)
        self.coef_lat, *_ = np.linalg.lstsq(A, np.asarray(bl), rcond=None)
        self.coef_lon, *_ = np.linalg.lstsq(A, np.asarray(bo), rcond=None)
        self.map_region = map_region
        self.anchor_cities = anchor_cities or []
        self._inv = np.linalg.inv(np.array([
            [self.coef_lat[0], self.coef_lat[1]],
            [self.coef_lon[0], self.coef_lon[1]],
        ]))

    def image_pixel_to_latlon(self, ix, iy):
        lx, ly = ix - self.map_region.x0, iy - self.map_region.y0
        return (float(self.coef_lat[0] * lx + self.coef_lat[1] * ly + self.coef_lat[2]),
                float(self.coef_lon[0] * lx + self.coef_lon[1] * ly + self.coef_lon[2]))

    def latlon_to_image_pixel(self, lat, lon):
        rhs = np.array([lat - self.coef_lat[2], lon - self.coef_lon[2]])
        xy = self._inv @ rhs
        return float(xy[0]) + self.map_region.x0, float(xy[1]) + self.map_region.y0

    def km_per_pixel(self, ix, iy):
        lat0, lon0 = self.image_pixel_to_latlon(ix, iy)
        lat1, lon1 = self.image_pixel_to_latlon(ix + 1, iy)
        lat2, lon2 = self.image_pixel_to_latlon(ix, iy + 1)
        return (haversine_km(lat0, lon0, lat1, lon1)
                + haversine_km(lat0, lon0, lat2, lon2)) / 2.0

    def latlon_offset_km(self, px, py, t_lat, t_lon):
        lat = self.coef_lat[0] * px + self.coef_lat[1] * py + self.coef_lat[2]
        lon = self.coef_lon[0] * px + self.coef_lon[1] * py + self.coef_lon[2]
        coslat = math.cos(math.radians(t_lat))
        return (lon - t_lon) * 111.32 * coslat, (lat - t_lat) * 111.32

    def radius_mask(self, h, w, t_lat, t_lon, radius_km):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        lx = xx - self.map_region.x0
        ly = yy - self.map_region.y0
        lat = self.coef_lat[0] * lx + self.coef_lat[1] * ly + self.coef_lat[2]
        lon = self.coef_lon[0] * lx + self.coef_lon[1] * ly + self.coef_lon[2]
        coslat = math.cos(math.radians(t_lat))
        dx = (lon - t_lon) * 111.32 * coslat
        dy = (lat - t_lat) * 111.32
        return (dx * dx + dy * dy) <= (radius_km * radius_km)


def load_calibration(path, inline, image_size=None):
    if inline:
        try:
            raw = json.loads(inline)
        except json.JSONDecodeError:
            raw = yaml.safe_load(inline)
    elif path:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    else:
        raise ValueError("No calibration given")
    if not raw or "points" not in raw:
        raise ValueError("Calibration missing 'points'")
    mr = parse_map_region(raw, image_size)
    if mr is None:
        raise ValueError("Calibration has no 'map' block and no image dims.")
    if image_size:
        validate_map_against_image(mr, image_size)
    used, anchors = [], []
    for i, entry in enumerate(raw["points"], start=1):
        edge_key = "edge" if "edge" in entry else "corner" if "corner" in entry else None
        has_pixel, has_edge = "pixel" in entry, edge_key is not None
        has_city, has_ll = "city" in entry, "latlon" in entry
        if has_city and has_ll:
            raise ValueError(f"Anchor #{i}: city or latlon, not both")
        if not (has_city or has_ll):
            raise ValueError(f"Anchor #{i}: need city or latlon")
        if has_pixel == has_edge:
            raise ValueError(f"Anchor #{i}: exactly one of pixel/edge")
        if has_pixel:
            px = entry["pixel"]; x, y = float(px[0]), float(px[1])
        else:
            x, y = _resolve_edge(entry[edge_key], mr)
        if has_city:
            r = _city_lookup(entry["city"])
            if not r:
                raise ValueError(f"Anchor #{i}: unknown city '{entry['city']}'")
            name, lat, lon = r
            anchors.append((city_label(name), lat, lon))
        else:
            lat, lon = float(entry["latlon"][0]), float(entry["latlon"][1])
        used.append((x, y, lat, lon))
    if len(used) < 3:
        raise ValueError(f"Only {len(used)} complete anchor(s); need >=3")
    return Calibration(used, mr, anchor_cities=anchors)


# ----------------------------------------------------------------------------
# SRI legend
# ----------------------------------------------------------------------------
class SRILegend:
    def __init__(self, region, orientation="horizontal",
                 breaks=None, break_positions=None, padding_px=1):
        if not breaks:
            raise ValueError("SRILegend needs a non-empty `breaks` list")
        if break_positions is not None and len(break_positions) != len(breaks):
            raise ValueError("break_positions length must match breaks")
        self.region = tuple(region)
        self.orientation = orientation
        self.breaks = list(breaks)
        if break_positions is None:
            n = len(breaks)
            self.break_positions = [(i / (n - 1)) if n > 1 else 0.0 for i in range(n)]
        else:
            self.break_positions = list(break_positions)
        self.padding_px = int(padding_px)
        self.samples_rgb = None
        self.samples_mmh = None
        self.tree = None
        self._ready = False

    def calibrate(self, frame):
        x0, y0, x1, y1 = self.region
        crop = frame.crop((x0, y0, x1, y1))
        arr = np.asarray(crop.convert("RGB"), dtype=np.float32)
        h, w = arr.shape[:2]
        pad = self.padding_px
        if self.orientation == "horizontal":
            usable = arr[:, pad:w - pad, :] if w > 2 * pad + 2 else arr
            bar_length = usable.shape[1]; axis = 1
        else:
            usable = arr[pad:h - pad, :, :] if h > 2 * pad + 2 else arr
            bar_length = usable.shape[0]; axis = 0
        if bar_length < 2:
            raise ValueError(f"SRI legend bar too short ({bar_length} px)")
        sample_rgb, sample_mmh = [], []
        for mmh, pos in zip(self.breaks, self.break_positions):
            idx = int(round(pos * (bar_length - 1)))
            idx = max(0, min(bar_length - 1, idx))
            lo, hi = max(0, idx - 1), min(bar_length, idx + 2)
            if axis == 1:
                color = usable[:, lo:hi, :].mean(axis=(0, 1))
            else:
                color = usable[lo:hi, :, :].mean(axis=(0, 1))
            sample_rgb.append(color); sample_mmh.append(mmh)
        self.samples_rgb = np.array(sample_rgb, dtype=np.float64)
        self.samples_mmh = np.array(sample_mmh, dtype=np.float64)
        self.tree = cKDTree(self.samples_rgb)
        self._ready = True
        for i in range(1, len(self.samples_rgb)):
            d = float(np.linalg.norm(self.samples_rgb[i] - self.samples_rgb[i - 1]))
            if d < 15:
                print(f"WARNING: legend samples at breaks {i-1} and {i} "
                      f"({self.breaks[i-1]:g} and {self.breaks[i]:g} mm/h) "
                      f"nearly identical (RGB dist {d:.1f}).", file=sys.stderr)

    def is_ready(self):
        return self._ready

    def decode(self, rgb_pixels):
        if not self._ready or len(rgb_pixels) == 0:
            return np.zeros(len(rgb_pixels), dtype=np.float32)
        _, indices = self.tree.query(rgb_pixels)
        return self.samples_mmh[indices].astype(np.float32)

    def print_samples(self):
        if not self._ready:
            print("SRI legend: not calibrated"); return
        print(f"SRI legend ({self.orientation}, {len(self.breaks)} breaks):")
        for rgb, mmh, pos in zip(self.samples_rgb, self.samples_mmh, self.break_positions):
            r, g, b = rgb.astype(int)
            print(f"  pos={pos:.3f}  #{r:02x}{g:02x}{b:02x} → {mmh:g} mm/h")


# ----------------------------------------------------------------------------
# Radar image analysis
# ----------------------------------------------------------------------------
def fetch_image(url, timeout=30):
    r = requests.get(url, headers=RADAR_HEADERS, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content))


def compute_rgb_and_mask(frame):
    rgb = np.asarray(frame.convert("RGB"), dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn
    sat = np.where(mx > 0, diff / mx, 0)
    val = mx / 255.0
    precip = (sat >= MIN_SAT) & (val >= MIN_VAL)
    return rgb, precip


def analyze_frame(frame, map_region, intensity_mode, sri_legend, legend_mask):
    rgb, precip = compute_rgb_and_mask(frame)
    precip &= map_region.crop_mask(precip.shape[0], precip.shape[1])
    if legend_mask is not None and legend_mask.shape == precip.shape:
        precip &= ~legend_mask
    mmh = np.zeros(precip.shape, dtype=np.float32)
    ys, xs = np.where(precip)
    if len(ys) > 0:
        rgbs = rgb[ys, xs]
        uniq, inv = np.unique(rgbs, axis=0, return_inverse=True)
        inv = inv.ravel()
        if intensity_mode != "simple" and sri_legend and sri_legend.is_ready():
            mmh[ys, xs] = sri_legend.decode(uniq)[inv]
        else:
            _, indices = _DHMZ_SRI_TREE.query(uniq)
            mmh[ys, xs] = _DHMZ_SRI_MIDPOINTS[indices][inv]
    return precip, mmh


# ----------------------------------------------------------------------------
# Storm Frame Archiver (deduplicated PNGs grouped by storm episode)
# ----------------------------------------------------------------------------
class StormArchiver:
    """
    Archives individual deduplicated radar PNG frames whenever precipitation
    occurs within `archive_radius_km` of the target location.

    Each frame is timestamped by its exact radar frame UTC timestamp
    (so overlapping 30-frame GIF polls never store duplicate frames, while
    automatic historical backfill still captures the full 2.5h ramp-up of a storm).
    Frames within 45 minutes of one another are grouped into an episode folder
    named `<YYYY-MM-DD_HHMMZ>` alongside an `event.json` summary.
    """
    EPISODE_GAP_MIN = 45.0

    def __init__(self, data_dir, calib, target_lat, target_lon, target_name):
        self.data_dir = data_dir
        self.root_dir = os.path.join(data_dir, "archive") if data_dir else None
        self.calib = calib
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.target_name = target_name
        self.lock = threading.Lock()
        self._mask_cache = {}  # (h, w, round(radius_km, 1)) -> bool ndarray
        self.total_frames_saved = 0
        self.total_episodes = 0
        self.last_episode_id = None
        self.last_saved_utc = None
        if self.root_dir:
            try:
                os.makedirs(self.root_dir, exist_ok=True)
                self._scan_existing()
            except OSError as e:
                print(f"archive: cannot create {self.root_dir}: {e}", file=sys.stderr)
                self.root_dir = None

    def _scan_existing(self):
        if not self.root_dir or not os.path.isdir(self.root_dir):
            return
        episodes = sorted(
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        )
        self.total_episodes = len(episodes)
        if episodes:
            self.last_episode_id = episodes[-1]
        count = 0
        for ep in episodes:
            ep_dir = os.path.join(self.root_dir, ep)
            try:
                count += sum(1 for f in os.listdir(ep_dir) if f.endswith(".png"))
            except OSError:
                pass
        self.total_frames_saved = count

    def _get_radius_mask(self, h, w, radius_km):
        key = (h, w, round(float(radius_km), 1))
        m = self._mask_cache.get(key)
        if m is None:
            m = self.calib.radius_mask(h, w, self.target_lat, self.target_lon, radius_km)
            if len(self._mask_cache) > 8:
                self._mask_cache.clear()
            self._mask_cache[key] = m
        return m

    def _parse_episode_start(self, ep_name):
        try:
            return datetime.strptime(ep_name, "%Y-%m-%d_%H%MZ").replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    def _resolve_episode_dir(self, frame_ts_utc):
        """Find an existing episode whose time range covers or borders `frame_ts_utc`."""
        best_ep = None
        best_gap = float("inf")
        for d in sorted(os.listdir(self.root_dir)):
            ep_dir = os.path.join(self.root_dir, d)
            if not os.path.isdir(ep_dir):
                continue
            meta_path = os.path.join(ep_dir, "event.json")
            meta = _load_json_file(meta_path)
            start_dt = _safe_fromiso(meta.get("start_utc")) if meta else self._parse_episode_start(d)
            end_dt = _safe_fromiso(meta.get("end_utc")) if meta else start_dt
            if start_dt == datetime.min.replace(tzinfo=timezone.utc):
                continue
            if (start_dt - timedelta(minutes=self.EPISODE_GAP_MIN)) <= frame_ts_utc <= (
                    end_dt + timedelta(minutes=self.EPISODE_GAP_MIN)):
                gap = abs((frame_ts_utc - end_dt).total_seconds()) / 60.0
                if gap < best_gap:
                    best_gap = gap
                    best_ep = d

        if best_ep is not None:
            return best_ep, os.path.join(self.root_dir, best_ep), False

        ep_id = frame_ts_utc.strftime("%Y-%m-%d_%H%MZ")
        ep_dir = os.path.join(self.root_dir, ep_id)
        os.makedirs(ep_dir, exist_ok=True)
        return ep_id, ep_dir, True

    def process_animation(self, img, frame_masks_mmh, last_frame_ts_utc,
                          eff_interval_min, fallback_now_utc, rt,
                          local_weather=None, target_raining=False):
        """
        Inspect all frames in the fetched GIF. Any frame with >= `archive_min_pixels`
        of precipitation inside `archive_radius_km` is saved as a deduplicated
        palette-optimized PNG inside its storm episode directory.
        """
        if not self.root_dir or not rt.get("archive_enabled", True):
            return 0

        radius_km = float(rt.get("archive_radius_km", 200.0))
        min_px = int(rt.get("archive_min_pixels", 40))
        n_frames = len(frame_masks_mmh)
        if n_frames == 0:
            return 0

        anchor_utc = last_frame_ts_utc or fallback_now_utc
        # Round anchor to nearest minute for deterministic filenames when OCR is unavailable
        anchor_utc = anchor_utc.replace(second=0, microsecond=0)

        saved_this_cycle = 0
        with self.lock:
            for idx, (mask, mmh) in enumerate(frame_masks_mmh):
                h, w = mask.shape
                rmask = self._get_radius_mask(h, w, radius_km)
                local_precip = mask & rmask
                px_count = int(local_precip.sum())
                if px_count < min_px:
                    continue

                minutes_before_last = (n_frames - 1 - idx) * eff_interval_min
                frame_ts = anchor_utc - timedelta(minutes=round(minutes_before_last))
                fname = frame_ts.strftime("%Y%m%d_%H%M%SZ.png")

                ep_id, ep_dir, is_new_ep = self._resolve_episode_dir(frame_ts)
                if is_new_ep:
                    self.total_episodes += 1

                fpath = os.path.join(ep_dir, fname)
                if os.path.exists(fpath):
                    continue

                try:
                    img.seek(idx)
                    frame_pil = img.convert("P", palette=Image.ADAPTIVE, colors=64)
                    tmp_path = fpath + ".tmp"
                    frame_pil.save(tmp_path, format="PNG", optimize=True)
                    os.replace(tmp_path, fpath)
                except Exception as e:
                    print(f"archive: frame save error {e}", file=sys.stderr)
                    continue

                peak_mmh = float(mmh[local_precip].max()) if px_count > 0 else 0.0
                mean_mmh = float(mmh[local_precip].mean()) if px_count > 0 else 0.0
                self._update_episode_meta(
                    ep_dir, ep_id, frame_ts, radius_km, px_count,
                    peak_mmh, mean_mmh,
                    local_weather if idx == n_frames - 1 else None,
                    target_raining if idx == n_frames - 1 else False,
                )
                saved_this_cycle += 1
                self.total_frames_saved += 1
                self.last_episode_id = ep_id
                self.last_saved_utc = frame_ts.isoformat()

        try:
            img.seek(0)
        except Exception:
            pass
        return saved_this_cycle

    def _update_episode_meta(self, ep_dir, ep_id, frame_ts, radius_km,
                             px_count, peak_mmh, mean_mmh,
                             local_weather, target_raining):
        meta_path = os.path.join(ep_dir, "event.json")
        meta = _load_json_file(meta_path) or {
            "episode_id": ep_id,
            "target": self.target_name,
            "target_lat": self.target_lat,
            "target_lon": self.target_lon,
            "archive_radius_km": radius_km,
            "start_utc": frame_ts.isoformat(),
            "end_utc": frame_ts.isoformat(),
            "frame_count": 0,
            "max_precip_pixels": 0,
            "peak_sri_mmh": 0.0,
            "target_hit": False,
        }
        ts_iso = frame_ts.isoformat()
        if ts_iso < meta["start_utc"]:
            meta["start_utc"] = ts_iso
        if ts_iso > meta["end_utc"]:
            meta["end_utc"] = ts_iso
        try:
            meta["frame_count"] = sum(
                1 for f in os.listdir(ep_dir) if f.endswith(".png")
            )
        except OSError:
            meta["frame_count"] = meta.get("frame_count", 0) + 1
        meta["max_precip_pixels"] = max(int(meta.get("max_precip_pixels", 0)), px_count)
        meta["peak_sri_mmh"] = round(max(float(meta.get("peak_sri_mmh", 0.0)), peak_mmh), 1)
        if target_raining:
            meta["target_hit"] = True
        if local_weather and local_weather.station_count > 0:
            meta["latest_local_weather"] = {
                "temp_c": local_weather.temp_c,
                "pressure_hpa": local_weather.pressure_hpa,
                "wind_speed_kmh": local_weather.wind_speed_kmh,
                "wind_gust_kmh": local_weather.wind_gust_kmh,
                "wind_dir_deg": local_weather.wind_dir_deg,
                "precip_today_mm": local_weather.precip_today_mm,
            }
        _save_json_file(meta_path, meta)

    def summary(self):
        with self.lock:
            return {
                "enabled": bool(self.root_dir and _snapshot().get("archive_enabled", True)),
                "archive_dir": self.root_dir,
                "radius_km": _snapshot().get("archive_radius_km", 200.0),
                "total_episodes": self.total_episodes,
                "total_frames": self.total_frames_saved,
                "last_episode": self.last_episode_id,
                "last_saved_utc": self.last_saved_utc,
            }

    def list_episodes(self, limit=30):
        if not self.root_dir or not os.path.isdir(self.root_dir):
            return []
        out = []
        with self.lock:
            dirs = sorted(
                (d for d in os.listdir(self.root_dir)
                 if os.path.isdir(os.path.join(self.root_dir, d))),
                reverse=True,
            )[:limit]
            for d in dirs:
                ep_dir = os.path.join(self.root_dir, d)
                meta = _load_json_file(os.path.join(ep_dir, "event.json"))
                if meta:
                    out.append(meta)
                else:
                    n = sum(1 for f in os.listdir(ep_dir) if f.endswith(".png"))
                    out.append({"episode_id": d, "frame_count": n})
        return out


# ----------------------------------------------------------------------------
# Clustering + tracking
# ----------------------------------------------------------------------------
@dataclass
class PixelCluster:
    cx: float
    cy: float
    size: int
    label: int = 0
    max_mmh: float = 0.0
    mean_mmh: float = 0.0


def clusters_from_mask(mask, min_pixels, mmh=None):
    label_array, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return [], label_array
    idx = np.arange(1, n + 1)
    sizes = ndimage.sum(mask, label_array, idx)
    geom_cents = ndimage.center_of_mass(mask, label_array, idx)
    if mmh is not None:
        weights = np.where(mask, np.maximum(mmh, 0.1), 0.0)
        cents = ndimage.center_of_mass(weights, label_array, idx)
        max_vals = ndimage.maximum(mmh, label_array, idx)
        mean_vals = ndimage.mean(mmh, label_array, idx)
    else:
        cents = geom_cents
        max_vals = np.zeros(n, dtype=np.float32)
        mean_vals = np.zeros(n, dtype=np.float32)
    out = []
    for lbl, sz, (cy, cx), (gcy, gcx), mx_v, mn_v in zip(
            idx, sizes, cents, geom_cents, max_vals, mean_vals):
        if sz < min_pixels:
            continue
        if not (math.isfinite(cx) and math.isfinite(cy)):
            cx, cy = gcx, gcy
        out.append(PixelCluster(float(cx), float(cy), int(sz), int(lbl),
                                float(mx_v), float(mn_v)))
    return out, label_array


@dataclass
class Track:
    cluster: PixelCluster
    cid: int = 0
    vx_kmh: float = 0.0
    vy_kmh: float = 0.0
    n_observations: int = 1
    span_frames: int = 0
    prev_size: Optional[int] = None
    event: Optional[str] = None

    @property
    def speed_kmh(self):
        return math.hypot(self.vx_kmh, self.vy_kmh)


def _select_recent_window(obs, frame_interval_min, split_frames=None):
    if len(obs) <= 1:
        return obs
    if split_frames:
        last_fidx = obs[-1][0]
        latest_split = max((sf for sf in split_frames if sf <= last_fidx), default=None)
        if latest_split is not None:
            post_split = [o for o in obs if o[0] >= latest_split]
            if len(post_split) >= 2:
                obs = post_split

    last_fidx = obs[-1][0]
    max_frame_span = max(1, int(round(VELOCITY_WINDOW_MIN / max(1.0, frame_interval_min))))
    cutoff_fidx = last_fidx - max_frame_span
    window = [o for o in obs if o[0] >= cutoff_fidx]
    if len(window) < 2:
        window = obs[-2:]
    return window


def _estimate_pixel_velocity(obs, frame_interval_min, split_frames=None):
    window = _select_recent_window(obs, frame_interval_min, split_frames)
    if len(window) < 2:
        return 0.0, 0.0
    ts = np.array([o[0] for o in window], dtype=np.float64)
    xs = np.array([o[1].cx for o in window], dtype=np.float64)
    ys = np.array([o[1].cy for o in window], dtype=np.float64)
    ws = np.sqrt(np.maximum(1.0, np.array([o[1].size for o in window], dtype=np.float64)))
    w_sum = float(np.sum(ws))
    t_mean = float(np.sum(ws * ts) / w_sum)
    dt = ts - t_mean
    denom = float(np.sum(ws * dt * dt))
    if denom <= 1e-9:
        return 0.0, 0.0
    x_mean = float(np.sum(ws * xs) / w_sum)
    y_mean = float(np.sum(ws * ys) / w_sum)
    vx_px = float(np.sum(ws * dt * (xs - x_mean)) / denom)
    vy_px = float(np.sum(ws * dt * (ys - y_mean)) / denom)
    return vx_px, vy_px


def _estimate_velocity_kmh(obs, calib, frame_interval_min, split_frames=None):
    window = _select_recent_window(obs, frame_interval_min, split_frames)
    if len(window) < 2:
        return 0.0, 0.0

    lat0, lon0 = calib.image_pixel_to_latlon(window[0][1].cx, window[0][1].cy)
    coslat = max(0.01, math.cos(math.radians(lat0)))

    ts_h = np.empty(len(window), dtype=np.float64)
    xs_km = np.empty(len(window), dtype=np.float64)
    ys_km = np.empty(len(window), dtype=np.float64)
    ws = np.empty(len(window), dtype=np.float64)

    for idx, (fidx, c) in enumerate(window):
        lat, lon = calib.image_pixel_to_latlon(c.cx, c.cy)
        ts_h[idx] = fidx * frame_interval_min / 60.0
        xs_km[idx] = (lon - lon0) * 111.32 * coslat
        ys_km[idx] = (lat - lat0) * 111.32
        ws[idx] = math.sqrt(max(1.0, float(c.size)))

    w_sum = float(np.sum(ws))
    t_mean = float(np.sum(ws * ts_h) / w_sum)
    dt = ts_h - t_mean
    denom = float(np.sum(ws * dt * dt))
    if denom <= 1e-9:
        return 0.0, 0.0

    x_mean = float(np.sum(ws * xs_km) / w_sum)
    y_mean = float(np.sum(ws * ys_km) / w_sum)
    vx = float(np.sum(ws * dt * (xs_km - x_mean)) / denom)
    vy = float(np.sum(ws * dt * (ys_km - y_mean)) / denom)

    speed = math.hypot(vx, vy)
    if speed > MAX_STORM_SPEED_KMH:
        scale = MAX_STORM_SPEED_KMH / speed
        vx *= scale
        vy *= scale
    return vx, vy


def track_animation(img, calib, frame_interval_min, target_lat, target_lon,
                    intensity_mode, sri_legend, legend_mask):
    tracks_by_id = {}
    split_frames_by_id = {}
    next_id = 0
    prev_clusters, prev_ids = [], []
    n_frames = 0
    rain_history = []
    frame_masks_mmh = []
    last_label_array = None
    last_mask = last_mmh = None
    last_frame_events = {}
    cluster_first_target_hit = {}

    mid_x = (calib.map_region.x0 + calib.map_region.x1) / 2.0
    mid_y = (calib.map_region.y0 + calib.map_region.y1) / 2.0
    kmpp = max(0.05, calib.km_per_pixel(mid_x, mid_y))
    max_step_km = MAX_STORM_SPEED_KMH * (max(1.0, frame_interval_min) / 60.0)
    base_max_jump_px = max(18.0, (max_step_km / kmpp) + 12.0)

    px, py = calib.latlon_to_image_pixel(target_lat, target_lon)
    xi, yi = int(round(px)), int(round(py))

    for i, frame in enumerate(ImageSequence.Iterator(img)):
        n_frames += 1
        mask, mmh = analyze_frame(frame, calib.map_region, intensity_mode,
                                  sri_legend, legend_mask)
        frame_masks_mmh.append((mask, mmh))
        last_mask, last_mmh = mask, mmh
        h, w = mask.shape
        if 0 <= yi < h and 0 <= xi < w:
            rain_history.append(bool(mask[max(0, yi-1):yi+2, max(0, xi-1):xi+2].any()))
        else:
            rain_history.append(False)
        clusters, label_array = clusters_from_mask(mask, MIN_CLUSTER_PIXELS, mmh=mmh)
        last_label_array = label_array
        if not clusters:
            prev_clusters, prev_ids = [], []
            last_frame_events = {}
            continue

        predictions = []
        for j, p in enumerate(prev_clusters):
            pid = prev_ids[j]
            obs = tracks_by_id[pid]
            dt_f = max(1, i - obs[-1][0])
            if len(obs) >= 2:
                vx_px, vy_px = _estimate_pixel_velocity(
                    obs, frame_interval_min, split_frames_by_id.get(pid))
                pred_x = p.cx + vx_px * dt_f
                pred_y = p.cy + vy_px * dt_f
                allowed_px = base_max_jump_px * dt_f
            else:
                pred_x, pred_y = p.cx, p.cy
                allowed_px = base_max_jump_px * 1.25 * dt_f
            predictions.append((pred_x, pred_y, allowed_px))

        candidates = []
        for k, c in enumerate(clusters):
            for j, p in enumerate(prev_clusters):
                pred_x, pred_y, allowed_px = predictions[j]
                d = math.hypot(c.cx - pred_x, c.cy - pred_y)
                if d > allowed_px:
                    continue
                size_ratio = max(c.size, p.size) / max(1, min(c.size, p.size))
                cost = d * (1.0 + 0.25 * max(0.0, size_ratio - 1.5))
                candidates.append((cost, j, k))
        candidates.sort(key=lambda item: item[0])

        matched_prev = {}
        matched_curr = {}
        for _, j, k in candidates:
            if j in matched_prev or k in matched_curr:
                continue
            matched_prev[j] = k
            matched_curr[k] = prev_ids[j]

        events = {}
        for k, c in enumerate(clusters):
            if k in matched_curr:
                continue
            best_parent_cid = None
            best_d = base_max_jump_px * 0.85
            for j, p in enumerate(prev_clusters):
                if j not in matched_prev:
                    continue
                pred_x, pred_y, _ = predictions[j]
                d = math.hypot(c.cx - pred_x, c.cy - pred_y)
                if d < best_d:
                    best_d = d
                    best_parent_cid = prev_ids[j]
            if best_parent_cid is not None:
                split_frames_by_id.setdefault(best_parent_cid, set()).add(i)
                for m_k, m_cid in matched_curr.items():
                    if m_cid == best_parent_cid:
                        events[m_cid] = "split"

        curr_ids = []
        for k, c in enumerate(clusters):
            if k in matched_curr:
                cid = matched_curr[k]
                prev_obs_size = tracks_by_id[cid][-1][1].size
                ratio = c.size / max(1, prev_obs_size)
                if ratio > 2.5 or ratio < 0.4:
                    split_frames_by_id.setdefault(cid, set()).add(i)
            else:
                cid = next_id
                next_id += 1
                if any(math.hypot(c.cx - predictions[j][0], c.cy - predictions[j][1])
                       < base_max_jump_px * 0.85
                       for j in matched_prev):
                    events[cid] = "split"
            curr_ids.append(cid)
            tracks_by_id.setdefault(cid, []).append((i, c))

        last_frame_events = events
        if 0 <= yi < h and 0 <= xi < w and label_array[yi, xi] > 0:
            target_lbl = int(label_array[yi, xi])
            for k, c in enumerate(clusters):
                if c.label == target_lbl:
                    cid_here = curr_ids[k]
                    if cid_here not in cluster_first_target_hit:
                        cluster_first_target_hit[cid_here] = i
                    break
        prev_clusters, prev_ids = clusters, curr_ids

    tracks = []
    for j, c in enumerate(prev_clusters):
        tid = prev_ids[j]
        obs = tracks_by_id[tid]
        dt_frames = obs[-1][0] - obs[0][0]
        tr = Track(cluster=c, cid=tid, n_observations=len(obs), span_frames=dt_frames)
        if dt_frames > 0:
            tr.vx_kmh, tr.vy_kmh = _estimate_velocity_kmh(
                obs, calib, frame_interval_min, split_frames_by_id.get(tid))
        if len(obs) >= 2:
            tr.prev_size = obs[-2][1].size
        tr.event = last_frame_events.get(tid)
        tracks.append(tr)
    return (last_mask, last_mmh, tracks, n_frames, rain_history,
            last_label_array, tracks_by_id, split_frames_by_id,
            cluster_first_target_hit, frame_masks_mmh)


# ----------------------------------------------------------------------------
# Approach + nowcast
# ----------------------------------------------------------------------------
@dataclass
class Approach:
    dist_km: float
    bearing_to_target: float
    bearing_of_motion: Optional[float]
    approach_angle: Optional[float]
    closest_km: Optional[float]
    eta_min: Optional[float]
    approaching: bool
    cluster_radius_km: float = 0.0


@dataclass
class Nowcast:
    raining_now: bool = False
    rain_duration_min: Optional[float] = None
    rain_end_eta_min: Optional[float] = None
    next_rain_eta_min: Optional[float] = None
    gap_min: Optional[float] = None
    summary: str = ""
    lines: list = field(default_factory=list)


def assess(track, target_lat, target_lon, calib):
    c = track.cluster
    clat, clon = calib.image_pixel_to_latlon(c.cx, c.cy)
    dist = haversine_km(clat, clon, target_lat, target_lon)
    kmpp = calib.km_per_pixel(c.cx, c.cy)
    c_rad = math.sqrt(max(0.0, c.size * kmpp * kmpp) / math.pi)
    coslat = math.cos(math.radians((clat + target_lat) / 2))
    dx_km = (target_lon - clon) * 111.32 * coslat
    dy_km = (target_lat - clat) * 111.32
    bearing = math.degrees(math.atan2(dx_km, dy_km)) % 360
    speed = track.speed_kmh
    if speed < MIN_SPEED_KMH:
        return Approach(dist, bearing, None, None, None, None, False, c_rad)
    bm = math.degrees(math.atan2(track.vx_kmh, track.vy_kmh)) % 360
    ux, uy = dx_km / dist, dy_km / dist
    angle = math.degrees(math.acos(max(-1.0, min(1.0,
                (track.vx_kmh * ux + track.vy_kmh * uy) / speed))))
    Rx, Ry = -dx_km, -dy_km
    t_star = -(Rx * track.vx_kmh + Ry * track.vy_kmh) / (speed * speed)
    if t_star > 0:
        cax = Rx + track.vx_kmh * t_star
        cay = Ry + track.vy_kmh * t_star
        return Approach(dist, bearing, bm, angle,
                        math.hypot(cax, cay), t_star * 60.0, True, c_rad)
    return Approach(dist, bearing, bm, angle, dist, None, False, c_rad)


def refine_approaching(track, a, rt):
    if not a.approaching:
        return False
    if a.approach_angle is None or a.closest_km is None:
        return False
    if a.approach_angle > rt["only_approach_deg"]:
        return False
    r_eff = rt["approaching_radius_km"] + a.cluster_radius_km * rt["cluster_radius_weight"]
    return a.closest_km <= r_eff


def refine_threat(track, a, rt):
    if not refine_approaching(track, a, rt):
        return False
    if track.n_observations < rt["threat_min_duration"]:
        return False
    min_sri = float(rt.get("threat_min_sri_mmh", 0.0))
    if min_sri > 0.0 and track.cluster.max_mmh < min_sri:
        return False
    if a.approach_angle is None or a.approach_angle > rt["threat_angle_deg"]:
        return False
    r_eff = rt["threat_radius_km"] + a.cluster_radius_km * rt["cluster_radius_weight"]
    return a.closest_km is not None and a.closest_km <= r_eff


def _project_along(calib, pixels_yx, t_lat, t_lon, ux, uy):
    dx, dy = calib.latlon_offset_km(pixels_yx[:, 1].astype(float),
                                    pixels_yx[:, 0].astype(float), t_lat, t_lon)
    return dx * ux + dy * uy


def predict_end_min(calib, pixels, t_lat, t_lon, vx, vy):
    speed = math.hypot(vx, vy)
    if speed < MIN_SPEED_KMH or len(pixels) == 0:
        return None
    proj = _project_along(calib, pixels, t_lat, t_lon, vx / speed, vy / speed)
    mn = float(proj.min())
    return 0.0 if mn >= 0 else -mn / speed * 60.0


def predict_arrival_min(calib, pixels, t_lat, t_lon, vx, vy):
    speed = math.hypot(vx, vy)
    if speed < MIN_SPEED_KMH or len(pixels) == 0:
        return None
    proj = _project_along(calib, pixels, t_lat, t_lon, vx / speed, vy / speed)
    mx = float(proj.max())
    return 0.0 if mx >= 0 else -mx / speed * 60.0


def is_upwind(calib, cluster, t_lat, t_lon, vx, vy):
    speed = math.hypot(vx, vy)
    if speed < MIN_SPEED_KMH:
        return False
    clat, clon = calib.image_pixel_to_latlon(cluster.cx, cluster.cy)
    coslat = math.cos(math.radians(t_lat))
    dx = (clon - t_lon) * 111.32 * coslat
    dy = (clat - t_lat) * 111.32
    return (dx * vx / speed + dy * vy / speed) < 0


def _find_upwind_candidates(calib, tracks, approaches, label_array,
                            t_lat, t_lon, rt, exclude_label=None):
    candidates = []
    for i, tr in enumerate(tracks):
        if exclude_label is not None and tr.cluster.label == exclude_label:
            continue
        if tr.speed_kmh < MIN_SPEED_KMH:
            continue
        if not is_upwind(calib, tr.cluster, t_lat, t_lon, tr.vx_kmh, tr.vy_kmh):
            continue
        a = approaches[i]
        if not refine_approaching(tr, a, rt):
            continue
        px_arr = np.argwhere(label_array == tr.cluster.label)
        arr = predict_arrival_min(calib, px_arr, t_lat, t_lon, tr.vx_kmh, tr.vy_kmh)
        if arr is not None and arr > 0:
            candidates.append((arr, i, tr))
    candidates.sort(key=lambda x: x[0])
    return candidates


def compute_nowcast(calib, t_lat, t_lon, tracks, approaches, rain_history,
                    label_array, interval_min, rt):
    nc = Nowcast()
    nc.raining_now = rain_history[-1] if rain_history else False
    if nc.raining_now:
        streak = 0
        for r in reversed(rain_history):
            if r: streak += 1
            else: break
        nc.rain_duration_min = streak * interval_min
        px, py = calib.latlon_to_image_pixel(t_lat, t_lon)
        xi, yi = int(round(px)), int(round(py))
        cov_label = 0
        if 0 <= yi < label_array.shape[0] and 0 <= xi < label_array.shape[1]:
            cov_label = int(label_array[yi, xi])
        cov_tr = next((t for t in tracks if t.cluster.label == cov_label), None)
        if cov_tr and cov_tr.speed_kmh >= MIN_SPEED_KMH:
            px_arr = np.argwhere(label_array == cov_label)
            nc.rain_end_eta_min = predict_end_min(calib, px_arr, t_lat, t_lon,
                                                  cov_tr.vx_kmh, cov_tr.vy_kmh)
        candidates = _find_upwind_candidates(calib, tracks, approaches, label_array,
                                             t_lat, t_lon, rt, exclude_label=cov_label)
        if candidates:
            nc.next_rain_eta_min, ci, tr = candidates[0]
            if (nc.rain_end_eta_min is not None
                    and nc.next_rain_eta_min > nc.rain_end_eta_min):
                nc.gap_min = nc.next_rain_eta_min - nc.rain_end_eta_min
    else:
        nc.rain_duration_min = 0.0
        candidates = _find_upwind_candidates(calib, tracks, approaches, label_array,
                                             t_lat, t_lon, rt)
        if candidates:
            nc.next_rain_eta_min, ci, tr = candidates[0]
    if nc.raining_now:
        parts = [f"Raining {nc.rain_duration_min:.0f} min."]
        if nc.rain_end_eta_min is not None:
            parts.append(f"Ends ~{nc.rain_end_eta_min:.0f} min.")
        if nc.gap_min is not None:
            parts.append(f"Gap ~{nc.gap_min:.0f} min, next ~{nc.next_rain_eta_min:.0f} min.")
        elif nc.next_rain_eta_min is not None:
            parts.append(f"Next cell ~{nc.next_rain_eta_min:.0f} min (no gap).")
        nc.summary = " ".join(parts)
    else:
        nc.summary = (f"Dry. Next rain ~{nc.next_rain_eta_min:.0f} min."
                      if nc.next_rain_eta_min is not None else
                      "Dry. No approaching rain.")
    return nc


# ----------------------------------------------------------------------------
# Pljusak client + full weather parsing
# ----------------------------------------------------------------------------
def _parse_opt_float(row, idx, lo=None, hi=None):
    if len(row) <= idx:
        return None
    val = row[idx]
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip().replace(",", ".")
        if not s or s in ("-", "—", "null", "None", "NaN", "nan"):
            return None
        try:
            f = float(s)
        except ValueError:
            return None
    else:
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
    if not math.isfinite(f):
        return None
    if lo is not None and f < lo:
        return None
    if hi is not None and f > hi:
        return None
    return f


def _parse_wind_dir_deg(raw_dir):
    if raw_dir is None:
        return None, None
    s = str(raw_dir).strip().upper()
    if not s or s in ("-", "—", "TIHO", "CALM", "C", "VRB", "0"):
        return s if s and s != "-" else None, None
    if s in COMPASS_16_DEG:
        return s, COMPASS_16_DEG[s]
    try:
        deg = float(s)
        if math.isfinite(deg) and 0.0 <= deg <= 360.0:
            deg = deg % 360.0
            return bearing_label(deg), deg
    except ValueError:
        pass
    return s, None


@dataclass
class StationReading:
    slug: str
    name: str
    lat: float
    lon: float
    precip_1h: float
    time_str: str
    elevation_m: Optional[float] = None
    temp_c: Optional[float] = None
    temp_trend: Optional[float] = None
    pressure_hpa: Optional[float] = None
    pressure_trend: Optional[float] = None
    humidity_pct: Optional[float] = None
    wind_dir_str: Optional[str] = None
    wind_dir_deg: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    wind_gust_kmh: Optional[float] = None
    precip_month: Optional[float] = None
    precip_today: Optional[float] = None
    precip_24h: Optional[float] = None
    precip_6h: Optional[float] = None
    max_temp_c: Optional[float] = None
    min_temp_c: Optional[float] = None
    uv: Optional[float] = None
    temp_24h_c: Optional[float] = None


@dataclass
class StationAnalysis:
    reading: StationReading
    dist_km: float
    age_min: Optional[float]
    slope_mmh: Optional[float]
    trend: str
    rain_rate_mmh: Optional[float] = None


@dataclass
class LocalWeather:
    radius_km: float = 5.0
    station_count: int = 0
    nearest_station: Optional[str] = None
    nearest_dist_km: Optional[float] = None
    temp_c: Optional[float] = None
    temp_trend: Optional[float] = None
    humidity_pct: Optional[float] = None
    pressure_hpa: Optional[float] = None
    pressure_trend: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    wind_gust_kmh: Optional[float] = None
    wind_dir_deg: Optional[int] = None
    wind_dir_cardinal: Optional[str] = None
    precip_today_mm: Optional[float] = None
    precip_1h_mm: Optional[float] = None
    precip_6h_mm: Optional[float] = None
    precip_24h_mm: Optional[float] = None
    rain_rate_mmh: Optional[float] = None


class PljusakClient:
    def __init__(self, url=PLJUSAK_URL, timeout=30):
        self.url = url
        self.timeout = timeout
        self.history = {}
        self.last_fetch_utc = None
        self.last_readings = []
        self.last_error = None

    def fetch(self, now_utc):
        try:
            r = requests.get(self.url, headers=PLJUSAK_HEADERS, timeout=self.timeout)
            r.raise_for_status()
            m = re.search(r"var\s+podaci\s*=\s*(\[.*?\]);\s*</script>", r.text, re.S)
            if not m:
                self.last_error = "regex miss"
                return False
            rows = json.loads(m.group(1))
            readings = []
            for row in rows:
                if not row or row[0] == "kamera":
                    continue
                lat = _parse_opt_float(row, 2, -90.0, 90.0)
                lon = _parse_opt_float(row, 3, -180.0, 180.0)
                if lat is None or lon is None:
                    continue

                elev_m = _parse_opt_float(row, 4, -50.0, 5000.0)
                ts = str(row[10]).strip() if len(row) > 10 and row[10] is not None else ""
                temp_c = _parse_opt_float(row, 12, -55.0, 60.0)
                temp_trend = _parse_opt_float(row, 13, -30.0, 30.0)
                pressure_hpa = _parse_opt_float(row, 14, 930.0, 1070.0)
                pressure_trend = _parse_opt_float(row, 15, -30.0, 30.0)
                humidity_pct = _parse_opt_float(row, 16, 1.0, 100.0)

                raw_dir = row[17] if len(row) > 17 else None
                wind_dir_str, wind_dir_deg = _parse_wind_dir_deg(raw_dir)

                wind_ms = _parse_opt_float(row, 18, 0.0, 90.0)
                wind_speed_kmh = (wind_ms * 3.6) if wind_ms is not None else None

                precip_month = _parse_opt_float(row, 19, 0.0, 5000.0)
                precip_today = _parse_opt_float(row, 20, 0.0, 1500.0)
                precip_24h = _parse_opt_float(row, 21, 0.0, 1500.0)
                p1h_opt = _parse_opt_float(row, 22, 0.0, 500.0)
                precip_1h = p1h_opt if p1h_opt is not None else 0.0
                precip_6h = _parse_opt_float(row, 23, 0.0, 1000.0)

                max_temp_c = _parse_opt_float(row, 24, -55.0, 60.0)
                min_temp_c = _parse_opt_float(row, 25, -55.0, 60.0)

                gust_ms = _parse_opt_float(row, 26, 0.0, 110.0)
                wind_gust_kmh = (gust_ms * 3.6) if gust_ms is not None else None

                uv = _parse_opt_float(row, 27, 0.0, 25.0)
                temp_24h_c = _parse_opt_float(row, 28, -55.0, 60.0)
                slug = str(row[29]).strip() if len(row) > 29 and row[29] is not None else ""
                name = str(row[1]).strip() if len(row) > 1 and row[1] is not None else ""

                readings.append(StationReading(
                    slug=slug, name=name, lat=lat, lon=lon,
                    precip_1h=precip_1h, time_str=ts,
                    elevation_m=elev_m, temp_c=temp_c, temp_trend=temp_trend,
                    pressure_hpa=pressure_hpa, pressure_trend=pressure_trend,
                    humidity_pct=humidity_pct, wind_dir_str=wind_dir_str,
                    wind_dir_deg=wind_dir_deg, wind_speed_kmh=wind_speed_kmh,
                    wind_gust_kmh=wind_gust_kmh, precip_month=precip_month,
                    precip_today=precip_today, precip_24h=precip_24h,
                    precip_6h=precip_6h, max_temp_c=max_temp_c,
                    min_temp_c=min_temp_c, uv=uv, temp_24h_c=temp_24h_c,
                ))
            self.last_readings = readings
            self.last_fetch_utc = now_utc
            self.last_error = None
            for st in readings:
                h = self.history.setdefault(st.slug, [])
                h.append((now_utc, st.precip_1h, st.precip_today))
                if len(h) > 60:
                    del h[:len(h) - 60]
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def analyze_nearby(self, t_lat, t_lon, radius_km, display_tz, max_age_min=15):
        out = []
        for st in self.last_readings:
            dist = haversine_km(st.lat, st.lon, t_lat, t_lon)
            if radius_km is not None and dist > radius_km:
                continue
            age = _station_age_min(st.time_str, self.last_fetch_utc, display_tz)
            slope, trend = self.slope_mmh(st.slug, window_min=12)
            rate = self.rain_rate_mmh(st.slug, window_min=12)
            out.append(StationAnalysis(st, dist, age, slope, trend, rain_rate_mmh=rate))
        out.sort(key=lambda a: a.dist_km)
        return out

    def get_all_stations(self, t_lat, t_lon, display_tz):
        return self.analyze_nearby(t_lat, t_lon, radius_km=None, display_tz=display_tz)

    def slope_mmh(self, slug, window_min=12):
        h = self.history.get(slug)
        if not h or len(h) < 2:
            return None, "insufficient"
        latest_ts, latest_mm, _ = h[-1]
        earliest = None
        for entry in reversed(h):
            if (latest_ts - entry[0]).total_seconds() >= window_min * 60:
                earliest = entry
                break
        if not earliest:
            return None, "insufficient"
        dt_h = (latest_ts - earliest[0]).total_seconds() / 3600.0
        if dt_h <= 0:
            return None, "insufficient"
        slope = (latest_mm - earliest[1]) / dt_h
        if slope > 50 or slope < -50:
            return None, "unknown"
        if slope > 0.5:
            return slope, "rising"
        if slope < -0.5:
            return slope, "falling"
        return slope, "stable"

    def rain_rate_mmh(self, slug, window_min=12):
        h = self.history.get(slug)
        if not h or len(h) < 2:
            return None
        latest_ts, latest_1h, latest_today = h[-1]
        earliest = None
        for entry in reversed(h[:-1]):
            if (latest_ts - entry[0]).total_seconds() >= window_min * 60:
                earliest = entry
                break
        if earliest is None:
            earliest = h[0]
        dt_h = (latest_ts - earliest[0]).total_seconds() / 3600.0
        if dt_h < (3.0 / 60.0):
            return None

        earliest_1h, earliest_today = earliest[1], earliest[2]
        if (latest_today is not None and earliest_today is not None
                and latest_today >= earliest_today):
            rate = (latest_today - earliest_today) / dt_h
        else:
            rate = max(0.0, (latest_1h - earliest_1h) / dt_h)

        if not math.isfinite(rate) or rate < 0.0 or rate > 250.0:
            return None
        return rate


def summarize_local_weather(stations, radius_km, max_age_min):
    lw = LocalWeather(radius_km=float(radius_km))
    fresh = [
        s for s in stations
        if s.dist_km <= radius_km
        and s.age_min is not None
        and s.age_min <= max_age_min
    ]
    lw.station_count = len(fresh)
    if not fresh:
        return lw

    nearest = min(fresh, key=lambda s: s.dist_km)
    lw.nearest_station = nearest.reading.name or nearest.reading.slug
    lw.nearest_dist_km = round(nearest.dist_km, 1)

    def _med(vals, digits=1):
        clean = [v for v in vals if v is not None and math.isfinite(v)]
        if not clean:
            return None
        m = float(np.median(clean))
        return round(m, digits) if digits is not None else m

    lw.temp_c = _med([s.reading.temp_c for s in fresh], 1)
    lw.temp_trend = _med([s.reading.temp_trend for s in fresh], 2)
    lw.humidity_pct = _med([s.reading.humidity_pct for s in fresh], 0)
    if lw.humidity_pct is not None:
        lw.humidity_pct = int(round(lw.humidity_pct))
    lw.pressure_hpa = _med([s.reading.pressure_hpa for s in fresh], 1)
    lw.pressure_trend = _med([s.reading.pressure_trend for s in fresh], 2)
    lw.wind_speed_kmh = _med([s.reading.wind_speed_kmh for s in fresh], 1)

    gusts = [s.reading.wind_gust_kmh for s in fresh if s.reading.wind_gust_kmh is not None]
    if gusts:
        lw.wind_gust_kmh = round(max(gusts), 1)

    sin_sum = 0.0
    cos_sum = 0.0
    dir_count = 0
    for s in fresh:
        deg = s.reading.wind_dir_deg
        if deg is None:
            continue
        spd = s.reading.wind_speed_kmh
        w = max(0.5, float(spd)) if spd is not None else 1.0
        rad = math.radians(deg)
        sin_sum += w * math.sin(rad)
        cos_sum += w * math.cos(rad)
        dir_count += 1
    if dir_count > 0 and math.hypot(sin_sum, cos_sum) > 1e-6:
        mean_deg = math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0
        deg_int = int(round(mean_deg)) % 360
        lw.wind_dir_deg = deg_int
        lw.wind_dir_cardinal = bearing_label(deg_int)

    lw.precip_today_mm = _med([s.reading.precip_today for s in fresh], 1)
    lw.precip_1h_mm = _med([s.reading.precip_1h for s in fresh], 1)
    lw.precip_6h_mm = _med([s.reading.precip_6h for s in fresh], 1)
    lw.precip_24h_mm = _med([s.reading.precip_24h for s in fresh], 1)
    lw.rain_rate_mmh = _med([s.rain_rate_mmh for s in fresh], 1)
    return lw


def _station_age_min(time_str, fetch_utc, display_tz):
    if not time_str or not fetch_utc:
        return None
    try:
        parts = time_str.split(":")
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    except (ValueError, IndexError):
        return None
    fetch_local = fetch_utc.astimezone(display_tz)
    dt = datetime(fetch_local.year, fetch_local.month, fetch_local.day,
                  h, m, s, tzinfo=display_tz)
    if dt > fetch_local + timedelta(minutes=2):
        dt -= timedelta(days=1)
    age = (fetch_local - dt).total_seconds() / 60.0
    return None if (age < 0 or age > 24 * 60) else age


# ----------------------------------------------------------------------------
# Combined analysis
# ----------------------------------------------------------------------------
@dataclass
class CombinedAnalysis:
    radar_raining: bool = False
    raining_now: bool = False
    confirmed: bool = False
    confidence: str = "dry"
    nearest_station: Optional[StationAnalysis] = None
    intensity_mmh: Optional[float] = None
    station_trend: str = "none"
    upstream_trend: str = "none"
    nearby_stations: list = field(default_factory=list)
    wet_stations: list = field(default_factory=list)
    station_count_wet: int = 0
    station_count_fresh: int = 0
    station_count_stale: int = 0
    wet_fraction: Optional[float] = None
    station_median_mm: Optional[float] = None


def combine_analysis(radar_raining, stations, tracks, approaches, calib,
                     t_lat, t_lon, confirm_radius, confirm_mm, confirm_age,
                     confirm_frac, confirm_min_count, rt):
    ca = CombinedAnalysis(radar_raining=radar_raining)
    fresh = [s for s in stations
             if s.dist_km <= confirm_radius
             and s.age_min is not None and s.age_min <= confirm_age]
    stale = [s for s in stations
             if s.dist_km <= confirm_radius
             and (s.age_min is None or s.age_min > confirm_age)]
    wet = [s for s in fresh if s.reading.precip_1h >= confirm_mm]
    ca.nearby_stations = fresh
    ca.wet_stations = wet
    ca.station_count_wet = len(wet)
    ca.station_count_fresh = len(fresh)
    ca.station_count_stale = len(stale)
    ca.wet_fraction = (len(wet) / len(fresh)) if fresh else None
    ca.nearest_station = min(wet, key=lambda s: s.dist_km) if wet else None
    rising_slopes = [s.slope_mmh for s in wet
                     if s.trend == "rising"
                     and s.slope_mmh is not None and s.slope_mmh > 0]
    if rising_slopes:
        ca.intensity_mmh = float(np.median(rising_slopes))
    elif wet:
        ca.intensity_mmh = 0.0
    if wet:
        ca.station_median_mm = float(np.median([s.reading.precip_1h for s in wet]))
    counts = {"rising": 0, "falling": 0, "stable": 0}
    for s in fresh:
        if s.trend in counts:
            counts[s.trend] += 1
    if any(counts.values()):
        ca.station_trend = max(counts, key=counts.get)
    else:
        ca.station_trend = "none"
    wet_frac = ca.wet_fraction or 0.0
    station_says_rain = (ca.station_count_fresh > 0
                         and ca.station_count_wet >= confirm_min_count
                         and wet_frac >= confirm_frac)
    ca.raining_now = station_says_rain or radar_raining
    if radar_raining and station_says_rain:
        ca.confirmed = True; ca.confidence = "confirmed"
    elif radar_raining:
        ca.confidence = "radar_only"
    elif station_says_rain:
        ca.confidence = "station_only"
    else:
        ca.confidence = "dry"
    mv = []
    for tr, a in zip(tracks, approaches):
        if tr.speed_kmh < MIN_SPEED_KMH:
            continue
        if not is_upwind(calib, tr.cluster, t_lat, t_lon, tr.vx_kmh, tr.vy_kmh):
            continue
        mv.append((tr.vx_kmh, tr.vy_kmh))
    if mv:
        vx, vy = max(mv, key=lambda v: math.hypot(*v))
        speed = math.hypot(vx, vy)
        ux, uy = vx / speed, vy / speed
        up = []
        for st in stations:
            if st.dist_km > 100:
                continue
            coslat = math.cos(math.radians(t_lat))
            dx = (st.reading.lon - t_lon) * 111.32 * coslat
            dy = (st.reading.lat - t_lat) * 111.32
            proj = dx * ux + dy * uy
            if -100 < proj < -5:
                up.append(st)
        counts2 = {"rising": 0, "falling": 0, "stable": 0}
        for st in up:
            if st.trend in counts2:
                counts2[st.trend] += 1
        if counts2["rising"] > max(counts2["falling"], counts2["stable"]):
            ca.upstream_trend = "rising"
        elif counts2["falling"] > max(counts2["rising"], counts2["stable"]):
            ca.upstream_trend = "falling"
        elif counts2["stable"] > 0 or counts2["rising"] > 0:
            ca.upstream_trend = "mixed"
        else:
            ca.upstream_trend = "unknown"
    return ca


# ----------------------------------------------------------------------------
# Satellite compositing + DHMZ SRI rendering
# ----------------------------------------------------------------------------
def _basemap_cache_path(data_dir, style, lat, lon, radius_km, view_rot, size_px):
    if not data_dir:
        return None
    safe_style = style.replace("/", "_")
    key = f"{safe_style}_{lat:.4f}_{lon:.4f}_{radius_km:.0f}_{view_rot:.1f}_{size_px}"
    return os.path.join(data_dir, "basemap_cache", key + ".png")


def fetch_basemap(style, token, lat, lon, radius_km, view_rot, size_px, data_dir):
    if not token or not style:
        return None
    cache_path = _basemap_cache_path(data_dir, style, lat, lon, radius_km, view_rot, size_px)
    if cache_path and os.path.exists(cache_path):
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            pass

    coslat = max(0.01, math.cos(math.radians(lat)))
    try:
        zoom = math.log2(size_px * 360.0 * 111.32 * coslat / (512.0 * 2 * radius_km))
    except ValueError:
        zoom = 8.0
    zoom = max(0.0, min(22.0, zoom))

    url = (f"https://api.mapbox.com/styles/v1/{style}/static/"
           f"{lon:.5f},{lat:.5f},{zoom:.2f},{view_rot:.1f},0/"
           f"{size_px}x{size_px}?access_token={token}")

    last_err = None
    for attempt in (1, 2):
        try:
            r = requests.get(url, timeout=(10, 30))
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            break
        except Exception as e:
            last_err = e
            if attempt == 1:
                time.sleep(2)
                continue
            print(f"basemap: fetch failed after {attempt} attempts: {last_err}",
                  file=sys.stderr)
            return None

    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            img.save(cache_path, format="PNG", optimize=True)
        except Exception as e:
            print(f"basemap: cache save failed: {e}", file=sys.stderr)
    return img


def _apply_basemap_color_control(pil_img, rt):
    sat = float(rt.get("basemap_saturation", 1.0))
    bri = float(rt.get("basemap_brightness", 1.0))
    con = float(rt.get("basemap_contrast", 1.0))
    if abs(sat - 1.0) > 1e-3:
        pil_img = ImageEnhance.Color(pil_img).enhance(sat)
    if abs(con - 1.0) > 1e-3:
        pil_img = ImageEnhance.Contrast(pil_img).enhance(con)
    if abs(bri - 1.0) > 1e-3:
        pil_img = ImageEnhance.Brightness(pil_img).enhance(bri)
    return pil_img


def _colorize_dhmz_sri(mask, mmh):
    H, W = mask.shape
    color = np.zeros((H, W, 3), dtype=np.uint8)
    if not mask.any():
        return color
    vals = np.clip(mmh, 0.2, 250.0)
    for lo, hi, rgb_col, _ in DHMZ_SRI_SCALE:
        sel = mask & (vals >= lo) & (vals < hi)
        if sel.any():
            color[sel] = rgb_col
    below = mask & (vals < DHMZ_SRI_SCALE[0][0])
    if below.any():
        color[below] = DHMZ_SRI_SCALE[0][2]
    return color


def _radar_to_grayscale(mask, mmh):
    H, W = mask.shape
    gray = np.zeros((H, W), dtype=np.uint8)
    if mask.any():
        clipped = np.clip(mmh, 0.1, 250.0)
        logmmh = np.log10(clipped)
        norm = np.clip((logmmh + 1.0) / 3.4, 0.0, 1.0)
        gray[mask] = (40.0 + norm[mask] * 180.0).astype(np.uint8)
    return gray


def _sample_radar_at_view(gray, radar_mask, radar_mmh, calib, target_lat, target_lon,
                          radius_km, view_rot, out_w, out_h):
    H, W = out_h, out_w
    xs = np.arange(W, dtype=np.float32) - W / 2.0
    ys = -(np.arange(H, dtype=np.float32) - H / 2.0)
    X, Y = np.meshgrid(xs, ys)
    kpp_x = 2.0 * radius_km / W
    kpp_y = 2.0 * radius_km / H
    dx_km = X * kpp_x
    dy_km = Y * kpp_y

    rot_rad = math.radians(view_rot)
    c, s = math.cos(rot_rad), math.sin(rot_rad)
    world_dx = dx_km * c + dy_km * s
    world_dy = -dx_km * s + dy_km * c

    lat = target_lat + world_dy / 111.32
    coslat = math.cos(math.radians(target_lat))
    lon = target_lon + world_dx / (111.32 * coslat)
    inv = calib._inv
    lat_c = calib.coef_lat[2]; lon_c = calib.coef_lon[2]
    radar_x = inv[0, 0] * (lat - lat_c) + inv[0, 1] * (lon - lon_c) + calib.map_region.x0
    radar_y = inv[1, 0] * (lat - lat_c) + inv[1, 1] * (lon - lon_c) + calib.map_region.y0
    rx = np.round(radar_x).astype(np.int32)
    ry = np.round(radar_y).astype(np.int32)
    valid = ((rx >= 0) & (rx < radar_mask.shape[1])
             & (ry >= 0) & (ry < radar_mask.shape[0]))
    gray_out = np.zeros((H, W), dtype=np.uint8)
    mmh_out = np.zeros((H, W), dtype=np.float32)
    mask_out = np.zeros((H, W), dtype=bool)
    gray_out[valid] = gray[ry[valid], rx[valid]]
    mmh_out[valid] = radar_mmh[ry[valid], rx[valid]]
    mask_out[valid] = radar_mask[ry[valid], rx[valid]]
    return gray_out, mask_out, mmh_out


def _latlon_to_overlay_px(calib, target_lat, target_lon, radius_km, size_px,
                           view_rot, lat, lon):
    coslat = math.cos(math.radians(target_lat))
    dx_km = (lon - target_lon) * 111.32 * coslat
    dy_km = (lat - target_lat) * 111.32

    rot_rad = math.radians(view_rot)
    c, s = math.cos(rot_rad), math.sin(rot_rad)
    rx_km = dx_km * c - dy_km * s
    ry_km = dx_km * s + dy_km * c

    kmpp = 2.0 * radius_km / size_px
    px = size_px / 2.0 + rx_km / kmpp
    py = size_px / 2.0 - ry_km / kmpp
    return px, py


def _draw_filled_circle(rgb, cx, cy, radius, color, alpha=0.55):
    r = int(radius)
    if r <= 0: return
    cx, cy = int(round(cx)), int(round(cy))
    H, W = rgb.shape[:2]
    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    y0, y1 = max(0, cy - r), min(H, cy + r + 1)
    if x0 >= x1 or y0 >= y1: return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    if not m.any(): return
    if alpha >= 1.0:
        rgb[y0:y1, x0:x1][m] = color
    else:
        bg = rgb[y0:y1, x0:x1][m].astype(np.float32)
        fg = np.array(color, dtype=np.float32)
        rgb[y0:y1, x0:x1][m] = (bg * (1 - alpha) + fg * alpha).astype(np.uint8)


def _draw_circle_outline(rgb, cx, cy, radius, color, thickness=1):
    r = int(radius)
    if r <= 0: return
    cx, cy = int(round(cx)), int(round(cy))
    H, W = rgb.shape[:2]
    n = max(48, int(2 * math.pi * r))
    for k in range(n):
        ang = 2 * math.pi * k / n
        for d in range(thickness):
            px = int(cx + (r - d) * math.cos(ang))
            py = int(cy + (r - d) * math.sin(ang))
            if 0 <= px < W and 0 <= py < H:
                rgb[py, px] = color


def _draw_coast_overlay(rgb, calib, target_lat, target_lon, radius_km, view_rot, size_px):
    coast_color = (200, 200, 200)
    for (la1, lo1), (la2, lo2) in zip(ADRIATIC_COAST, ADRIATIC_COAST[1:]):
        steps = max(2, int(max(abs(la2 - la1), abs(lo2 - lo1)) / 0.03))
        prev_px = prev_py = None
        for k in range(steps + 1):
            t = k / steps
            la = la1 + (la2 - la1) * t
            lo = lo1 + (lo2 - lo1) * t
            px, py = _latlon_to_overlay_px(calib, target_lat, target_lon,
                                           radius_km, size_px, view_rot, la, lo)
            if 0 <= px < size_px and 0 <= py < size_px:
                if prev_px is not None:
                    dx = px - prev_px; dy = py - prev_py
                    n_line = int(max(1, math.hypot(dx, dy)))
                    for s in range(n_line + 1):
                        fp = s / n_line
                        lx = int(round(prev_px + dx * fp))
                        ly = int(round(prev_py + dy * fp))
                        if 0 <= lx < size_px and 0 <= ly < size_px:
                            rgb[ly, lx] = coast_color
                prev_px, prev_py = px, py
            else:
                prev_px = prev_py = None


def _draw_text_outlined(draw, x, y, text, fill, outline):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0: continue
            draw.text((x + dx, y + dy), text, fill=outline)
    draw.text((x, y), text, fill=fill)


def _draw_arrow(draw, x0, y0, x1, y1, color, outline=(0, 0, 0)):
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length < 1.5:
        draw.ellipse([x0 - 2.5, y0 - 2.5, x0 + 2.5, y0 + 2.5],
                     fill=color, outline=outline)
        return

    ux, uy = dx / length, dy / length
    px, py = -uy, ux

    head_len = min(11.0, max(5.0, length * 0.35))
    head_w = head_len * 0.55

    bx = x1 - ux * head_len
    by = y1 - uy * head_len

    draw.line([(x0, y0), (bx, by)], fill=outline, width=4)
    tip_out = (x1 + ux * 1.5, y1 + uy * 1.5)
    left_out = (bx - ux * 1.0 + px * (head_w + 1.5), by - uy * 1.0 + py * (head_w + 1.5))
    right_out = (bx - ux * 1.0 - px * (head_w + 1.5), by - uy * 1.0 - py * (head_w + 1.5))
    draw.polygon([tip_out, left_out, right_out], fill=outline)

    draw.line([(x0, y0), (bx, by)], fill=color, width=2)
    tip = (x1, y1)
    left = (bx + px * head_w, by + py * head_w)
    right = (bx - px * head_w, by - py * head_w)
    draw.polygon([tip, left, right], fill=color)
    draw.ellipse([x0 - 2.5, y0 - 2.5, x0 + 2.5, y0 + 2.5],
                 fill=color, outline=outline)


def _draw_compass(img, cx, cy, view_rot, size=24):
    draw = ImageDraw.Draw(img)
    draw.ellipse([cx - size, cy - size, cx + size, cy + size],
                 outline=(255, 255, 255), width=2)
    angle_rad = math.radians(view_rot)
    nx = cx - size * 0.75 * math.sin(angle_rad)
    ny = cy - size * 0.75 * math.cos(angle_rad)
    draw.line([cx, cy, nx, ny], fill=(255, 255, 255), width=2)
    draw.ellipse([nx - 2, ny - 2, nx + 2, ny + 2], fill=(255, 255, 255))
    _draw_text_outlined(draw, nx + 3, ny - 5, "N", (255, 255, 255), (0, 0, 0))


def _draw_scale_bar(img, cx, cy, radius_km, size_px, view_rot):
    draw = ImageDraw.Draw(img)
    kmpp = 2 * radius_km / size_px
    bar_km = 50 if radius_km >= 200 else 25 if radius_km >= 100 else 10
    bar_px = bar_km / kmpp
    x0 = cx
    y0 = cy
    draw.line([x0, y0, x0 + bar_px, y0], fill=(255, 255, 255), width=2)
    draw.line([x0, y0 - 4, x0, y0 + 4], fill=(255, 255, 255), width=2)
    draw.line([x0 + bar_px, y0 - 4, x0 + bar_px, y0 + 4], fill=(255, 255, 255), width=2)
    _draw_text_outlined(draw, x0 + bar_px + 4, y0 - 6, f"{bar_km} km",
                        (255, 255, 255), (0, 0, 0))


def _draw_legend(pil_img, confirm_mm):
    W, H = pil_img.size
    box_w, box_h = 228, 74
    x0 = W - box_w - 10
    y0 = H - box_h - 10
    if x0 < 0 or y0 < 0:
        return

    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rounded_rectangle(
        [x0, y0, x0 + box_w, y0 + box_h],
        radius=5,
        fill=(12, 16, 22, 185),
        outline=(110, 120, 135, 200),
        width=1,
    )
    pil_rgba = pil_img.convert("RGBA")
    pil_rgba = Image.alpha_composite(pil_rgba, overlay)
    pil_img.paste(pil_rgba.convert("RGB"))

    draw = ImageDraw.Draw(pil_img)

    r1_y = y0 + 11
    draw.ellipse([x0 + 7, r1_y - 4, x0 + 15, r1_y + 4],
                 fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((x0 + 18, r1_y - 6), "Dry", fill=(235, 235, 235))

    rx = x0 + 46
    draw.ellipse([rx, r1_y - 4, rx + 8, r1_y + 4],
                 fill=(235, 35, 35), outline=(0, 0, 0))
    draw.text((rx + 11, r1_y - 6), f"≥{confirm_mm:g}mm", fill=(255, 190, 190))

    ax = x0 + 106
    _draw_arrow(draw, ax, r1_y, ax + 18, r1_y, (110, 235, 255))
    draw.text((ax + 22, r1_y - 6), "1h track", fill=(195, 240, 255))

    draw.text((x0 + 8, y0 + 25), "DHMZ SRI (mm/h):", fill=(200, 215, 230))
    bar_x0 = x0 + 8
    bar_y0 = y0 + 40
    bar_w = box_w - 16
    bar_h = 10
    n_bins = len(DHMZ_SRI_SCALE)
    step_w = bar_w / n_bins
    for i, (_, _, col, _) in enumerate(DHMZ_SRI_SCALE):
        bx0 = int(round(bar_x0 + i * step_w))
        bx1 = int(round(bar_x0 + (i + 1) * step_w))
        draw.rectangle([bx0, bar_y0, bx1, bar_y0 + bar_h], fill=col)
    draw.rectangle([bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h],
                   outline=(160, 170, 180))

    ticks = [(0, "0.2"), (2, "1"), (4, "3"), (7, "10"), (10, "30"), (13, "70+")]
    for idx, lbl in ticks:
        tx = int(round(bar_x0 + idx * step_w))
        draw.text((tx, bar_y0 + bar_h + 2), lbl, fill=(220, 220, 220))


def compose_map(radar_mask, radar_mmh, gray, calib, target_lat, target_lon,
                radius_km, view_rot, size_px, basemap_img, basemap_kind,
                tracks, approaches, approaching_flags, threat_flags,
                stations, show_cities, intensity_mode, sri_thresholds, rt,
                confirm_mm=0.2):
    gray_out, mask_out, mmh_out = _sample_radar_at_view(
        gray, radar_mask, radar_mmh, calib, target_lat, target_lon,
        radius_km, view_rot, size_px, size_px)

    if basemap_img is not None:
        try:
            base_pil = basemap_img.convert("RGB")
            if base_pil.size != (size_px, size_px):
                base_pil = base_pil.resize((size_px, size_px), Image.LANCZOS)
            base_pil = _apply_basemap_color_control(base_pil, rt)
            rgb = np.array(base_pil, dtype=np.uint8).copy()
        except Exception:
            rgb = np.stack([gray_out, gray_out, gray_out], axis=-1)
            basemap_kind = "greyscale"
    else:
        rgb = np.stack([gray_out, gray_out, gray_out], axis=-1)

    if mask_out.any():
        color = _colorize_dhmz_sri(mask_out, mmh_out)
        if basemap_kind == "mapbox":
            alpha = 0.72
            rgb[mask_out] = (rgb[mask_out].astype(np.float32) * (1 - alpha)
                             + color[mask_out].astype(np.float32) * alpha).astype(np.uint8)
        else:
            rgb[mask_out] = color[mask_out]

    if basemap_kind != "mapbox":
        _draw_coast_overlay(rgb, calib, target_lat, target_lon,
                            radius_km, view_rot, size_px)

    for st in stations:
        px, py = _latlon_to_overlay_px(calib, target_lat, target_lon,
                                       radius_km, size_px, view_rot,
                                       st.reading.lat, st.reading.lon)
        if 0 <= px < size_px and 0 <= py < size_px:
            is_raining = st.reading.precip_1h >= confirm_mm
            if is_raining:
                color = (235, 35, 35)
                r = 5
            else:
                color = (255, 255, 255)
                r = 4
            _draw_filled_circle(rgb, px, py, r, color, alpha=0.95)
            _draw_circle_outline(rgb, px, py, r, (0, 0, 0), thickness=1)

    for i, (tr, a, tf) in enumerate(zip(tracks, approaches, threat_flags)):
        if not approaching_flags[i]:
            continue
        clat, clon = calib.image_pixel_to_latlon(tr.cluster.cx, tr.cluster.cy)
        px, py = _latlon_to_overlay_px(calib, target_lat, target_lon,
                                       radius_km, size_px, view_rot, clat, clon)
        if 0 <= px < size_px and 0 <= py < size_px:
            pix_r = max(8.0, a.cluster_radius_km / (2 * radius_km / size_px))
            color = (220, 60, 60) if tf else (230, 140, 40)
            _draw_circle_outline(rgb, px, py, pix_r, color, thickness=2)

    pil = Image.fromarray(rgb.astype(np.uint8))
    draw = ImageDraw.Draw(pil)

    for i, (tr, a, af, tf) in enumerate(zip(tracks, approaches, approaching_flags, threat_flags)):
        if tr.speed_kmh < MIN_SPEED_KMH:
            continue
        clat, clon = calib.image_pixel_to_latlon(tr.cluster.cx, tr.cluster.cy)
        px0, py0 = _latlon_to_overlay_px(calib, target_lat, target_lon,
                                         radius_km, size_px, view_rot, clat, clon)
        coslat = max(0.01, math.cos(math.radians(clat)))
        lat_1h = clat + tr.vy_kmh / 111.32
        lon_1h = clon + tr.vx_kmh / (111.32 * coslat)
        px1, py1 = _latlon_to_overlay_px(calib, target_lat, target_lon,
                                         radius_km, size_px, view_rot, lat_1h, lon_1h)
        if ((0 <= px0 < size_px and 0 <= py0 < size_px)
                or (0 <= px1 < size_px and 0 <= py1 < size_px)):
            if tf:
                arrow_color = (245, 55, 55)
            elif af:
                arrow_color = (255, 175, 40)
            else:
                arrow_color = (110, 235, 255)
            _draw_arrow(draw, px0, py0, px1, py1, arrow_color)

    if show_cities:
        for label, name, clat, clon in CITIES:
            px, py = _latlon_to_overlay_px(calib, target_lat, target_lon,
                                           radius_km, size_px, view_rot, clat, clon)
            if 0 <= px < size_px and 0 <= py < size_px:
                _draw_text_outlined(draw, px + 3, py - 6, label,
                                    (255, 255, 255), (0, 0, 0))

    _draw_compass(pil, size_px - 34, 34, view_rot)
    _draw_scale_bar(pil, 14, size_px - 18, radius_km, size_px, view_rot)
    _draw_legend(pil, confirm_mm)
    return pil


def render_composed_png(img, calib, target_lat, target_lon, radius_km,
                        view_rot, size_px, basemap_img, basemap_kind,
                        tracks, approaches, approaching_flags, threat_flags,
                        stations, show_cities,
                        intensity_mode, sri_legend, legend_mask,
                        sri_thresholds, rt, confirm_mm=0.2, header_text=None):
    try:
        img.seek(max(0, getattr(img, "n_frames", 1) - 1))
    except Exception:
        pass
    frame = img.convert("RGB")
    mask, mmh = analyze_frame(frame, calib.map_region, intensity_mode,
                              sri_legend, legend_mask)
    gray = _radar_to_grayscale(mask, mmh)
    pil = compose_map(mask, mmh, gray, calib, target_lat, target_lon,
                      radius_km, view_rot, size_px, basemap_img, basemap_kind,
                      tracks, approaches, approaching_flags, threat_flags,
                      stations, show_cities,
                      intensity_mode, sri_thresholds, rt, confirm_mm=confirm_mm)
    if header_text:
        draw = ImageDraw.Draw(pil)
        _draw_text_outlined(draw, 10, 8, header_text, (255, 220, 60), (0, 0, 0))
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_composed_gif(img, calib, target_lat, target_lon, radius_km,
                        view_rot, size_px, basemap_img, basemap_kind,
                        tracks, tracks_by_id, split_frames_by_id,
                        frame_interval_min, approaches, approaching_flags,
                        threat_flags, stations, show_cities,
                        intensity_mode, sri_legend, legend_mask,
                        sri_thresholds, rt, confirm_mm=0.2,
                        frame_duration_ms=400, max_frames=30):
    nf = getattr(img, "n_frames", 1)
    start = max(0, nf - max_frames)
    split_frames_by_id = split_frames_by_id or {}
    out = []
    try:
        for fidx in range(start, nf):
            try:
                img.seek(fidx)
            except Exception:
                continue
            frame = img.convert("RGB")
            mask, mmh = analyze_frame(frame, calib.map_region, intensity_mode,
                                      sri_legend, legend_mask)
            gray = _radar_to_grayscale(mask, mmh)

            frame_tracks = []
            frame_approaches = []
            frame_appr_flags = []
            frame_threat_flags = []
            min_obs_needed = min(rt["min_duration_frames"], fidx + 1)

            for cid, obs in tracks_by_id.items():
                obs_up_to = [o for o in obs if o[0] <= fidx]
                if not obs_up_to or obs_up_to[-1][0] != fidx:
                    continue
                cluster_here = obs_up_to[-1][1]
                if cluster_here.size < rt["min_size_px"]:
                    continue
                if len(obs_up_to) < min_obs_needed:
                    continue
                vx_kmh, vy_kmh = _estimate_velocity_kmh(
                    obs_up_to, calib, frame_interval_min, split_frames_by_id.get(cid))
                tr_copy = Track(
                    cluster=PixelCluster(
                        cluster_here.cx, cluster_here.cy,
                        cluster_here.size, cluster_here.label,
                        cluster_here.max_mmh, cluster_here.mean_mmh),
                    cid=cid,
                    vx_kmh=vx_kmh,
                    vy_kmh=vy_kmh,
                    n_observations=len(obs_up_to),
                    span_frames=obs_up_to[-1][0] - obs_up_to[0][0])
                a_here = assess(tr_copy, target_lat, target_lon, calib)
                frame_tracks.append(tr_copy)
                frame_approaches.append(a_here)
                frame_appr_flags.append(refine_approaching(tr_copy, a_here, rt))
                frame_threat_flags.append(refine_threat(tr_copy, a_here, rt))

            pil = compose_map(mask, mmh, gray, calib, target_lat, target_lon,
                              radius_km, view_rot, size_px, basemap_img, basemap_kind,
                              frame_tracks, frame_approaches,
                              frame_appr_flags, frame_threat_flags,
                              stations, show_cities,
                              intensity_mode, sri_thresholds, rt,
                              confirm_mm=confirm_mm)
            out.append(pil)
    finally:
        try: img.seek(0)
        except Exception: pass
    if not out:
        return None
    buf = io.BytesIO()
    out[0].save(buf, format="GIF", save_all=True, append_images=out[1:],
                duration=frame_duration_ms, loop=0, optimize=True)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# HTTP server
# ----------------------------------------------------------------------------
_INDEX_HTML = """<!DOCTYPE html><html><head><title>DHMZ Radar Tracker</title>
<style>
body{background:#111;color:#ddd;font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:0;padding:20px}
h1{color:#fff;margin-top:0}.layout{display:flex;gap:20px;flex-wrap:wrap}
.left{flex:1 1 450px;max-width:560px}.right{flex:2 1 600px}
.section{background:#1a1a1a;border:1px solid #333;border-radius:4px;padding:12px;margin-bottom:12px}
.section h3{margin:0 0 8px 0;font-size:13px;text-transform:uppercase;color:#5cf;letter-spacing:0.5px}
.section h4{margin:8px 0 4px 0;font-size:12px;color:#5cf}
.field{display:flex;align-items:center;margin-bottom:6px;font-size:13px}
.field label{flex:1;color:#ccc}.field input[type=number]{width:90px;background:#0a0a0a;color:#ddd;border:1px solid #333;padding:4px;border-radius:3px}
.field input[type=text]{flex:0 0 160px;background:#0a0a0a;color:#ddd;border:1px solid #333;padding:4px;border-radius:3px}
.field select{width:110px;background:#0a0a0a;color:#ddd;border:1px solid #333;padding:4px;border-radius:3px}
.field input[type=checkbox]{accent-color:#5cf;width:18px;height:18px}
.actions{margin-top:8px;display:flex;gap:8px;flex-wrap:wrap}
button{background:#2a5c8a;color:#fff;border:0;padding:8px 14px;border-radius:3px;cursor:pointer;font-size:13px}
button:hover{background:#3a7cba}button.secondary{background:#444}
button.secondary:hover{background:#555}button.danger{background:#8a2a2a}
button.danger:hover{background:#ba3a3a}
.status{background:#1a1a1a;border:1px solid #333;border-radius:4px;padding:12px;margin-bottom:12px;font-size:13px}
.status .row{display:flex;margin-bottom:4px}.status .label{color:#888;flex:0 0 160px}
.wx-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.wx-box{background:#121519;border:1px solid #282c34;border-radius:4px;padding:7px 8px}
.wx-k{font-size:11px;color:#888;text-transform:uppercase}
.wx-v{font-size:15px;font-weight:600;color:#fff;margin-top:2px;font-variant-numeric:tabular-nums}
.wx-sub{font-size:11px;color:#9aa}
.threat-item{color:#f66;font-weight:bold}
table.acc{width:100%;border-collapse:collapse;font-size:12px}
table.acc th,table.acc td{padding:4px 6px;text-align:left;border-bottom:1px solid #222}
table.acc th{color:#5cf;font-weight:normal;text-transform:uppercase;font-size:11px}
table.acc td.num{text-align:right;font-variant-numeric:tabular-nums}
img{max-width:100%;border:1px solid #333;display:block;margin-bottom:12px}
a{color:#5cf}.hint{color:#666;font-size:11px;margin-top:4px}
.cfgpath{font-family:monospace;color:#7c7}
</style></head><body>
<h1>DHMZ Radar Tracker</h1>
<p><a href="/threat.gif">/threat.gif</a> · <a href="/threat.png">/threat.png</a> · <a href="/status">/status</a> · <a href="/settings">/settings</a> · <a href="/config">/config</a> · <a href="/archive">/archive</a> · <a href="/accuracy">/accuracy</a></p>
<div class="layout">
<div class="left">
<div class="status" id="status">Loading...</div>
<div class="section"><h3>Local Weather (Pljusak)</h3>
<div id="local_wx">Loading...</div>
</div>
<div class="section"><h3>Storm Frame Archive</h3>
<div id="archive_box">Loading...</div>
</div>
<div class="section"><h3>Accuracy</h3>
<div id="accuracy">Loading...</div>
<div class="actions" style="margin-top:8px">
<button class="secondary" onclick="replayAccuracy()">Replay scores</button>
<button class="danger" onclick="resetAccuracy()">Reset scores</button>
</div></div>
<div class="section"><h3>Sensors (HA)</h3>
<div id="sensors">Loading...</div>
</div>
<div class="section"><h3>Runtime settings (hot)</h3>
<div class="field"><label>Base approach radius (km)</label><input type="number" id="approaching_radius_km" min="1" max="500" step="5"></div>
<div class="field"><label>Cluster size weight</label><input type="number" id="cluster_radius_weight" min="0" max="5" step="0.1"></div>
<div class="field"><label>Hard angle cap (°)</label><input type="number" id="only_approach_deg" min="0" max="180" step="5"></div>
<div class="field"><label>Threat radius (km)</label><input type="number" id="threat_radius_km" min="1" max="500" step="5"></div>
<div class="field"><label>Threat angle (°)</label><input type="number" id="threat_angle_deg" min="0" max="180" step="5"></div>
<div class="field"><label>Threat min duration (fr)</label><input type="number" id="threat_min_duration" min="1" max="50" step="1"></div>
<div class="field"><label>Threat min SRI (mm/h)</label><input type="number" id="threat_min_sri_mmh" min="0" max="500" step="0.5"></div>
<div class="field"><label>Map radius (km)</label><input type="number" id="radius_km" min="20" max="1000" step="10"></div>
<div class="field"><label>Local weather radius (km)</label><input type="number" id="weather_radius_km" min="0.5" max="100" step="0.5"></div>
<div class="field"><label>Archive storm frames</label><input type="checkbox" id="archive_enabled"></div>
<div class="field"><label>Archive radius (km)</label><input type="number" id="archive_radius_km" min="20" max="500" step="10"></div>
<div class="field"><label>Archive min precip (px)</label><input type="number" id="archive_min_pixels" min="10" max="5000" step="10"></div>
<div class="field"><label>Min cell size (px)</label><input type="number" id="min_size_px" min="5" max="5000" step="5"></div>
<div class="field"><label>Min duration (fr)</label><input type="number" id="min_duration_frames" min="1" max="50" step="1"></div>
<div class="field"><label>View rotation: auto (coast)</label><input type="checkbox" id="view_rotation_auto"></div>
<div class="field"><label>View rotation (° CW, up)</label><input type="number" id="view_rotation" min="0" max="360" step="5"></div>
<div class="field"><label>Basemap</label>
<select id="basemap">
<option value="auto">auto</option>
<option value="mapbox">mapbox</option>
<option value="greyscale">greyscale</option>
<option value="none">none</option>
</select></div>
<div class="field"><label>Basemap saturation (0=grey)</label><input type="number" id="basemap_saturation" min="0" max="2" step="0.1"></div>
<div class="field"><label>Basemap brightness</label><input type="number" id="basemap_brightness" min="0.2" max="2" step="0.05"></div>
<div class="field"><label>Basemap contrast</label><input type="number" id="basemap_contrast" min="0.2" max="2" step="0.05"></div>
<div class="field"><label>Cities</label><input type="checkbox" id="show_cities"></div>
<div class="field"><label>Poll interval (s)</label><input type="number" id="poll_s" min="30" max="7200" step="30"></div>
<div class="field"><label>Image scale</label><input type="number" id="web_scale" min="0.1" max="4" step="0.1"></div>
<div class="field"><label>Settings tag</label><input type="text" id="settings_tag" placeholder="baseline, tuned-v1"></div>
<div class="hint">Saved to config.json. Applies next cycle.</div>
<div class="actions">
<button onclick="saveAll(false)">Save</button>
<button onclick="saveAll(true)">Save & refresh</button>
</div></div>
</div>
<div class="right">
<img id="g" src="/threat.gif">
<img id="p" src="/threat.png">
</div>
</div>
<script>
const RT_FIELDS=['approaching_radius_km','cluster_radius_weight','only_approach_deg','threat_radius_km','threat_angle_deg','threat_min_duration','threat_min_sri_mmh','radius_km','weather_radius_km','archive_enabled','archive_radius_km','archive_min_pixels','min_size_px','min_duration_frames','view_rotation_auto','view_rotation','basemap','basemap_saturation','basemap_brightness','basemap_contrast','show_cities','poll_s','web_scale','settings_tag'];
const MAIN_SENSORS=['raining_now','raining_confirmed','rain_confidence','rain_duration_min','rain_end_eta_min','next_rain_eta_min','gap_min','summary','radar_intensity_mmh','nearest_station_mm','intensity_mmh','station_trend','upstream_trend','local_temp_c','local_humidity','local_pressure_hpa','local_wind_speed_kmh','local_wind_gust_kmh','local_wind_dir_deg','local_wind_dir','local_rain_today_mm','local_rain_1h_mm','local_rain_rate_mmh'];
const DIAG_SENSORS=['coverage_map','nowcast_error_min','data_age_min','frame_timestamp','cell_count','nearest_cell_km','nearest_cell_eta_min','nearest_station_km','nearest_station_age_min','station_count_wet','station_count_fresh','station_wet_fraction','station_median_mm','local_weather_stations'];
function setField(k,v){const el=document.getElementById(k);if(!el)return;if(el.type==='checkbox')el.checked=!!v;else el.value=v;}
function getField(k){const el=document.getElementById(k);if(!el)return null;if(el.type==='checkbox')return el.checked;if(k==='settings_tag'||k==='basemap')return el.value;return parseFloat(el.value);}
function loadSettings(){fetch('/settings').then(r=>r.json()).then(s=>{RT_FIELDS.forEach(k=>{if(k in s)setField(k,s[k]);});}).catch(e=>{});}
function saveAll(refresh){const body={};RT_FIELDS.forEach(k=>body[k]=getField(k));
 fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
 .then(r=>r.json()).then(d=>{if(d.ok&&refresh)return fetch('/refresh',{method:'POST'});})
 .then(()=>{bumpImages();loadStatus();loadAccuracy();});}
function fmtVal(v){if(v===null||v===undefined)return '—';if(typeof v==='boolean')return v?'yes':'no';return String(v);}
function loadStatus(){fetch('/status').then(r=>r.json()).then(s=>{const el=document.getElementById('status');if(!s||!s.updated){el.textContent='No data yet — waiting for first cycle';return;}
 const threatsHtml=(s.threats||[]).map(t=>`<div class="threat-item">• ${t.label} ${t.size_px}px, ${t.max_mmh!=null?t.max_mmh+'mm/h, ':''}miss ${t.miss_km}km, ETA ${t.eta_min!=null?t.eta_min+'min':'?'}</div>`).join('');
 el.innerHTML=`<div class="row"><span class="label">Target</span><span>${s.target||'?'}</span></div>
 <div class="row"><span class="label">Updated</span><span>${s.updated}</span></div>
 <div class="row"><span class="label">Confidence</span><span>${s.confidence||'?'}</span></div>
 <div class="row"><span class="label">Raining</span><span>${s.raining_now?'yes':'no'}</span></div>
 <div class="row"><span class="label">Approaching</span><span>${s.approaching_count||0}</span></div>
 <div class="row"><span class="label">Threats</span><span>${s.threat_count||0}</span></div>
 <div class="row"><span class="label">View rotation</span><span>${s.view_rotation_used!=null?s.view_rotation_used.toFixed(0)+'°'+(s.view_rotation_auto?' (auto)':'') :'—'}</span></div>
 <div class="row"><span class="label">Basemap</span><span>${s.basemap_used||'—'}</span></div>
 <div class="row"><span class="label">Config</span><span class="cfgpath">${s.config_path||'(not persisted)'}</span></div>
 ${threatsHtml}`;
 const arch=s.archive||{};
 const archEl=document.getElementById('archive_box');
 if(!arch.archive_dir){
   archEl.innerHTML='<div class="hint">Persistence disabled (no data-dir)</div>';
 }else{
   archEl.innerHTML=`<div class="hint">Status: ${arch.enabled?'Active':'Paused'} (radius ≤ ${arch.radius_km||200} km) · Episodes: <b>${arch.total_episodes||0}</b> · Frames: <b>${arch.total_frames||0}</b></div>
   <div class="hint">Last episode: <span class="cfgpath">${arch.last_episode||'none'}</span>${arch.last_saved_utc?' · '+arch.last_saved_utc:''}</div>`;
 }
 const wx=s.local_weather||{};
 const wxEl=document.getElementById('local_wx');
 if(!wx.station_count){
   wxEl.innerHTML=`<div class="hint">No fresh Pljusak stations within ${wx.radius_km||5} km</div>`;
 }else{
   const windTxt=wx.wind_speed_kmh!=null?`${wx.wind_speed_kmh} km/h`:'—';
   const gustTxt=wx.wind_gust_kmh!=null?`gust ${wx.wind_gust_kmh}`:'';
   const dirTxt=wx.wind_dir_deg!=null?`${wx.wind_dir_cardinal} (${wx.wind_dir_deg}°)`:'—';
   const pTrend=wx.pressure_trend!=null?(wx.pressure_trend>0?'+':'')+wx.pressure_trend+' hPa':'';
   const tTrend=wx.temp_trend!=null?(wx.temp_trend>0?'+':'')+wx.temp_trend+'°/h':'';
   wxEl.innerHTML=`<div class="wx-grid">
     <div class="wx-box"><div class="wx-k">Temp</div><div class="wx-v">${wx.temp_c!=null?wx.temp_c+' °C':'—'}</div><div class="wx-sub">${tTrend||' '}</div></div>
     <div class="wx-box"><div class="wx-k">Humidity</div><div class="wx-v">${wx.humidity_pct!=null?wx.humidity_pct+' %':'—'}</div><div class="wx-sub"> </div></div>
     <div class="wx-box"><div class="wx-k">Pressure</div><div class="wx-v">${wx.pressure_hpa!=null?wx.pressure_hpa+' hPa':'—'}</div><div class="wx-sub">${pTrend||' '}</div></div>
     <div class="wx-box"><div class="wx-k">Wind</div><div class="wx-v">${windTxt}</div><div class="wx-sub">${gustTxt||' '}</div></div>
     <div class="wx-box"><div class="wx-k">Heading</div><div class="wx-v">${dirTxt}</div><div class="wx-sub"> </div></div>
     <div class="wx-box"><div class="wx-k">Rain (Today / 1h)</div><div class="wx-v">${wx.precip_today_mm!=null?wx.precip_today_mm:'—'} / ${wx.precip_1h_mm!=null?wx.precip_1h_mm:'—'} mm</div><div class="wx-sub">${wx.rain_rate_mmh!=null?wx.rain_rate_mmh+' mm/h':' '}</div></div>
   </div>
   <div class="hint" style="margin-top:6px">${wx.station_count} fresh station(s) ≤ ${wx.radius_km} km · nearest: ${wx.nearest_station||'?'} (${wx.nearest_dist_km} km)</div>`;
 }
 const sensors=s.sensors||{};
 let html='<table class="acc"><tr><th>Sensor</th><th>Value</th></tr>';
 for(const k of MAIN_SENSORS){html+=`<tr><td>${k}</td><td class="num">${fmtVal(sensors[k])}</td></tr>`;}
 html+='</table><h4>Diagnostic</h4><table class="acc">';
 for(const k of DIAG_SENSORS){html+=`<tr><td>${k}</td><td class="num">${fmtVal(sensors[k])}</td></tr>`;}
 html+='</table>';
 document.getElementById('sensors').innerHTML=html;
}).catch(e=>{});}
function loadAccuracy(){fetch('/accuracy').then(r=>r.json()).then(a=>{const el=document.getElementById('accuracy');
 if(!a){el.innerHTML='<div class="hint">No accuracy data</div>';return;}
 let html='<div class="hint">Preds '+a.total_predictions+' · Outcomes '+a.total_outcomes+' · Scores '+a.total_scores+' · Superseded '+(a.superseded_predictions||0)+'</div>';
 const pend=a.pending_by_kind||{};
 if(Object.keys(pend).length){html+='<div class="hint">Pending: '+Object.entries(pend).map(([k,v])=>k+':'+v).join(', ')+'</div>';}
 if(a.total_scores>0){
   const order=['cluster_arrive','rain_start','rain_end','threat'];
   html+='<table class="acc"><tr><th>Kind</th><th>N</th><th>Bias</th><th>MAE</th><th>p90</th></tr>';
   for(const k of order){const v=a.by_kind[k];if(!v||!v.n)continue;
     const bias=v.median_error_min!=null?(v.median_error_min>0?'+':'')+v.median_error_min.toFixed(0):'—';
     html+=`<tr><td>${k}</td><td class="num">${v.n}</td><td class="num">${bias}</td><td class="num">${v.mean_abs_error_min.toFixed(1)}</td><td class="num">${v.p90_abs_error_min.toFixed(1)}</td></tr>`;}
   html+='</table>';
   const tags=Object.keys(a.by_tag||{});
   if(tags.length){html+='<h4>By tag</h4><table class="acc"><tr><th>Tag</th><th>N</th><th>Bias</th><th>MAE</th></tr>';
     for(const t of tags){const v=a.by_tag[t];if(!v||!v.n)continue;
       const bias=v.median_error_min!=null?(v.median_error_min>0?'+':'')+v.median_error_min.toFixed(0):'—';
       html+=`<tr><td>${t||'(untagged)'}</td><td class="num">${v.n}</td><td class="num">${bias}</td><td class="num">${v.mean_abs_error_min.toFixed(1)}</td></tr>`;}
     html+='</table>';}
 }
 el.innerHTML=html;
}).catch(e=>{});}
function replayAccuracy(){if(!confirm('Re-score all predictions?'))return;fetch('/accuracy/replay',{method:'POST'}).then(()=>loadAccuracy());}
function resetAccuracy(){if(!confirm('Clear all scores?'))return;fetch('/accuracy/reset',{method:'POST'}).then(()=>loadAccuracy());}
function bumpImages(){const t=Date.now();document.getElementById('g').src='/threat.gif?t='+t;document.getElementById('p').src='/threat.png?t='+t;}
loadSettings();loadStatus();loadAccuracy();
setInterval(loadStatus,10000);setInterval(loadAccuracy,30000);setInterval(bumpImages,60000);
</script></body></html>"""


class _VizHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        qs = self.path.split("?", 1)[1] if "?" in self.path else ""
        params = {}
        for part in qs.split("&"):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k] = v
        if path == "/":
            self._html()
        elif path == "/threat.png":
            self._bytes("png", "image/png")
        elif path == "/threat.gif":
            self._bytes("gif", "image/gif")
        elif path == "/status":
            self._status()
        elif path == "/settings":
            self._json(_snapshot())
        elif path == "/config":
            self._config_get()
        elif path == "/archive":
            self._archive(params)
        elif path == "/accuracy":
            self._accuracy(params)
        elif path == "/accuracy/predictions":
            self._accuracy_predictions(params)
        elif path == "/accuracy/events":
            self._accuracy_events(params)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/settings":
            self._settings_post()
        elif path == "/config":
            self._config_post()
        elif path == "/refresh":
            _FORCE_REFRESH.set()
            self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()
        elif path == "/accuracy/reset":
            acc = _VIZ_STATE.get("acc_store")
            if acc: acc.reset_scores()
            self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()
        elif path == "/accuracy/replay":
            acc = _VIZ_STATE.get("acc_store")
            if acc: acc.replay()
            self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()
        else:
            self.send_response(404); self.end_headers()

    def do_DELETE(self):
        if self.path.split("?")[0] == "/config":
            cfg_path = _VIZ_STATE.get("config_path")
            if cfg_path:
                for p in (cfg_path, cfg_path + ".bak"):
                    try: os.remove(p)
                    except FileNotFoundError: pass
            self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()
        else:
            self.send_response(404); self.end_headers()

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers(); self.wfile.write(body)

    def _bytes(self, key, ctype):
        with _VIZ_STATE["lock"]:
            data = _VIZ_STATE[key]
        if data is None:
            self.send_response(503); self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", "9"); self.end_headers()
            self.wfile.write(b"not ready"); return
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache, max-age=0")
        self.end_headers(); self.wfile.write(data)

    def _status(self):
        with _VIZ_STATE["lock"]:
            status = dict(_VIZ_STATE["status"])
        status["config_path"] = _VIZ_STATE.get("config_path")
        self._json(status)

    def _archive(self, params):
        arch = _VIZ_STATE.get("archiver")
        if not arch:
            self._json({"enabled": False, "episodes": []}); return
        limit = int(params.get("n", 30))
        res = arch.summary()
        res["episodes"] = arch.list_episodes(limit=limit)
        self._json(res)

    def _config_get(self):
        cfg_path = _VIZ_STATE.get("config_path")
        if cfg_path and os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("static", {}).get("mapbox_token"):
                cfg["static"]["mapbox_token"] = "***"
            self._json(cfg)
        else:
            self._json({"version": 1, "note": "no config persisted",
                        "runtime": _snapshot(), "static": _static_snapshot()})

    def _config_post(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            new_cfg = json.loads(self.rfile.read(length))
        except Exception:
            self._json({"ok": False, "error": "bad request"}, 400); return
        cfg_path = _VIZ_STATE.get("config_path")
        if not cfg_path:
            self._json({"ok": False, "error": "no config path"}, 400); return
        if not isinstance(new_cfg, dict):
            self._json({"ok": False, "error": "config must be object"}, 400); return
        existing = _load_json_file(cfg_path) or {}
        now = datetime.now(timezone.utc).isoformat()
        if new_cfg.get("static", {}).get("mapbox_token") == "***":
            new_cfg["static"]["mapbox_token"] = existing.get("static", {}).get("mapbox_token")
        new_cfg.setdefault("version", 1)
        new_cfg.setdefault("created_at", existing.get("created_at", now))
        new_cfg["updated_at"] = now
        if _save_json_file(cfg_path, new_cfg):
            self._json({"ok": True, "restart_required": True, "path": cfg_path})
        else:
            self._json({"ok": False, "error": "save failed"}, 500)

    def _settings_post(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            updates = json.loads(self.rfile.read(length))
        except Exception:
            self._json({"ok": False, "error": "bad request"}, 400); return
        changed = []
        with _RUNTIME_LOCK:
            for k, v in updates.items():
                if k not in _RUNTIME_TYPES: continue
                try:
                    _RUNTIME[k] = _coerce(k, v); changed.append(k)
                except (TypeError, ValueError):
                    continue
            current = dict(_RUNTIME)
        cfg_path = _VIZ_STATE.get("config_path")
        if cfg_path:
            existing = _load_json_file(cfg_path) or {}
            now = datetime.now(timezone.utc).isoformat()
            merged = {"version": 1,
                      "created_at": existing.get("created_at", now),
                      "updated_at": now,
                      "runtime": current,
                      "static": existing.get("static") or _static_snapshot()}
            _save_json_file(cfg_path, merged)
        self._json({"ok": True, "changed": changed, "settings": current})

    def _accuracy(self, params):
        acc = _VIZ_STATE.get("acc_store")
        if not acc:
            self._json({"error": "no store"}, 503); return
        self._json(acc.stats(since_iso=params.get("since"),
                             tag=params.get("tag"),
                             kind=params.get("kind")))

    def _accuracy_predictions(self, params):
        acc = _VIZ_STATE.get("acc_store")
        if not acc: self._json({"error": "no store"}, 503); return
        self._json(acc.recent_predictions(n=int(params.get("n", 50)),
                                          kind=params.get("kind")))

    def _accuracy_events(self, params):
        acc = _VIZ_STATE.get("acc_store")
        if not acc: self._json({"error": "no store"}, 503); return
        self._json(acc.recent_outcomes(n=int(params.get("n", 50)),
                                       kind=params.get("kind")))

    def _html(self):
        body = _INDEX_HTML.encode("utf-8")
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def _start_web_server(port):
    srv = ThreadingHTTPServer(("0.0.0.0", port), _VizHandler)
    threading.Thread(target=srv.serve_forever, daemon=True, name="viz-http").start()
    return srv


# ----------------------------------------------------------------------------
# Accuracy store
# ----------------------------------------------------------------------------
class AccuracyStore:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.pred_path = os.path.join(data_dir, "predictions.jsonl")
        self.out_path = os.path.join(data_dir, "outcomes.jsonl")
        self.score_path = os.path.join(data_dir, "scores.jsonl")
        self.lock = threading.Lock()
        self.predictions = []; self.outcomes = []; self.scores = []
        self._next_id = 1
        self.observed_cluster_arrivals = set()
        self.enabled = bool(data_dir)
        if self.enabled:
            try: os.makedirs(data_dir, exist_ok=True)
            except OSError as e:
                print(f"accuracy: cannot create {data_dir}: {e}", file=sys.stderr)
                self.enabled = False; return
            self._load()

    def _load(self):
        for path, target in [(self.pred_path, self.predictions),
                             (self.out_path, self.outcomes),
                             (self.score_path, self.scores)]:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line: continue
                        try: target.append(json.loads(line))
                        except json.JSONDecodeError: continue
            except FileNotFoundError: pass
        for lst in (self.predictions, self.outcomes, self.scores):
            if len(lst) > 5000: del lst[:len(lst) - 5000]
        max_id = 0
        for p in self.predictions:
            if isinstance(p.get("id"), int) and p["id"] > max_id: max_id = p["id"]
        self._next_id = max_id + 1
        resolved = {s.get("pred_id") for s in self.scores}
        for p in self.predictions:
            if p.get("id") in resolved: p["resolved"] = True

    def _append_line(self, path, entry):
        if not self.enabled: return
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e: print(f"accuracy: append error {e}", file=sys.stderr)

    def predict(self, ts_iso, kind, value, key=None, tag=None,
                settings=None, horizon_min=None):
        if not self.enabled: return None
        entry = {"id": self._next_id, "ts": ts_iso, "kind": kind,
                 "value": float(value),
                 "horizon_min": float(horizon_min if horizon_min is not None else value),
                 "key": key, "tag": tag or ""}
        if settings: entry["settings"] = settings
        with self.lock:
            self._next_id += 1
            self.predictions.append(entry)
            if len(self.predictions) > 5000: del self.predictions[:1000]
            self._append_line(self.pred_path, entry)
        return entry["id"]

    def observe(self, ts_iso, kind, key=None, tag=None, extra=None):
        if not self.enabled: return
        entry = {"ts": ts_iso, "kind": kind, "key": key, "tag": tag or ""}
        if extra: entry.update(extra)
        with self.lock:
            self.outcomes.append(entry)
            if len(self.outcomes) > 5000: del self.outcomes[:1000]
            self._append_line(self.out_path, entry)
            self._match_and_score(entry)

    def _match_and_score(self, outcome):
        try: now = datetime.fromisoformat(outcome["ts"])
        except Exception: return
        best = None
        for p in reversed(self.predictions):
            if p.get("resolved"): continue
            if p.get("kind") != outcome["kind"]: continue
            if outcome.get("key") is not None and p.get("key") != outcome["key"]: continue
            try: pred_ts = datetime.fromisoformat(p["ts"])
            except Exception: continue
            delta_min = (now - pred_ts).total_seconds() / 60.0
            if delta_min < 0: continue
            horizon = max(p.get("horizon_min") or 0.0, 0.0)
            tol = max(MATCH_TOLERANCE_MIN, horizon * MATCH_TOLERANCE_FACTOR)
            if delta_min > tol: continue
            best = p; break
        if best is None: return
        best["resolved"] = True
        for p in self.predictions:
            if (p is not best and p.get("kind") == best.get("kind")
                and p.get("key") == best.get("key")
                and (p.get("ts") or "") < (best.get("ts") or "")
                and not p.get("resolved")):
                p["resolved"] = True; p["superseded"] = True
        try: pred_ts = datetime.fromisoformat(best["ts"])
        except Exception: return
        predicted_at = pred_ts + timedelta(minutes=best["value"])
        error_min = (now - predicted_at).total_seconds() / 60.0
        score = {"pred_id": best["id"], "pred_ts": best["ts"],
                 "event_ts": outcome["ts"], "kind": outcome["kind"],
                 "key": best.get("key"), "predicted_min": best["value"],
                 "horizon_min": best.get("horizon_min"),
                 "actual_min": (now - pred_ts).total_seconds() / 60.0,
                 "error_min": error_min, "tag": best.get("tag") or ""}
        self.scores.append(score)
        if len(self.scores) > 5000: del self.scores[:1000]
        self._append_line(self.score_path, score)

    def _summarize(self, errs):
        if not errs:
            return {"n": 0, "median_error_min": None, "mean_abs_error_min": None,
                    "median_abs_error_min": None, "p90_abs_error_min": None,
                    "min_error_min": None, "max_error_min": None, "samples": []}
        abs_e = sorted(abs(e) for e in errs); n = len(abs_e)
        return {"n": n, "median_error_min": float(np.median(errs)),
                "mean_abs_error_min": float(np.mean(abs_e)),
                "median_abs_error_min": float(np.median(abs_e)),
                "p90_abs_error_min": float(abs_e[int(0.9 * (n - 1))]),
                "min_error_min": float(min(errs)),
                "max_error_min": float(max(errs)),
                "samples": [round(e, 1) for e in errs[-20:]]}

    def stats(self, since_iso=None, tag=None, kind=None):
        with self.lock:
            scores = list(self.scores); predictions = list(self.predictions)
            n_outcomes = len(self.outcomes)
        if since_iso:
            try:
                cutoff = datetime.fromisoformat(since_iso)
                scores = [s for s in scores
                          if _safe_fromiso(s.get("event_ts")) >= cutoff]
            except Exception: pass
        if tag is not None: scores = [s for s in scores if (s.get("tag") or "") == tag]
        if kind is not None: scores = [s for s in scores if s.get("kind") == kind]
        by_kind, by_tag = {}, {}
        for s in scores:
            err = s.get("error_min")
            if err is None: continue
            by_kind.setdefault(s["kind"], []).append(err)
            by_tag.setdefault(s.get("tag") or "(untagged)", []).append(err)
        pending_by_kind = {}; superseded_count = 0
        for p in predictions:
            if not p.get("resolved"):
                k = p.get("kind", "unknown")
                pending_by_kind[k] = pending_by_kind.get(k, 0) + 1
            elif p.get("superseded"):
                superseded_count += 1
        return {"total_scores": len(scores), "total_predictions": len(predictions),
                "total_outcomes": n_outcomes, "superseded_predictions": superseded_count,
                "pending_by_kind": pending_by_kind,
                "by_kind": {k: self._summarize(v) for k, v in by_kind.items()},
                "by_tag": {k: self._summarize(v) for k, v in by_tag.items()}}

    def recent_predictions(self, n=50, kind=None):
        with self.lock: pool = list(self.predictions)
        if kind: pool = [p for p in pool if p.get("kind") == kind]
        return pool[-n:]

    def recent_outcomes(self, n=50, kind=None):
        with self.lock: pool = list(self.outcomes)
        if kind: pool = [p for p in pool if p.get("kind") == kind]
        return pool[-n:]

    def reset_scores(self):
        with self.lock: self.scores = []
        if not self.enabled: return
        try: open(self.score_path, "w").close()
        except OSError as e: print(f"accuracy: reset error {e}", file=sys.stderr)
        for p in self.predictions:
            p.pop("resolved", None); p.pop("superseded", None)

    def replay(self):
        with self.lock:
            preds = sorted(self.predictions, key=lambda p: p["ts"])
            outcomes = sorted(self.outcomes, key=lambda o: o["ts"])
            self.scores = []
            for p in preds:
                p.pop("resolved", None); p.pop("superseded", None)
            self.predictions = preds
            for o in outcomes: self._match_and_score(o)
        if not self.enabled: return
        try:
            with open(self.score_path, "w", encoding="utf-8") as fh:
                for s in self.scores:
                    fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        except OSError as e: print(f"accuracy: replay error {e}", file=sys.stderr)


def _safe_fromiso(s):
    if not s: return datetime.min.replace(tzinfo=timezone.utc)
    try: return datetime.fromisoformat(s)
    except Exception: return datetime.min.replace(tzinfo=timezone.utc)


# ----------------------------------------------------------------------------
# HA sensors
# ----------------------------------------------------------------------------
@dataclass
class SensorSpec:
    component: str
    name: str
    device_class: Optional[str] = None
    unit: Optional[str] = None
    icon: Optional[str] = None
    diagnostic: bool = False


SENSORS = {
    "raining_now": SensorSpec("binary_sensor", "Raining now", "moisture", None, "mdi:weather-pouring"),
    "raining_confirmed": SensorSpec("binary_sensor", "Raining (confirmed)", "moisture", None, "mdi:check-circle"),
    "rain_confidence": SensorSpec("sensor", "Rain confidence", "enum", None, "mdi:shield-check"),
    "rain_duration_min": SensorSpec("sensor", "Rain duration", "duration", "min", "mdi:timer-outline"),
    "rain_end_eta_min": SensorSpec("sensor", "Rain end ETA", "duration", "min", "mdi:weather-rainy"),
    "next_rain_eta_min": SensorSpec("sensor", "Next rain ETA", "duration", "min", "mdi:weather-pouring"),
    "gap_min": SensorSpec("sensor", "Rain gap", "duration", "min", "mdi:weather-sunny"),
    "summary": SensorSpec("sensor", "Summary", None, None, "mdi:text-box-outline"),
    "radar_intensity_mmh": SensorSpec("sensor", "Radar intensity at target", "precipitation_intensity", "mm/h", "mdi:weather-pouring"),
    "nearest_station_mm": SensorSpec("sensor", "Nearest wet station 1h", "precipitation", "mm", "mdi:cup-water"),
    "intensity_mmh": SensorSpec("sensor", "Station intensity (slope)", "precipitation_intensity", "mm/h", "mdi:weather-rainy"),
    "station_trend": SensorSpec("sensor", "Nearby stations consensus trend", "enum", None, "mdi:trending-up"),
    "upstream_trend": SensorSpec("sensor", "Upstream stations trend", "enum", None, "mdi:trending-up"),
    "local_temp_c": SensorSpec("sensor", "Local temperature", "temperature", "°C", "mdi:thermometer"),
    "local_humidity": SensorSpec("sensor", "Local humidity", "humidity", "%", "mdi:water-percent"),
    "local_pressure_hpa": SensorSpec("sensor", "Local air pressure", "atmospheric_pressure", "hPa", "mdi:gauge"),
    "local_wind_speed_kmh": SensorSpec("sensor", "Local wind speed", "wind_speed", "km/h", "mdi:weather-windy"),
    "local_wind_gust_kmh": SensorSpec("sensor", "Local wind gust", "wind_speed", "km/h", "mdi:weather-windy-variant"),
    "local_wind_dir_deg": SensorSpec("sensor", "Local wind direction", "wind_direction", "°", "mdi:compass-outline"),
    "local_wind_dir": SensorSpec("sensor", "Local wind cardinal", None, None, "mdi:compass"),
    "local_rain_today_mm": SensorSpec("sensor", "Local rainfall today", "precipitation", "mm", "mdi:weather-rainy"),
    "local_rain_1h_mm": SensorSpec("sensor", "Local rainfall 1h", "precipitation", "mm", "mdi:weather-pouring"),
    "local_rain_rate_mmh": SensorSpec("sensor", "Local rain rate", "precipitation_intensity", "mm/h", "mdi:weather-pouring"),
    "local_weather_stations": SensorSpec("sensor", "Local weather stations", None, "stations", "mdi:weather-partly-cloudy", diagnostic=True),
    "coverage_map": SensorSpec("sensor", "Coverage map", None, None, "mdi:map", diagnostic=True),
    "nowcast_error_min": SensorSpec("sensor", "Nowcast error (median)", "duration", "min",
                                    "mdi:chart-timeline-variant", True),
    "data_age_min": SensorSpec("sensor", "Radar frame age", "duration", "min", "mdi:clock-alert-outline", True),
    "frame_timestamp": SensorSpec("sensor", "Radar frame timestamp", "timestamp", None, "mdi:clock-outline", True),
    "cell_count": SensorSpec("sensor", "Approaching cell count", None, None, "mdi:weather-lightning-rainy", True),
    "nearest_cell_km": SensorSpec("sensor", "Nearest approaching cell distance", "distance", "km", "mdi:map-marker-distance", True),
    "nearest_cell_eta_min": SensorSpec("sensor", "Nearest approaching cell ETA", "duration", "min", "mdi:map-marker-path", True),
    "nearest_station_km": SensorSpec("sensor", "Nearest wet station distance", "distance", "km", "mdi:map-marker", True),
    "nearest_station_age_min": SensorSpec("sensor", "Nearest wet station age", "duration", "min", "mdi:clock-outline", True),
    "station_count_wet": SensorSpec("sensor", "Wet stations nearby", None, "stations", "mdi:map-marker-check", True),
    "station_count_fresh": SensorSpec("sensor", "Fresh stations nearby", None, "stations", "mdi:map-marker-multiple", True),
    "station_wet_fraction": SensorSpec("sensor", "Nearby stations wet fraction", None, "%", "mdi:percent", True),
    "station_median_mm": SensorSpec("sensor", "Median wet station 1h", "precipitation", "mm", "mdi:cup-water", True),
}


def parse_sensor_list(spec):
    if not spec or spec.strip().lower() == "all":
        return list(SENSORS.keys())
    if spec.strip().lower() == "none":
        return []
    keys = [k.strip() for k in spec.split(",") if k.strip()]
    bad = [k for k in keys if k not in SENSORS]
    if bad: raise ValueError(f"Unknown sensor keys: {', '.join(bad)}")
    return keys


# ----------------------------------------------------------------------------
# MQTT publisher
# ----------------------------------------------------------------------------
class HAPublisher:
    def __init__(self, host, port, user, password, use_tls,
                 discovery_prefix, state_prefix, device_id, device_name,
                 target_name, sensors, mark_offline_on_exit, retain=True,
                 threat_topic=None):
        if not HAVE_MQTT: raise RuntimeError("paho-mqtt not installed")
        self.host, self.port = host, port
        self.dp = discovery_prefix; self.sp = state_prefix
        self.device_id = device_id; self.device_name = device_name
        self.target_name = target_name; self.sensors = sensors
        self.mark_offline = mark_offline_on_exit; self.retain = retain
        self.threat_topic = threat_topic; self.connected = False
        self.client = mqtt.Client(client_id=f"{device_id}_pub", clean_session=True)
        if user: self.client.username_pw_set(user, password)
        if use_tls: self.client.tls_set()
        self.client.reconnect_delay_set(min_delay=2, max_delay=30)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            print(f"MQTT: connected {self.host}:{self.port}")
            self._discovery(); self._avail(True)
        else:
            print(f"MQTT: connect failed rc={rc}", file=sys.stderr)

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False

    def connect(self):
        try:
            self.client.connect_async(self.host, self.port, keepalive=60)
            self.client.loop_start()
        except Exception as e: print(f"MQTT: connection error {e}", file=sys.stderr)

    def _discovery(self):
        avail = f"{self.sp}/{self.device_id}/availability"
        for key in self.sensors:
            spec = SENSORS[key]
            payload = {"name": spec.name, "unique_id": f"{self.device_id}_{key}",
                       "state_topic": f"{self.sp}/{self.device_id}/{key}/state",
                       "availability_topic": avail,
                       "device": {"identifiers": [self.device_id], "name": self.device_name,
                                  "manufacturer": "DHMZ+pljusak",
                                  "model": f"Radar+station — {self.target_name}"}}
            if spec.device_class: payload["device_class"] = spec.device_class
            if spec.unit: payload["unit_of_measurement"] = spec.unit
            if spec.icon: payload["icon"] = spec.icon
            if spec.diagnostic: payload["entity_category"] = "diagnostic"
            self.client.publish(f"{self.dp}/{spec.component}/{self.device_id}/{key}/config",
                                json.dumps(payload), retain=True)

    def _avail(self, online):
        self.client.publish(f"{self.sp}/{self.device_id}/availability",
                            "online" if online else "offline", retain=self.retain)

    def publish_states(self, states):
        if not self.connected: return
        for key, v in states.items():
            if key not in self.sensors: continue
            topic = f"{self.sp}/{self.device_id}/{key}/state"
            if v is None:
                self.client.publish(topic, "", retain=self.retain); continue
            if isinstance(v, bool): p = "ON" if v else "OFF"
            elif isinstance(v, datetime): p = v.astimezone(timezone.utc).isoformat()
            else: p = str(v)
            self.client.publish(topic, p, retain=self.retain)

    def publish_threat(self, payload):
        if not self.connected or not self.threat_topic: return
        try:
            self.client.publish(self.threat_topic,
                                json.dumps(payload, ensure_ascii=False), retain=False)
        except Exception as e: print(f"MQTT: threat publish error {e}", file=sys.stderr)

    def disconnect(self):
        try:
            if self.mark_offline: self._avail(False)
            time.sleep(0.2); self.client.loop_stop(); self.client.disconnect()
        except Exception: pass


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def run(target_name, t_lat, t_lon, calib, once, dump_jsonl,
        frame_interval_min, auto_frame_interval,
        timestamp_region, timestamp_source, timestamp_tz,
        display_tz, display_tz_name, ha, pljusak, pljusak_interval,
        intensity_mode, sri_legend, sri_thresholds,
        confirm_radius, confirm_mm, confirm_age,
        confirm_frac, confirm_min_count, threat_topic,
        web_enabled, web_port, acc_store, archiver, data_dir,
        mapbox_token, mapbox_style, basemap_size):
    out_fh = open(dump_jsonl, "a") if dump_jsonl else None
    rt = _snapshot()
    print(f"Target: {target_name} ({t_lat:.4f}, {t_lon:.4f})")
    if data_dir:
        print(f"Data dir: {data_dir}")
    print(f"Runtime: radius={rt['radius_km']:g}km "
          f"wx_radius={rt['weather_radius_km']:g}km "
          f"archive={'on' if rt['archive_enabled'] else 'off'}({rt['archive_radius_km']:g}km) "
          f"approach={rt['approaching_radius_km']:g}km "
          f"threat={rt['threat_radius_km']:g}km "
          f"min_sri={rt['threat_min_sri_mmh']:g}mm/h "
          f"basemap={rt['basemap']} "
          f"rot={'auto' if rt['view_rotation_auto'] else rt['view_rotation']}")
    print(f"Pljusak: {'on' if pljusak else 'off'} | HA: {'on' if ha else 'off'} | "
          f"Web: " + (f"http://0.0.0.0:{web_port}" if web_enabled else "off"))
    print(f"Frame interval: {frame_interval_min:.1f} min | once: {once}\n")

    legend_mask = None
    last_pljusak = None
    prev_threat_keys = set()
    prev_raining_now = None
    cached_basemap = None
    cached_basemap_key = None
    cached_basemap_kind = "greyscale"
    cached_coast_rot = None
    cached_coast_rot_key = None

    def maybe_pljusak():
        nonlocal last_pljusak
        if pljusak is None: return
        now = datetime.now(timezone.utc)
        if last_pljusak is None or (now - last_pljusak).total_seconds() >= pljusak_interval:
            if pljusak.fetch(now): last_pljusak = now

    def get_rotation_with_satellite(rt_current):
        nonlocal cached_coast_rot, cached_coast_rot_key
        if not rt_current["view_rotation_auto"]:
            return False, rt_current["view_rotation"]

        bm = rt_current["basemap"]
        use_sat = bool(mapbox_token) and bm in ("auto", "mapbox")
        c_key = (use_sat, mapbox_style, mapbox_token, t_lat, t_lon,
                 rt_current["radius_km"], basemap_size)
        if c_key == cached_coast_rot_key and cached_coast_rot is not None:
            return True, cached_coast_rot

        rot = None
        if use_sat:
            sat_north_up = fetch_basemap(
                mapbox_style, mapbox_token, t_lat, t_lon,
                rt_current["radius_km"], 0.0, basemap_size, data_dir)
            rot = detect_coast_rotation_from_satellite(sat_north_up)

        if rot is None:
            rot, _ = compute_coast_rotation(t_lat, t_lon)
        if rot is None:
            rot = rt_current["view_rotation"]

        cached_coast_rot = rot
        cached_coast_rot_key = c_key
        return True, rot

    def resolve_basemap(rt_current, view_rot_used):
        bm = rt_current["basemap"]
        if bm == "none":
            return None, "none"
        if bm == "greyscale":
            return None, "greyscale"
        if bm == "mapbox" and not mapbox_token:
            print("WARNING: basemap=mapbox but no token set; falling back to greyscale",
                  file=sys.stderr)
            return None, "greyscale"
        if bm in ("mapbox", "auto") and mapbox_token:
            img = fetch_basemap(mapbox_style, mapbox_token, t_lat, t_lon,
                                rt_current["radius_km"], view_rot_used,
                                basemap_size, data_dir)
            if img is not None:
                return img, "mapbox"
        return None, "greyscale"

    try:
        while True:
            try:
                rt = _snapshot()
                cur_tag = rt["settings_tag"]
                view_rot_auto, view_rot_used = get_rotation_with_satellite(rt)

                maybe_pljusak()
                img = fetch_image(GIF_URL)
                now_utc = datetime.now(timezone.utc)
                now_iso = now_utc.isoformat()
                now_local = now_utc.astimezone(display_tz)
                tz_ab = tz_abbrev(now_local)

                if sri_legend is not None and not sri_legend.is_ready():
                    try: img.seek(0)
                    except Exception: pass
                    sri_legend.calibrate(img.convert("RGB"))
                    sri_legend.print_samples()
                    try:
                        img.seek(0); f0 = img.convert("RGB")
                        h_full, w_full = f0.size[1], f0.size[0]
                    except Exception: h_full, w_full = None, None
                    if h_full and w_full:
                        lx0, ly0, lx1, ly1 = sri_legend.region
                        legend_mask = np.zeros((h_full, w_full), dtype=bool)
                        legend_mask[max(0, ly0):min(h_full, ly1),
                                    max(0, lx0):min(w_full, lx1)] = True
                    try: img.seek(0)
                    except Exception: pass

                frame_ts_utc, frame_raw, ts_idx = read_frame_timestamp(
                    img, timestamp_region, timestamp_tz, timestamp_source)
                data_age_min = ((now_utc - frame_ts_utc).total_seconds() / 60.0
                                if frame_ts_utc else None)

                eff_interval = frame_interval_min
                if auto_frame_interval and HAVE_TESSERACT and timestamp_region:
                    first_ts, _, _ = read_frame_timestamp(
                        img, timestamp_region, timestamp_tz, "first")
                    nf = getattr(img, "n_frames", 1)
                    if first_ts and frame_ts_utc and nf > 1:
                        span = (frame_ts_utc - first_ts).total_seconds() / 60.0
                        if span > 0: eff_interval = span / (nf - 1)

                (last_mask, last_mmh, tracks, n_frames, rain_history,
                 label_array, tracks_by_id, split_frames_by_id,
                 cluster_first_hit, frame_masks_mmh) = track_animation(
                    img, calib, eff_interval, t_lat, t_lon, intensity_mode,
                    sri_legend, legend_mask)

                filtered = []
                for tr in tracks:
                    if tr.n_observations < rt["min_duration_frames"]: continue
                    if tr.cluster.size < rt["min_size_px"]: continue
                    filtered.append(tr)
                tracks = filtered

                approaches = [assess(tr, t_lat, t_lon, calib) for tr in tracks]
                approaching_flags = [refine_approaching(tr, a, rt)
                                     for tr, a in zip(tracks, approaches)]
                threat_flags = [refine_threat(tr, a, rt)
                                for tr, a in zip(tracks, approaches)]
                n_approaching = sum(approaching_flags)

                nc = compute_nowcast(calib, t_lat, t_lon, tracks, approaches,
                                     rain_history, label_array, eff_interval, rt)

                wx_radius = float(rt.get("weather_radius_km", 5.0))
                nearby_stations = []
                all_stations = []
                if pljusak is not None:
                    nearby_stations = pljusak.analyze_nearby(
                        t_lat, t_lon, max(confirm_radius * 2, wx_radius * 2, 40),
                        display_tz, max_age_min=confirm_age * 2)
                    all_stations = pljusak.get_all_stations(
                        t_lat, t_lon, display_tz)

                lw = summarize_local_weather(nearby_stations, wx_radius, confirm_age)

                ca = combine_analysis(nc.raining_now, nearby_stations, tracks, approaches,
                                      calib, t_lat, t_lon, confirm_radius,
                                      confirm_mm, confirm_age,
                                      confirm_frac, confirm_min_count, rt)

                archived_now = 0
                if archiver is not None:
                    last_frame_utc = frame_ts_utc
                    if last_frame_utc and timestamp_source == "first" and n_frames > 1:
                        last_frame_utc = last_frame_utc + timedelta(
                            minutes=(n_frames - 1) * eff_interval)
                    archived_now = archiver.process_animation(
                        img, frame_masks_mmh, last_frame_utc, eff_interval,
                        now_utc, rt, local_weather=lw,
                        target_raining=ca.raining_now)

                if acc_store and acc_store.enabled:
                    settings_snapshot = {k: rt[k] for k in
                                         ("approaching_radius_km", "cluster_radius_weight",
                                          "only_approach_deg", "threat_radius_km",
                                          "threat_angle_deg", "threat_min_duration",
                                          "threat_min_sri_mmh")}
                    if (not nc.raining_now) and nc.next_rain_eta_min is not None:
                        acc_store.predict(now_iso, "rain_start", nc.next_rain_eta_min,
                                          tag=cur_tag, settings=settings_snapshot)
                    if nc.raining_now and nc.rain_end_eta_min is not None:
                        acc_store.predict(now_iso, "rain_end", nc.rain_end_eta_min,
                                          tag=cur_tag, settings=settings_snapshot)
                    for tr, a, af, tf in zip(tracks, approaches, approaching_flags, threat_flags):
                        if af and a.eta_min is not None \
                                and tr.cid not in acc_store.observed_cluster_arrivals:
                            acc_store.predict(now_iso, "cluster_arrive", a.eta_min,
                                              key=str(tr.cid), tag=cur_tag,
                                              settings=settings_snapshot)
                        if tf:
                            acc_store.predict(now_iso, "threat", 0.0,
                                              key=str(tr.cid), tag=cur_tag,
                                              settings=settings_snapshot)
                    if prev_raining_now is not None and prev_raining_now != ca.raining_now:
                        acc_store.observe(now_iso,
                                          "rain_start" if ca.raining_now else "rain_end",
                                          tag=cur_tag)
                    prev_raining_now = ca.raining_now
                    for cid, first_hit_frame in cluster_first_hit.items():
                        if cid in acc_store.observed_cluster_arrivals: continue
                        arrival_min_ago = (n_frames - 1 - first_hit_frame) * eff_interval
                        arrival_ts = now_utc - timedelta(minutes=arrival_min_ago)
                        acc_store.observe(arrival_ts.isoformat(), "cluster_arrive",
                                          key=str(cid), tag=cur_tag)
                        acc_store.observed_cluster_arrivals.add(cid)
                else:
                    prev_raining_now = ca.raining_now

                radar_int = None
                px, py = calib.latlon_to_image_pixel(t_lat, t_lon)
                xi, yi = int(round(px)), int(round(py))
                if (last_mmh is not None and 0 <= yi < last_mmh.shape[0]
                        and 0 <= xi < last_mmh.shape[1]):
                    v = float(last_mmh[max(0, yi-1):yi+2, max(0, xi-1):xi+2].max())
                    if v > 0: radar_int = v

                nearest_cell = None
                for i, (tr, a) in enumerate(zip(tracks, approaches)):
                    if not approaching_flags[i]: continue
                    clat, clon = calib.image_pixel_to_latlon(tr.cluster.cx, tr.cluster.cy)
                    d = haversine_km(clat, clon, t_lat, t_lon)
                    if nearest_cell is None or d < nearest_cell[0]:
                        nearest_cell = (d, a.eta_min)

                age_str = ""
                if data_age_min is not None:
                    if data_age_min > VERY_STALE_MIN:
                        age_str = f" | radar {data_age_min:.0f} min ⚠⚠"
                    elif data_age_min > STALE_MIN:
                        age_str = f" | radar {data_age_min:.0f} min ⚠"
                arch_str = f" | archived +{archived_now}" if archived_now > 0 else ""
                print(f"\n=== {now_local.strftime('%H:%M:%S')} {tz_ab} | "
                      f"cells:{len(tracks)} ({n_approaching} approaching) | "
                      f"frames:{n_frames} | {ca.confidence}{age_str}{arch_str} | "
                      f"rot:{view_rot_used:.0f}°{'(auto)' if view_rot_auto else ''} ===")
                for line in nc.lines: print(line)
                if lw.station_count > 0:
                    wx_parts = [f"Local ({lw.station_count} st ≤{wx_radius:g}km):"]
                    if lw.temp_c is not None: wx_parts.append(f"{lw.temp_c:.1f}°C")
                    if lw.humidity_pct is not None: wx_parts.append(f"{lw.humidity_pct}%")
                    if lw.pressure_hpa is not None: wx_parts.append(f"{lw.pressure_hpa:.1f}hPa")
                    if lw.wind_speed_kmh is not None:
                        w_s = f"wind {lw.wind_speed_kmh:.1f}km/h"
                        if lw.wind_gust_kmh is not None:
                            w_s += f" (gust {lw.wind_gust_kmh:.1f})"
                        if lw.wind_dir_deg is not None:
                            w_s += f" {lw.wind_dir_cardinal}/{lw.wind_dir_deg}°"
                        wx_parts.append(w_s)
                    if lw.precip_today_mm is not None:
                        r_s = f"rain {lw.precip_today_mm:.1f}mm"
                        if lw.rain_rate_mmh is not None and lw.rain_rate_mmh > 0:
                            r_s += f" ({lw.rain_rate_mmh:.1f}mm/h)"
                        wx_parts.append(r_s)
                    print("  " + " | ".join(wx_parts))
                if radar_int is not None:
                    print(f"  Radar intensity at target: {radar_int:.1f} mm/h")
                if ca.nearest_station:
                    ns = ca.nearest_station
                    print(f"  Nearest wet: {ns.reading.name} "
                          f"({ns.dist_km:.1f} km, {ns.reading.precip_1h:.1f} mm/1h, {ns.trend})")
                for i, (tr, a, tf) in enumerate(zip(tracks, approaches, threat_flags)):
                    if not approaching_flags[i]: continue
                    r_eff = rt["approaching_radius_km"] + a.cluster_radius_km * rt["cluster_radius_weight"]
                    line = (f"  {cluster_label(i)} [{tr.cluster.size:4d}px "
                            f"max={tr.cluster.max_mmh:4.1f}mm/h {tr.speed_kmh:4.1f}km/h]")
                    if a.closest_km is not None:
                        line += f" miss={a.closest_km:4.1f}/{r_eff:4.1f}km"
                    if a.eta_min is not None:
                        line += f" ETA={a.eta_min:4.0f}min"
                    if tf: line += " ⚠ THREAT"
                    print(line)

                bm_key = (rt["basemap"], mapbox_style, mapbox_token, t_lat, t_lon,
                          rt["radius_km"], view_rot_used, basemap_size)
                if bm_key != cached_basemap_key:
                    cached_basemap, cached_basemap_kind = resolve_basemap(rt, view_rot_used)
                    cached_basemap_key = bm_key

                png = gif = None
                if web_enabled:
                    try:
                        header = (f"{target_name} | {ca.confidence} | "
                                  f"{n_approaching} appr | {sum(threat_flags)} thr | "
                                  f"{now_local.strftime('%H:%M %Z')} | "
                                  f"rot {view_rot_used:.0f}°")
                        png = render_composed_png(
                            img, calib, t_lat, t_lon, rt["radius_km"],
                            view_rot_used, basemap_size, cached_basemap,
                            cached_basemap_kind,
                            tracks, approaches, approaching_flags, threat_flags,
                            all_stations, rt["show_cities"],
                            intensity_mode, sri_legend, legend_mask,
                            sri_thresholds, rt, confirm_mm=confirm_mm,
                            header_text=header)
                        gif = render_composed_gif(
                            img, calib, t_lat, t_lon, rt["radius_km"],
                            view_rot_used, basemap_size, cached_basemap,
                            cached_basemap_kind,
                            tracks, tracks_by_id, split_frames_by_id,
                            eff_interval, approaches, approaching_flags,
                            threat_flags, all_stations, rt["show_cities"],
                            intensity_mode, sri_legend, legend_mask,
                            sri_thresholds, rt, confirm_mm=confirm_mm)
                    except Exception as e:
                        print(f"  viz error: {e}", file=sys.stderr)

                threat_keys_now = {tr.cid for tr, tf in zip(tracks, threat_flags) if tf}
                if ha and threat_topic:
                    for i, tr in enumerate(tracks):
                        if tr.cid in (threat_keys_now - prev_threat_keys):
                            a = approaches[i]
                            r_eff = rt["threat_radius_km"] + a.cluster_radius_km * rt["cluster_radius_weight"]
                            ha.publish_threat({
                                "ts": now_iso, "cluster": cluster_label(i),
                                "size_px": tr.cluster.size,
                                "max_mmh": round(tr.cluster.max_mmh, 1),
                                "mean_mmh": round(tr.cluster.mean_mmh, 1),
                                "speed_kmh": round(tr.speed_kmh, 1),
                                "eta_min": round(a.eta_min) if a.eta_min is not None else None,
                                "miss_km": round(a.closest_km, 1) if a.closest_km is not None else None,
                                "threat_radius_km": round(r_eff, 1),
                                "approach_angle": round(a.approach_angle) if a.approach_angle is not None else None,
                                "dist_km": round(a.dist_km, 1), "summary": nc.summary})
                prev_threat_keys = threat_keys_now

                ns = ca.nearest_station
                if ca.confidence == "radar_only":
                    summary_text = "[unconfirmed] " + nc.summary
                elif ca.confidence == "station_only":
                    summary_text = (f"[station-only] Rain at stations, radar dry. "
                                    f"Radar next cell ~{nc.next_rain_eta_min:.0f} min."
                                    if nc.next_rain_eta_min is not None
                                    else "[station-only] Rain at stations, radar dry.")
                elif ca.confidence == "dry" and ca.upstream_trend == "rising":
                    summary_text = nc.summary + " Upstream stations rising."
                elif ca.confidence == "dry" and ca.upstream_trend == "falling":
                    summary_text = nc.summary + " Upstream stations weakening."
                else:
                    summary_text = nc.summary

                nowcast_err_val = None
                if acc_store and acc_store.enabled:
                    ca_stat = acc_store.stats().get("by_kind", {}).get("cluster_arrive", {})
                    if ca_stat.get("median_abs_error_min") is not None:
                        nowcast_err_val = round(ca_stat["median_abs_error_min"])

                sensor_values = {
                    "raining_now": ca.raining_now, "raining_confirmed": ca.confirmed,
                    "rain_confidence": ca.confidence,
                    "rain_duration_min": (round(nc.rain_duration_min)
                                          if nc.rain_duration_min is not None else 0),
                    "rain_end_eta_min": (round(nc.rain_end_eta_min)
                                         if nc.rain_end_eta_min is not None else None),
                    "next_rain_eta_min": (round(nc.next_rain_eta_min)
                                          if nc.next_rain_eta_min is not None else None),
                    "gap_min": round(nc.gap_min) if nc.gap_min is not None else None,
                    "summary": summary_text,
                    "data_age_min": round(data_age_min, 1) if data_age_min is not None else None,
                    "frame_timestamp": frame_ts_utc, "cell_count": n_approaching,
                    "nearest_cell_km": round(nearest_cell[0], 1) if nearest_cell else None,
                    "nearest_cell_eta_min": (round(nearest_cell[1])
                                             if nearest_cell and nearest_cell[1] is not None else None),
                    "radar_intensity_mmh": round(radar_int, 1) if radar_int is not None else None,
                    "nearest_station_mm": round(ns.reading.precip_1h, 1) if ns else None,
                    "nearest_station_km": round(ns.dist_km, 1) if ns else None,
                    "nearest_station_age_min": (round(ns.age_min, 1)
                                                if ns and ns.age_min is not None else None),
                    "intensity_mmh": round(ca.intensity_mmh, 1) if ca.intensity_mmh is not None else None,
                    "station_trend": ca.station_trend if ca.station_trend != "none" else None,
                    "upstream_trend": ca.upstream_trend if ca.upstream_trend != "none" else None,
                    "local_temp_c": lw.temp_c,
                    "local_humidity": lw.humidity_pct,
                    "local_pressure_hpa": lw.pressure_hpa,
                    "local_wind_speed_kmh": lw.wind_speed_kmh,
                    "local_wind_gust_kmh": lw.wind_gust_kmh,
                    "local_wind_dir_deg": lw.wind_dir_deg,
                    "local_wind_dir": lw.wind_dir_cardinal,
                    "local_rain_today_mm": lw.precip_today_mm,
                    "local_rain_1h_mm": lw.precip_1h_mm,
                    "local_rain_rate_mmh": lw.rain_rate_mmh,
                    "local_weather_stations": lw.station_count,
                    "station_count_wet": ca.station_count_wet,
                    "station_count_fresh": ca.station_count_fresh,
                    "station_wet_fraction": (round(ca.wet_fraction * 100)
                                             if ca.wet_fraction is not None else None),
                    "station_median_mm": (round(ca.station_median_mm, 1)
                                          if ca.station_median_mm is not None else None),
                    "coverage_map": "ok" if png else "disabled",
                    "nowcast_error_min": nowcast_err_val,
                }
                if ha: ha.publish_states(sensor_values)

                if web_enabled:
                    status_sensors = {}
                    for k, v in sensor_values.items():
                        status_sensors[k] = v.isoformat() if isinstance(v, datetime) else v
                    threat_list = []
                    for i, (tr, a, tf) in enumerate(zip(tracks, approaches, threat_flags)):
                        if not tf: continue
                        r_eff = rt["threat_radius_km"] + a.cluster_radius_km * rt["cluster_radius_weight"]
                        threat_list.append({"label": cluster_label(i),
                                            "size_px": tr.cluster.size,
                                            "max_mmh": round(tr.cluster.max_mmh, 1),
                                            "mean_mmh": round(tr.cluster.mean_mmh, 1),
                                            "speed_kmh": round(tr.speed_kmh, 1),
                                            "miss_km": (round(a.closest_km, 1)
                                                        if a.closest_km is not None else None),
                                            "threat_radius_km": round(r_eff, 1),
                                            "eta_min": (round(a.eta_min)
                                                        if a.eta_min is not None else None)})
                    local_weather_dict = {
                        "radius_km": lw.radius_km,
                        "station_count": lw.station_count,
                        "nearest_station": lw.nearest_station,
                        "nearest_dist_km": lw.nearest_dist_km,
                        "temp_c": lw.temp_c,
                        "temp_trend": lw.temp_trend,
                        "humidity_pct": lw.humidity_pct,
                        "pressure_hpa": lw.pressure_hpa,
                        "pressure_trend": lw.pressure_trend,
                        "wind_speed_kmh": lw.wind_speed_kmh,
                        "wind_gust_kmh": lw.wind_gust_kmh,
                        "wind_dir_deg": lw.wind_dir_deg,
                        "wind_dir_cardinal": lw.wind_dir_cardinal,
                        "precip_today_mm": lw.precip_today_mm,
                        "precip_1h_mm": lw.precip_1h_mm,
                        "precip_6h_mm": lw.precip_6h_mm,
                        "precip_24h_mm": lw.precip_24h_mm,
                        "rain_rate_mmh": lw.rain_rate_mmh,
                    }
                    status = {
                        "updated": now_iso, "target": target_name,
                        "confidence": ca.confidence, "raining_now": ca.raining_now,
                        "approaching_count": n_approaching,
                        "threat_count": sum(threat_flags),
                        "threats": threat_list,
                        "local_weather": local_weather_dict,
                        "archive": archiver.summary() if archiver else {"enabled": False},
                        "sensors": status_sensors,
                        "view_rotation_auto": view_rot_auto,
                        "view_rotation_used": view_rot_used,
                        "basemap_used": cached_basemap_kind,
                    }
                    with _VIZ_STATE["lock"]:
                        _VIZ_STATE["status"] = status
                        if png is not None: _VIZ_STATE["png"] = png
                        if gif is not None: _VIZ_STATE["gif"] = gif
                        _VIZ_STATE["updated"] = now_iso

            except Exception as e:
                print(f"  fetch error: {e}", file=sys.stderr)

            if once: return
            cycle_start = time.monotonic()
            while True:
                if _FORCE_REFRESH.is_set():
                    _FORCE_REFRESH.clear(); break
                maybe_pljusak()
                cur_poll = _snapshot()["poll_s"]
                if time.monotonic() >= cycle_start + cur_poll: break
                rem = (cycle_start + cur_poll) - time.monotonic()
                try: time.sleep(max(0.5, min(rem, 5)))
                except KeyboardInterrupt: return
    finally:
        if out_fh: out_fh.close()
        if ha: ha.disconnect()


def get_effective_rotation(rt, t_lat, t_lon):
    if rt["view_rotation_auto"]:
        rot, _ = compute_coast_rotation(t_lat, t_lon)
        if rot is not None:
            return True, rot
        return True, rt["view_rotation"]
    return False, rt["view_rotation"]


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_region(s):
    if not s: return None
    parts = [int(p.strip()) for p in s.split(",")]
    if len(parts) != 4: raise ValueError("region must be 'x0,y0,x1,y1'")
    return tuple(parts)


def parse_floats(s):
    if not s: return None
    return tuple(float(x.strip()) for x in s.split(","))


def parse_sri_breaks(s):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts: raise ValueError("empty break list")
    has_at = ["@" in p for p in parts]
    if all(has_at):
        vals, poss = [], []
        for p in parts:
            v, pos = p.split("@", 1)
            vals.append(float(v)); poss.append(float(pos))
        if any(p < 0 or p > 1 for p in poss):
            raise ValueError("positions must be in [0, 1]")
        if poss != sorted(poss): raise ValueError("positions must be sorted ascending")
        return vals, poss
    if not any(has_at): return [float(p) for p in parts], None
    raise ValueError("either all or none of --sri-breaks must use @position")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--location", type=str, default=None)
    p.add_argument("--lat", type=float, default=None)
    p.add_argument("--lon", type=float, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--calibration", type=str, default=None)
    p.add_argument("--calibration-inline", type=str, default=None)
    p.add_argument("--help-calibration", action="store_true", default=False)
    p.add_argument("--radar-intensity", choices=["simple", "decoded"], default=None)
    p.add_argument("--sri-legend-region", type=str, default=None)
    p.add_argument("--sri-legend-orientation", choices=["horizontal", "vertical"], default=None)
    p.add_argument("--sri-legend-padding", type=int, default=None)
    p.add_argument("--sri-breaks", type=str, default=None)
    p.add_argument("--sri-thresholds", type=str, default=None)
    p.add_argument("--frame-interval-min", type=float, default=None)
    p.add_argument("--auto-frame-interval", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--timestamp-region", type=str, default=None)
    p.add_argument("--timestamp", dest="timestamp_enabled",
                   action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--timestamp-tz", type=str, default=None)
    p.add_argument("--timestamp-source", choices=["first", "last"], default=None)
    p.add_argument("--display-tz", type=str, default=None)
    p.add_argument("--no-cities", dest="cities", action="store_false", default=None)
    p.add_argument("--radius", type=float, default=None)
    p.add_argument("--weather-radius", type=float, default=None)
    p.add_argument("--archive", dest="archive_enabled",
                   action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--archive-radius", type=float, default=None)
    p.add_argument("--archive-min-pixels", type=int, default=None)
    p.add_argument("--poll", type=int, default=None)
    p.add_argument("--min-duration", type=int, default=None)
    p.add_argument("--min-size", type=int, default=None)
    p.add_argument("--approaching-radius", type=float, default=None)
    p.add_argument("--cluster-radius-weight", type=float, default=None)
    p.add_argument("--only-approach", type=float, default=None)
    p.add_argument("--threat-radius", type=float, default=None)
    p.add_argument("--threat-angle", type=float, default=None)
    p.add_argument("--threat-min-duration", type=int, default=None)
    p.add_argument("--threat-min-sri", type=float, default=None)
    p.add_argument("--view-rotation-auto", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--view-rotation", type=float, default=None)
    p.add_argument("--basemap", choices=["auto", "mapbox", "greyscale", "none"], default=None)
    p.add_argument("--basemap-saturation", type=float, default=None)
    p.add_argument("--basemap-brightness", type=float, default=None)
    p.add_argument("--basemap-contrast", type=float, default=None)
    p.add_argument("--mapbox-token", type=str, default=None)
    p.add_argument("--mapbox-style", type=str, default=None)
    p.add_argument("--basemap-size", type=int, default=None)
    p.add_argument("--pljusak", dest="pljusak_enabled",
                   action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--pljusak-url", type=str, default=None)
    p.add_argument("--pljusak-interval", type=int, default=None)
    p.add_argument("--station-confirm-radius", type=float, default=None)
    p.add_argument("--station-confirm-min-mm", type=float, default=None)
    p.add_argument("--station-confirm-max-age-min", type=float, default=None)
    p.add_argument("--station-confirm-fraction", type=float, default=None)
    p.add_argument("--station-confirm-min-count", type=int, default=None)
    p.add_argument("--mqtt", dest="mqtt_enabled",
                   action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--mqtt-host", type=str, default=None)
    p.add_argument("--mqtt-port", type=int, default=None)
    p.add_argument("--mqtt-tls", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--ha-discovery-prefix", type=str, default=None)
    p.add_argument("--ha-state-prefix", type=str, default=None)
    p.add_argument("--ha-device-id", type=str, default=None)
    p.add_argument("--ha-device-name", type=str, default=None)
    p.add_argument("--ha-sensors", type=str, default=None)
    p.add_argument("--mqtt-no-retain", action="store_true", default=None)
    p.add_argument("--ha-threat-topic", type=str, default=None)
    p.add_argument("--web", dest="web_enabled",
                   action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--web-port", type=int, default=None)
    p.add_argument("--web-scale", type=float, default=None)
    p.add_argument("--accuracy", dest="accuracy_enabled",
                   action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--once", action="store_true", default=False)
    p.add_argument("--dump-jsonl", type=str, default=None)
    p.add_argument("--settings-tag", type=str, default=None)

    args = p.parse_args()

    if args.help_calibration:
        print("Default calibration:\n" + DEFAULT_CALIBRATION_JSON); return

    data_dir = args.data_dir or os.environ.get("DATA_DIR") or "/data"
    try:
        os.makedirs(data_dir, exist_ok=True)
    except OSError as e:
        print(f"WARNING: cannot create {data_dir}: {e} — persistence disabled",
              file=sys.stderr)
        data_dir = None
    config_path = os.path.join(data_dir, "config.json") if data_dir else None

    existing = _load_json_file(config_path) if config_path else None
    runtime, static = _merge_config(args, existing)

    if config_path:
        now_iso = datetime.now(timezone.utc).isoformat()
        merged = {"version": 1,
                  "created_at": (existing or {}).get("created_at", now_iso),
                  "updated_at": now_iso,
                  "runtime": runtime, "static": static}
        _save_json_file(config_path, merged)
        print(f"Config {'created' if existing is None else 'loaded'} at {config_path}")

    with _RUNTIME_LOCK: _RUNTIME.update(runtime)
    _apply_static(static)
    _VIZ_STATE["config_path"] = config_path

    if static.get("location"):
        c = resolve_city(static["location"])
        if not c:
            print(f"ERROR: unknown city '{static['location']}'", file=sys.stderr); sys.exit(2)
        target_name, t_lat, t_lon = c
    elif static.get("lat") is not None and static.get("lon") is not None:
        target_name = f"{static['lat']:.4f},{static['lon']:.4f}"
        t_lat, t_lon = static["lat"], static["lon"]
    else:
        print("ERROR: no target — pass --location or --lat/--lon", file=sys.stderr); sys.exit(2)

    try:
        probe = fetch_image(GIF_URL); image_size = probe.size
    except Exception as e:
        print(f"ERROR probing image: {e}", file=sys.stderr); sys.exit(2)
    calib_inline = static.get("calibration_inline") or DEFAULT_CALIBRATION_JSON
    try:
        calib = load_calibration(static.get("calibration"), calib_inline, image_size=image_size)
    except Exception as e:
        print(f"ERROR loading calibration: {e}", file=sys.stderr); sys.exit(2)

    px, py = calib.latlon_to_image_pixel(t_lat, t_lon)
    check_lat, check_lon = calib.image_pixel_to_latlon(px, py)
    check_dist_km = haversine_km(t_lat, t_lon, check_lat, check_lon)
    if check_dist_km > 5.0:
        print(f"WARNING: target maps to ({check_lat:.3f},{check_lon:.3f}), "
              f"{check_dist_km:.1f} km from intended ({t_lat:.3f},{t_lon:.3f}) — "
              f"check calibration if unexpected", file=sys.stderr)

    sri_legend = None
    sri_thresholds = parse_floats(static.get("sri_thresholds")) or DEFAULT_SRI_THRESHOLDS
    if static.get("radar_intensity") == "decoded":
        if not static.get("sri_legend_region"):
            print("ERROR: --radar-intensity decoded requires --sri-legend-region",
                  file=sys.stderr); sys.exit(2)
        try:
            region = parse_region(static["sri_legend_region"])
        except ValueError as e:
            print(f"ERROR parsing sri-legend-region: {e}", file=sys.stderr); sys.exit(2)
        if static.get("sri_breaks"):
            try: break_values, break_positions = parse_sri_breaks(static["sri_breaks"])
            except ValueError as e:
                print(f"ERROR parsing sri-breaks: {e}", file=sys.stderr); sys.exit(2)
        else:
            print("ERROR: --radar-intensity decoded requires --sri-breaks.",
                  file=sys.stderr); sys.exit(2)
        sri_legend = SRILegend(region,
                               orientation=static.get("sri_legend_orientation") or "horizontal",
                               breaks=break_values,
                               break_positions=break_positions,
                               padding_px=static.get("sri_legend_padding") or 1)

    display_tz, display_tz_name = get_zoneinfo(static.get("display_tz"))
    timestamp_tz, _ = get_zoneinfo(static.get("timestamp_tz") or display_tz_name)

    timestamp_region = None
    if static.get("timestamp_enabled", True):
        try:
            timestamp_region = parse_region(static.get("timestamp_region") or DEFAULT_TIMESTAMP_REGION_STR)
        except ValueError as e:
            print(f"ERROR parsing timestamp-region: {e}", file=sys.stderr); sys.exit(2)
        if not HAVE_TESSERACT:
            print("WARNING: pytesseract not installed; timestamp OCR disabled.", file=sys.stderr)
            timestamp_region = None

    pljusak = PljusakClient(url=static.get("pljusak_url") or PLJUSAK_URL) \
        if static.get("pljusak_enabled", True) else None

    mqtt_host = static.get("mqtt_host") or os.environ.get("MQTT_HOST")
    mqtt_user = os.environ.get("MQTT_USER"); mqtt_pass = os.environ.get("MQTT_PASS")
    ha = None
    if mqtt_host and static.get("mqtt_enabled", True):
        if not HAVE_MQTT:
            print("WARNING: MQTT_HOST set but paho-mqtt not installed.", file=sys.stderr)
        else:
            sensors = parse_sensor_list(static.get("mqtt_sensors") or "all")
            device_id = static.get("mqtt_device_id") or \
                f"dhmz_{target_name.lower().replace(' ', '_').replace(',', '')}"
            mark_offline = (static.get("mqtt_mark_offline")
                            if static.get("mqtt_mark_offline") is not None
                            else not args.once)
            ha = HAPublisher(mqtt_host, static.get("mqtt_port") or 1883,
                             mqtt_user, mqtt_pass, bool(static.get("mqtt_tls")),
                             static.get("mqtt_discovery_prefix") or "homeassistant",
                             static.get("mqtt_state_prefix") or "dhmz_radar",
                             device_id,
                             static.get("mqtt_device_name") or "DHMZ Radar",
                             target_name, sensors, mark_offline,
                             not bool(static.get("mqtt_no_retain")),
                             static.get("mqtt_threat_topic"))
            ha.connect()

    acc_store = None
    if data_dir and static.get("accuracy_enabled", True):
        try:
            acc_store = AccuracyStore(data_dir)
            if not acc_store.enabled: acc_store = None
        except Exception as e:
            print(f"WARNING: accuracy init failed: {e}", file=sys.stderr)
    _VIZ_STATE["acc_store"] = acc_store

    archiver = None
    if data_dir:
        try:
            archiver = StormArchiver(data_dir, calib, t_lat, t_lon, target_name)
        except Exception as e:
            print(f"WARNING: archive init failed: {e}", file=sys.stderr)
    _VIZ_STATE["archiver"] = archiver

    web_enabled = bool(static.get("web_enabled", True))
    web_port = static.get("web_port") or 8080
    if web_enabled:
        if args.once:
            print("WARNING: --once with web port exits before browse.", file=sys.stderr)
        try:
            _start_web_server(web_port)
            print(f"Web server on port {web_port}")
        except OSError as e:
            print(f"WARNING: web server failed: {e}", file=sys.stderr)
            web_enabled = False

    run(target_name, t_lat, t_lon, calib, args.once,
        static.get("jsonl_dump"), static.get("frame_interval_min") or 5.0,
        bool(static.get("auto_frame_interval")), timestamp_region,
        static.get("timestamp_source") or "last", timestamp_tz,
        display_tz, display_tz_name, ha, pljusak,
        static.get("pljusak_interval") or 60,
        static.get("radar_intensity") or "simple",
        sri_legend, sri_thresholds,
        static.get("station_confirm_radius") or 15.0,
        static.get("station_confirm_min_mm") or 0.2,
        static.get("station_confirm_max_age_min") or 15.0,
        static.get("station_confirm_fraction") or 0.5,
        static.get("station_confirm_min_count") or 1,
        static.get("mqtt_threat_topic"), web_enabled, web_port,
        acc_store, archiver, data_dir,
        static.get("mapbox_token"), static.get("mapbox_style") or "mapbox/satellite-v9",
        static.get("basemap_size") or 640)


if __name__ == "__main__":
    main()
