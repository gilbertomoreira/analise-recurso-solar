"""Global constants. SI units unless stated otherwise."""

from pathlib import Path

SOLAR_CONSTANT_W_M2 = 1361.0  # Gsc [W/m2]
DEFAULT_ALBEDO = 0.2  # ground reflectance [-]
KWH_TO_MJ = 3.6  # 1 kWh = 3.6 MJ

NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal"
NASA_FILL_VALUE = -999.0
NASA_COMMUNITY = "RE"
HTTP_TIMEOUT_S = 60
HTTP_RETRIES = 4
CACHE_DIR = Path.home() / ".solarapp_cache"
CACHE_TTL_DAYS = 30
MAX_HOURLY_SPAN_DAYS = 366

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
USER_AGENT = "solarapp/0.1 (educational solar resource tool)"
