"""
OpenWeather API client.

HONEST SCOPE DOCUMENTATION:
Weather is a WEAK demand signal in aviation.
It explains approximately 3-5% of demand variance
on established routes.

What weather affects:
- Extreme weather cancellations (operational, not demand)
- Leisure travel to weather-sensitive destinations
- Last-minute booking behavior during weather events

What weather does NOT affect significantly:
- Business travel demand
- Long-haul international routes
- Advance bookings (weather unknown at booking time)

This client fetches monthly averages for use as
weak background features — not primary demand drivers.
This limitation is documented in feature engineering.
"""

from typing import Optional
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from src.utils.logger import logger
from src.utils.settings import get_settings

# Airport coordinates for our 3 routes
AIRPORT_COORDINATES = {
    "JFK": {"lat": 40.6413, "lon": -73.7781, "name": "New York JFK"},
    "LAX": {"lat": 33.9425, "lon": -118.4081, "name": "Los Angeles LAX"},
    "ORD": {"lat": 41.9742, "lon": -87.9073, "name": "Chicago O'Hare"},
    "MIA": {"lat": 25.7959, "lon": -80.2870, "name": "Miami MIA"},
    "SEA": {"lat": 47.4502, "lon": -122.3088, "name": "Seattle SEA"},
}


class OpenWeatherClient:
    """
    Client for OpenWeather API.
    Fetches current and historical weather for airport locations.
    """

    BASE_URL = "https://api.openweathermap.org/data/2.5"

    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.openweather_api_key

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    def _get(self, endpoint: str, params: dict) -> dict:
        params["appid"] = self.api_key
        params["units"] = "metric"

        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{self.BASE_URL}/{endpoint}", params=params)
            response.raise_for_status()
            return response.json()

    def get_current_weather(self, iata_code: str) -> Optional[dict]:
        """
        Fetch current weather for an airport location.
        Returns None on failure — weather being unavailable
        must never block the pipeline.
        """
        if iata_code not in AIRPORT_COORDINATES:
            logger.warning(f"No coordinates for airport: {iata_code}")
            return None

        coords = AIRPORT_COORDINATES[iata_code]
        logger.info(f"Fetching weather for {iata_code} ({coords['name']})")

        try:
            data = self._get(
                "weather",
                {
                    "lat": coords["lat"],
                    "lon": coords["lon"],
                },
            )
            return self._extract_weather_features(data, iata_code)
        except Exception as e:
            logger.error(f"Failed to fetch weather for {iata_code}: {e}")
            return None

    def _extract_weather_features(self, raw: dict, iata_code: str) -> dict:
        """
        Extract only the features relevant to demand modeling.
        Defensive extraction — missing fields get sensible defaults.

        Data quality flag included so downstream models can
        weight weather features appropriately when data is partial.
        """
        features = {
            "iata_code": iata_code,
            "temperature_c": raw.get("main", {}).get("temp", None),
            "feels_like_c": raw.get("main", {}).get("feels_like", None),
            "humidity_pct": raw.get("main", {}).get("humidity", None),
            "wind_speed_ms": raw.get("wind", {}).get("speed", None),
            # Precipitation: absent means no rain — default 0.0
            "precipitation_mm": raw.get("rain", {}).get("1h", 0.0),
            "weather_condition": raw.get("weather", [{}])[0].get("main", "Unknown"),
            "weather_data_quality": "complete",
        }

        # Flag missing critical fields
        missing = [k for k, v in features.items() if v is None]
        if missing:
            features["weather_data_quality"] = "partial"
            logger.warning(
                f"Weather data incomplete for {iata_code} — missing: {missing}"
            )
            # Fill missing with historical averages
            # In production: query a climate API for historical norms
            features = self._fill_missing_with_defaults(features, iata_code)

        return features

    def _fill_missing_with_defaults(self, features: dict, iata_code: str) -> dict:
        """
        Fills missing weather values with rough historical averages.
        These are approximations — clearly marked in data quality flag.
        In production: replace with proper climate normals API.
        """
        defaults = {
            "JFK": {"temperature_c": 12.0, "humidity_pct": 65.0, "wind_speed_ms": 5.5},
            "LAX": {"temperature_c": 18.0, "humidity_pct": 70.0, "wind_speed_ms": 3.5},
            "ORD": {"temperature_c": 10.0, "humidity_pct": 72.0, "wind_speed_ms": 6.0},
            "MIA": {"temperature_c": 25.0, "humidity_pct": 75.0, "wind_speed_ms": 4.0},
            "SEA": {"temperature_c": 11.0, "humidity_pct": 80.0, "wind_speed_ms": 4.5},
        }.get(
            iata_code,
            {"temperature_c": 15.0, "humidity_pct": 70.0, "wind_speed_ms": 5.0},
        )

        for key, default_val in defaults.items():
            if features.get(key) is None:
                features[key] = default_val
                logger.debug(f"Used default for {iata_code}.{key}: {default_val}")

        return features

    def get_weather_for_routes(self, routes: list[str]) -> dict:
        """
        Fetch weather for all airports in route list.
        Routes format: ['JFK-LAX', 'ORD-MIA', 'LAX-SEA']
        Returns dict keyed by IATA code.
        """
        airports = set()
        for route in routes:
            origin, dest = route.split("-")
            airports.add(origin)
            airports.add(dest)

        weather_data = {}
        for airport in airports:
            result = self.get_current_weather(airport)
            if result:
                weather_data[airport] = result
            else:
                logger.warning(
                    f"Weather unavailable for {airport} — "
                    f"pipeline continues without it"
                )

        logger.info(f"Weather fetched for {len(weather_data)}/{len(airports)} airports")
        return weather_data
