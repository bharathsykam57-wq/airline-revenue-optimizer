"""
Aviationstack API client.

HONEST SCOPE DOCUMENTATION:
Aviationstack free tier provides:
- Flight schedules and routes
- Flight status and delays
- Airport and airline metadata

What it does NOT provide (relevant to revenue management):
- Load factors or seat occupancy
- Booking data or demand curves
- Fare information

This client is used for route metadata and schedule features only.
It is NOT a demand data source.
"""

import time
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

# Aviationstack free tier: 100 calls/month
# We cache aggressively to stay within limits
RATE_LIMIT_DELAY = 2.0  # seconds between calls


class AviationstackClient:
    """
    Client for Aviationstack API.
    Handles retries, rate limiting, and response validation.
    """

    BASE_URL = "https://api.aviationstack.com/v1"

    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.aviationstack_api_key
        self._last_call_time: float = 0.0

    def _respect_rate_limit(self) -> None:
        """
        Enforces minimum delay between API calls.
        Free tier has strict rate limits — violating them
        results in 429 errors and temporary blocks.
        """
        elapsed = time.time() - self._last_call_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_call_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    def _get(self, endpoint: str, params: dict) -> dict:
        """
        Core GET request with retry logic.
        Exponential backoff: 2s, 4s, 8s between retries.
        """
        self._respect_rate_limit()
        params["access_key"] = self.api_key

        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{self.BASE_URL}/{endpoint}", params=params)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logger.error(f"Aviationstack API error: {data['error']}")
                raise ValueError(f"API error: {data['error']}")

            return data

    def get_routes(self, origin: str, destination: str) -> Optional[dict]:
        """
        Fetch route metadata for a given origin-destination pair.

        Returns schedule and airline information.
        Does NOT return demand, load factors, or pricing.
        """
        logger.info(f"Fetching route metadata: {origin} -> {destination}")
        try:
            data = self._get(
                "flights",
                {
                    "dep_iata": origin,
                    "arr_iata": destination,
                    "limit": 10,
                },
            )
            logger.info(
                f"Retrieved {len(data.get('data', []))} flights for {origin}-{destination}"
            )
            return data
        except Exception as e:
            logger.error(f"Failed to fetch routes {origin}-{destination}: {e}")
            return None

    def get_airport_info(self, iata_code: str) -> Optional[dict]:
        """
        Fetch airport metadata — timezone, country, coordinates.
        Used as route features, not demand signals.
        """
        logger.info(f"Fetching airport info: {iata_code}")
        try:
            data = self._get("airports", {"iata_code": iata_code})
            return data
        except Exception as e:
            logger.error(f"Failed to fetch airport {iata_code}: {e}")
            return None
