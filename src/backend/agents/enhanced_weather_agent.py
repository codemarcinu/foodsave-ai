import logging
import os
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field

from ..agents.base_agent import AgentResponse, EnhancedBaseAgent
from ..core.llm_client import llm_client

logger = logging.getLogger(__name__)

# Constants
WEATHER_ALERT_LEVELS = {
    "warning": 3,  # Highest severity
    "watch": 2,  # Moderate severity
    "advisory": 1,  # Low severity
}


class WeatherProvider(BaseModel):
    """Model for a weather data provider configuration"""

    name: str
    api_key_env_var: str
    base_url: str
    api_key: Optional[str] = None
    is_enabled: bool = True
    priority: int = 1  # Lower number = higher priority
    last_error: Optional[str] = None
    error_count: int = 0
    last_success: Optional[datetime] = None


class WeatherRequest(BaseModel):
    """Model for weather request parameters"""

    location: str
    days: int = Field(default=3, ge=1, le=7)
    include_alerts: bool = True
    model: str = "SpeakLeash/bielik-11b-v2.3-instruct:Q8_0"


class WeatherAlert(BaseModel):
    """Model for weather alerts"""

    event: str  # Type of alert (e.g., "Thunderstorm Warning")
    severity: int  # 1-3 scale, 3 being most severe
    headline: str  # Short description
    description: Optional[str] = None  # Detailed description
    effective: Optional[datetime] = None  # When alert starts
    expires: Optional[datetime] = None  # When alert ends
    areas: List[str] = []  # Affected areas


class WeatherData(BaseModel):
    """Model for standardized weather data across providers"""

    location: str
    current: Dict[str, Any]
    forecast: List[Dict[str, Any]]
    alerts: List[WeatherAlert] = []
    provider: str
    last_updated: datetime = Field(default_factory=datetime.now)


class EnhancedWeatherAgent(EnhancedBaseAgent[WeatherRequest]):
    """Enhanced weather agent with multi-provider fallback and alert handling"""

    def __init__(
        self,
        name: str = "EnhancedWeatherAgent",
        error_handler=None,
        fallback_manager=None,
        alert_service=None,
    ):
        super().__init__(
            name=name,
            error_handler=error_handler,
            fallback_manager=fallback_manager,
            alert_service=alert_service,
        )
        self.input_model = WeatherRequest
        self.providers: List[WeatherProvider] = self._init_providers()
        self.http_client = httpx.AsyncClient(timeout=10.0)
        self.cache: Dict[str, Tuple[WeatherData, datetime]] = {}
        self.cache_ttl = timedelta(minutes=15)  # Cache weather data for 15 minutes

    def _init_providers(self) -> List[WeatherProvider]:
        """Initialize weather data providers"""
        providers = [
            WeatherProvider(
                name="weatherapi",
                api_key_env_var="WEATHER_API_KEY",
                base_url="https://api.weatherapi.com/v1",
                priority=1,
            ),
            WeatherProvider(
                name="openweathermap",
                api_key_env_var="OPENWEATHER_API_KEY",
                base_url="https://api.openweathermap.org/data/2.5",
                priority=2,
            ),
        ]

        # Populate API keys from environment variables
        for provider in providers:
            provider.api_key = os.environ.get(provider.api_key_env_var)
            if not provider.api_key:
                provider.is_enabled = False
                logger.warning(f"No API key found for {provider.name}, disabling")

        return sorted(providers, key=lambda p: p.priority)

    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """Process a weather forecast request with provider fallback"""
        try:
            # Validate input
            validated_data = self._validate_input(input_data)
            query = validated_data.get("query", "")
            model = validated_data.get(
                "model", "SpeakLeash/bielik-11b-v2.3-instruct:Q8_0"
            )
            include_alerts = validated_data.get("include_alerts", True)

            # Extract location from query
            location = await self._extract_location(query, model)
            if not location:
                return self._handle_error(
                    ValueError("Nie mogłem rozpoznać lokalizacji w zapytaniu.")
                )

            # Check cache first
            cache_key = f"{location}_{include_alerts}"
            cached_data = self._get_from_cache(cache_key)
            if cached_data:
                logger.info(f"Using cached weather data for {location}")
                return self._format_response(cached_data, model)

            # Try each provider until success
            weather_data = None
            for provider in self.providers:
                if not provider.is_enabled:
                    continue

                try:
                    logger.info(
                        f"Fetching weather data for {location} from {provider.name}"
                    )
                    if provider.name == "weatherapi":
                        weather_data = await self._fetch_weatherapi(
                            location, validated_data.get("days", 3), include_alerts
                        )
                    elif provider.name == "openweathermap":
                        weather_data = await self._fetch_openweathermap(
                            location, validated_data.get("days", 3), include_alerts
                        )

                    if weather_data:
                        # Update provider stats
                        provider.last_success = datetime.now()
                        provider.last_error = None

                        # Cache successful result
                        self._add_to_cache(cache_key, weather_data)
                        break

                except Exception as e:
                    logger.error(f"Error with provider {provider.name}: {str(e)}")
                    provider.last_error = str(e)
                    provider.error_count += 1

            # Check if we got data from any provider
            if not weather_data:
                return self._handle_error(
                    ValueError(
                        f"Nie udało się pobrać prognozy pogody dla {location} z żadnego dostępnego źródła."
                    )
                )

            # Check for severe weather alerts
            has_severe_alerts = any(
                alert.severity >= 2 for alert in weather_data.alerts
            )

            # Format response with LLM
            return self._format_response(weather_data, model, has_severe_alerts)

        except Exception as e:
            return self._handle_error(e)

    async def _extract_location(self, query: str, model: str) -> str:
        """Extract location from user query using LLM"""
        prompt = (
            f"Przeanalizuj poniższe zapytanie i wyodrębnij nazwę lokalizacji (miasto, kraj itp.):\n\n"
            f"Zapytanie: '{query}'\n\n"
            f"Zwróć tylko nazwę lokalizacji, bez dodatkowego tekstu. Jeśli nie ma lokalizacji, zwróć 'Warszawa'."
        )

        try:
            response = await llm_client.chat(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Jesteś pomocnym asystentem, który ekstrahuje lokalizacje z tekstu.",
                    },
                    {"role": "user", "content": prompt},
                ],
                stream=False,
            )

            logging.info(f"LLM response for location extraction: {response}")

            # Check if response has expected structure
            if not response or not response.get("message"):
                logging.warning(
                    "Invalid LLM response format for location extraction, defaulting to Warsaw"
                )
                return "Warszawa"  # Default to Warsaw if extraction fails

            location = response["message"]["content"].strip()
            logging.info(f"Extracted location: '{location}'")
            return location

        except Exception as e:
            logger.error(f"Error extracting location: {e}")
            return "Warszawa"  # Default to Warsaw

    async def _fetch_weatherapi(
        self, location: str, days: int = 3, include_alerts: bool = True
    ) -> Optional[WeatherData]:
        """Fetch weather data from WeatherAPI.com"""
        provider = next((p for p in self.providers if p.name == "weatherapi"), None)
        if not provider or not provider.api_key:
            return None

        try:
            params = {
                "key": provider.api_key,
                "q": location,
                "days": days,
                "aqi": "no",
                "alerts": "yes" if include_alerts else "no",
            }

            response = await self.http_client.get(
                f"{provider.base_url}/forecast.json", params=params
            )
            response.raise_for_status()
            data = response.json()

            # Parse alerts if included
            alerts = []
            if include_alerts and "alerts" in data and "alert" in data["alerts"]:
                for alert_data in data["alerts"]["alert"]:
                    severity = 1  # Default low severity

                    # Determine severity based on alert category
                    category = alert_data.get("category", "").lower()
                    if "warning" in category:
                        severity = 3
                    elif "watch" in category:
                        severity = 2

                    alerts.append(
                        WeatherAlert(
                            event=alert_data.get("event", "Weather Alert"),
                            severity=severity,
                            headline=alert_data.get("headline", ""),
                            description=alert_data.get("desc", ""),
                            areas=alert_data.get("areas", "").split(";"),
                            effective=(
                                datetime.fromisoformat(
                                    alert_data.get("effective", "").replace(
                                        "Z", "+00:00"
                                    )
                                )
                                if "effective" in alert_data
                                else None
                            ),
                            expires=(
                                datetime.fromisoformat(
                                    alert_data.get("expires", "").replace("Z", "+00:00")
                                )
                                if "expires" in alert_data
                                else None
                            ),
                        )
                    )

            # Create standardized weather data
            return WeatherData(
                location=location,
                current=data.get("current", {}),
                forecast=data.get("forecast", {}).get("forecastday", []),
                alerts=alerts,
                provider="weatherapi",
            )

        except Exception as e:
            logger.error(f"Error fetching data from WeatherAPI: {e}")
            return None

    async def _fetch_openweathermap(
        self, location: str, days: int = 3, include_alerts: bool = True
    ) -> Optional[WeatherData]:
        """Fetch weather data from OpenWeatherMap"""
        provider = next((p for p in self.providers if p.name == "openweathermap"), None)
        if not provider or not provider.api_key:
            return None

        try:
            # First get current weather and coordinates
            current_params = {
                "q": location,
                "appid": provider.api_key,
                "units": "metric",
            }

            current_response = await self.http_client.get(
                f"{provider.base_url}/weather", params=current_params
            )
            current_response.raise_for_status()
            current_data = current_response.json()

            # Get coordinates for forecast
            lat = current_data.get("coord", {}).get("lat")
            lon = current_data.get("coord", {}).get("lon")

            if not lat or not lon:
                logger.error("Could not get coordinates from OpenWeatherMap")
                return None

            # Get forecast data
            forecast_params = {
                "lat": lat,
                "lon": lon,
                "appid": provider.api_key,
                "units": "metric",
                "exclude": "minutely,hourly",
            }

            forecast_response = await self.http_client.get(
                f"{provider.base_url}/onecall", params=forecast_params
            )
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()

            # Transform to standard format
            current = {
                "temp_c": current_data.get("main", {}).get("temp"),
                "condition": {
                    "text": current_data.get("weather", [{}])[0].get(
                        "description", "Unknown"
                    ),
                    "icon": current_data.get("weather", [{}])[0].get("icon", ""),
                },
                "wind_kph": current_data.get("wind", {}).get("speed")
                * 3.6,  # m/s to km/h
                "humidity": current_data.get("main", {}).get("humidity"),
            }

            # Process forecast days
            forecast = []
            if "daily" in forecast_data:
                for i, day in enumerate(forecast_data["daily"]):
                    if i >= days:
                        break

                    date_obj = datetime.fromtimestamp(day.get("dt", 0))
                    forecast.append(
                        {
                            "date": date_obj.strftime("%Y-%m-%d"),
                            "day": {
                                "maxtemp_c": day.get("temp", {}).get("max"),
                                "mintemp_c": day.get("temp", {}).get("min"),
                                "condition": {
                                    "text": day.get("weather", [{}])[0].get(
                                        "description", "Unknown"
                                    ),
                                    "icon": day.get("weather", [{}])[0].get("icon", ""),
                                },
                            },
                        }
                    )

            # Process alerts
            alerts = []
            if include_alerts and "alerts" in forecast_data:
                for alert_data in forecast_data.get("alerts", []):
                    # Determine severity
                    event = alert_data.get("event", "").lower()
                    severity = 1  # Default

                    for level, value in WEATHER_ALERT_LEVELS.items():
                        if level in event:
                            severity = value
                            break

                    alerts.append(
                        WeatherAlert(
                            event=alert_data.get("event", "Weather Alert"),
                            severity=severity,
                            headline=alert_data.get("event", ""),
                            description=alert_data.get("description", ""),
                            effective=datetime.fromtimestamp(
                                alert_data.get("start", 0)
                            ),
                            expires=datetime.fromtimestamp(alert_data.get("end", 0)),
                        )
                    )

            return WeatherData(
                location=location,
                current=current,
                forecast=forecast,
                alerts=alerts,
                provider="openweathermap",
            )

        except Exception as e:
            logger.error(f"Error fetching data from OpenWeatherMap: {e}")
            return None

    def _get_from_cache(self, cache_key: str) -> Optional[WeatherData]:
        """Get weather data from cache if available and fresh"""
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return data
        return None

    def _add_to_cache(self, cache_key: str, data: WeatherData) -> None:
        """Add weather data to cache"""
        self.cache[cache_key] = (data, datetime.now())

    def _format_response(
        self, weather_data: WeatherData, model: str, has_severe_alerts: bool = False
    ) -> AgentResponse:
        """Format weather data into a user-friendly response"""
        # Generate text representation of weather data
        current = weather_data.current
        forecast = weather_data.forecast

        weather_summary = (
            f"Lokalizacja: {weather_data.location}\n"
            f"Aktualna temperatura: {current.get('temp_c', 'N/A')}°C\n"
            f"Warunki: {current.get('condition', {}).get('text', 'N/A')}\n"
            f"Wiatr: {current.get('wind_kph', 'N/A')} km/h\n"
            f"Wilgotność: {current.get('humidity', 'N/A')}%\n\n"
        )

        if forecast:
            weather_summary += "Prognoza na najbliższe dni:\n"
            for day in forecast:
                weather_summary += (
                    f"- {day.get('date', 'N/A')}: "
                    f"{day.get('day', {}).get('mintemp_c', 'N/A')}°C to "
                    f"{day.get('day', {}).get('maxtemp_c', 'N/A')}°C, "
                    f"{day.get('day', {}).get('condition', {}).get('text', 'N/A')}\n"
                )

        # Add alerts if present
        if weather_data.alerts:
            weather_summary += "\nALERTY POGODOWE:\n"
            for alert in weather_data.alerts:
                severity_text = (
                    "WYSOKI"
                    if alert.severity >= 3
                    else "ŚREDNI" if alert.severity == 2 else "NISKI"
                )
                weather_summary += (
                    f"- {alert.event} (Poziom: {severity_text}): {alert.headline}\n"
                )

        # Create LLM prompt
        prompt_prefix = ""
        if has_severe_alerts:
            prompt_prefix = (
                "UWAGA: Występują poważne alerty pogodowe. "
                "Zaznacz to wyraźnie na początku prognozy i podkreśl wagę ostrzeżeń. "
            )

        prompt = (
            f"{prompt_prefix}Na podstawie poniższych danych pogodowych, utwórz przyjazną i naturalnie brzmiącą "
            f"prognozę pogody w języku polskim:\n\n{weather_summary}"
        )

        # Generate streaming response
        response_stream = self._stream_weather_response(model, prompt)

        # Return formatted response
        return AgentResponse(
            success=True,
            data={
                "location": weather_data.location,
                "current": weather_data.current,
                "forecast": weather_data.forecast,
                "alerts": [alert.dict() for alert in weather_data.alerts],
                "provider": weather_data.provider,
            },
            text_stream=response_stream,
            message=f"Pogoda dla {weather_data.location}",
        )

    async def _stream_weather_response(
        self, model: str, prompt: str
    ) -> AsyncGenerator[str, None]:
        """Stream weather response from LLM"""
        messages = [
            {"role": "system", "content": "Jesteś pomocnym asystentem pogodowym."},
            {"role": "user", "content": prompt},
        ]

        async for chunk in self._stream_llm_response(model, messages):
            yield chunk

    async def close(self) -> None:
        """Close HTTP client on shutdown"""
        await self.http_client.aclose()
