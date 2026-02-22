# Data

## realistic_flight_dataset.csv

10,000 synthetic flights generated with OpenAP aircraft performance models.

**Key columns:**
- `flight_id` — unique flight identifier
- `aircraft_type` — one of: A320, B737, B738, A321, B788, A359
- `origin_sector`, `dest_sector` — US-CONUS ARTCC sector pair
- `distance`, `distance_nm` — route distance in km and nautical miles
- `cruise_altitude_ft` — computed cruise altitude from OpenAP
- `fuel_consumption_kg` — computed fuel burn from OpenAP
- `origin_weather`, `dest_weather`, `route_weather` — severity score 0–10
- `airport_congestion` — 0–1 normalized congestion index
- `delay` — binary label (0: on-time, 1: delayed)

## realistic_rerouting_dataset.csv

Alternative route comparisons for each flight. Three alternatives per flight:

- **Route 1** — weather-optimized (flies around storm systems, slightly longer)
- **Route 2** — wind-optimized (altitude change to find favorable jet stream)
- **Route 3** — congestion-optimized (avoids high-traffic sectors)

Each route has: distance, time, weather score, fuel consumption, congestion, and altitude features.

## Data Generation

Regenerate with:
```bash
python src/train_realistic_models.py
```

> Data is synthetic and physics-grounded via OpenAP. Not sourced from real airline records.
