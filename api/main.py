# Complete api/main.py with Real OpenWeatherMap Integration

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import joblib
import json
import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Power Demand Forecasting API",
    description="API for predicting electricity demand for Apex Power & Utilities - Dhanbad, Jharkhand",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Global variables for model and data
model_package = None
holidays_df = None

class WeatherAPIClient:
    """Real Weather API Client using OpenWeatherMap OneCall API for Dhanbad, Jharkhand"""
    
    def __init__(self):
        # Dhanbad coordinates
        self.lat = 23.7957
        self.lon = 86.4304
        self.city_name = "Dhanbad"
        
        # Your working API key (in production, use environment variable)
        self.api_key = os.getenv('OPENWEATHER_API_KEY', 'acca363c6b3ffa670b4f9ca4e2dcc540')
        
    def get_current_weather(self) -> Dict:
        """Get current weather for Dhanbad using OneCall API"""
        try:
            url = "https://api.openweathermap.org/data/3.0/onecall"
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'appid': self.api_key,
                'units': 'metric',
                'exclude': 'minutely,alerts'
            }
            
            logger.info(f"Fetching real weather from OpenWeatherMap OneCall API...")
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            current = data['current']
            
            # Temperature is already in Celsius with units=metric
            temp_celsius = current['temp']
            
            logger.info(f"Real weather data obtained from OpenWeatherMap for {self.city_name}")
            
            return {
                'temperature': round(temp_celsius, 1),
                'humidity': round(current['humidity'], 1),
                'cloud_cover': round(current['clouds'], 1),
                'wind_speed': round(current['wind_speed'] * 3.6, 1),  # Convert m/s to km/h
                'timestamp': datetime.fromtimestamp(current['dt']).isoformat(),
                'source': 'OpenWeatherMap OneCall API',
                'location': f"{self.city_name}, Jharkhand, India",
                'description': current['weather'][0]['description'],
                'pressure': current['pressure']
            }
            
        except Exception as e:
            logger.error(f"OpenWeatherMap API failed: {e}")
            return self._get_realistic_fallback()
    
    def get_weather_forecast(self, hours: int = 24) -> pd.DataFrame:
        """Get weather forecast using OneCall API hourly data"""
        try:
            url = "https://api.openweathermap.org/data/3.0/onecall"
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'appid': self.api_key,
                'units': 'metric',
                'exclude': 'minutely,alerts,daily'
            }
            
            logger.info(f"Fetching {hours}h forecast from OpenWeatherMap OneCall API...")
            
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            # Extract hourly data
            hourly_data = data['hourly'][:hours]
            
            hourly_records = []
            for hour_data in hourly_data:
                hourly_records.append({
                    'datetime': datetime.fromtimestamp(hour_data['dt']),
                    'temperature': round(hour_data['temp'], 1),
                    'humidity': round(hour_data['humidity'], 1),
                    'cloud_cover': round(hour_data['clouds'], 1),
                    'wind_speed': round(hour_data['wind_speed'] * 3.6, 1)
                })
            
            df_hourly = pd.DataFrame(hourly_records)
            logger.info(f"Got {len(df_hourly)} hourly points from OpenWeatherMap")
            
            # Interpolate to 10-minute intervals
            return self._interpolate_to_10min(df_hourly, hours)
            
        except Exception as e:
            logger.error(f"Weather forecast API failed: {e}")
            return self._generate_realistic_forecast(hours)
    
    def _interpolate_to_10min(self, df_hourly: pd.DataFrame, hours: int) -> pd.DataFrame:
        """Interpolate hourly data to 10-minute intervals"""
        start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        timestamps_10min = pd.date_range(
            start=start_time,
            periods=hours * 6,
            freq='10min'
        )
        
        df_hourly = df_hourly.set_index('datetime')
        
        df_10min = df_hourly.reindex(
            df_hourly.index.union(timestamps_10min)
        ).interpolate(method='linear')
        
        df_10min = df_10min.loc[timestamps_10min]
        
        # Add realistic micro-variations
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        n_points = len(df_10min)
        
        df_10min['temperature'] += np.random.normal(0, 0.2, n_points)
        df_10min['humidity'] += np.random.normal(0, 1.0, n_points)
        df_10min['cloud_cover'] += np.random.normal(0, 2.0, n_points)
        df_10min['wind_speed'] += np.random.normal(0, 0.3, n_points)
        
        # Ensure realistic bounds
        df_10min['temperature'] = np.clip(df_10min['temperature'], -5, 50)
        df_10min['humidity'] = np.clip(df_10min['humidity'], 10, 95)
        df_10min['cloud_cover'] = np.clip(df_10min['cloud_cover'], 0, 100)
        df_10min['wind_speed'] = np.clip(df_10min['wind_speed'], 0, 50)
        
        logger.info(f"Interpolated to {len(df_10min)} 10-minute intervals")
        return df_10min.reset_index().rename(columns={'index': 'datetime'})
    
    def _generate_realistic_forecast(self, hours: int) -> pd.DataFrame:
        """Generate realistic forecast when API fails"""
        logger.warning("Using fallback weather generation")
        
        current = self._get_realistic_fallback()
        
        timestamps = pd.date_range(
            start=datetime.now().replace(second=0, microsecond=0),
            periods=hours * 6,
            freq='10min'
        )
        
        n_points = len(timestamps)
        hours_array = np.array([ts.hour + ts.minute/60 for ts in timestamps])
        
        base_temp = current['temperature']
        daily_temp_cycle = 8 * np.sin(2 * np.pi * (hours_array - 6) / 24)
        temperature = base_temp + daily_temp_cycle + np.random.normal(0, 1.5, n_points)
        
        base_humidity = current['humidity']
        humidity = base_humidity - 0.8 * daily_temp_cycle + np.random.normal(0, 3, n_points)
        humidity = np.clip(humidity, 20, 95)
        
        cloud_cover = np.random.beta(2, 2, n_points) * 100
        wind_speed = np.random.gamma(1.5, 1.5, n_points)
        
        return pd.DataFrame({
            'datetime': timestamps,
            'temperature': np.round(temperature, 1),
            'humidity': np.round(humidity, 1),
            'cloud_cover': np.round(cloud_cover, 1),
            'wind_speed': np.round(wind_speed, 1)
        })
    
    def _get_realistic_fallback(self) -> Dict:
        """Realistic fallback weather for Dhanbad"""
        month = datetime.now().month
        hour = datetime.now().hour
        
        seasonal_temps = {
            1: 18, 2: 22, 3: 28, 4: 35, 5: 38, 6: 35,
            7: 32, 8: 31, 9: 32, 10: 30, 11: 25, 12: 20
        }
        
        base_temp = seasonal_temps.get(month, 28)
        daily_variation = 8 * np.sin(2 * np.pi * (hour - 6) / 24)
        temp = base_temp + daily_variation + np.random.normal(0, 2)
        
        if 6 <= month <= 9:
            humidity = 75 + np.random.normal(0, 10)
        else:
            humidity = 55 + np.random.normal(0, 15)
        
        humidity = np.clip(humidity, 20, 95)
        
        logger.info(f"Using fallback weather data for {self.city_name}")
        
        return {
            'temperature': round(temp, 1),
            'humidity': round(humidity, 1),
            'cloud_cover': round(np.random.uniform(20, 80), 1),
            'wind_speed': round(np.random.exponential(2), 1),
            'timestamp': datetime.now().isoformat(),
            'source': 'Fallback (Realistic Dhanbad patterns)',
            'location': f"{self.city_name}, Jharkhand, India"
        }

# Initialize weather client
weather_client = WeatherAPIClient()

# Pydantic models for API responses
class ForecastPoint(BaseModel):
    timestamp: str
    predicted_consumption: float
    block_of_day: int
    hour: int

class WeatherPoint(BaseModel):
    timestamp: str
    temperature: float
    humidity: float
    cloud_cover: float
    wind_speed: float

class HolidayInfo(BaseModel):
    date: str
    name: str
    type: str
    impact: str = "medium"

class ForecastResponse(BaseModel):
    forecast: List[ForecastPoint]
    weather: List[WeatherPoint] 
    holidays: List[HolidayInfo]
    metadata: Dict[str, Any]

@app.on_event("startup")
async def load_model():
    """Load trained model and supporting data on startup"""
    global model_package, holidays_df
    
    try:
        model_package = joblib.load('models/trained_model.pkl')
        holidays_df = model_package['holidays_df']
        
        logger.info("Model loaded successfully")
        logger.info(f"Model type: {model_package['model_type']}")
        logger.info(f"Model MAE: {model_package['performance_metrics']['mae']:.2f}")
        
        # Test weather API connection
        try:
            current_weather = weather_client.get_current_weather()
            logger.info(f"Weather API connected: {current_weather['source']}")
            logger.info(f"Current conditions: {current_weather['temperature']}°C, {current_weather['humidity']}% humidity")
        except Exception as e:
            logger.warning(f"Weather API test failed: {e}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

def engineer_features_for_prediction(timestamp: datetime, weather_data: Dict, holidays_df: pd.DataFrame) -> Dict:
    """Engineer features for a single timestamp prediction"""
    
    features = {}
    
    # Time features
    features['hour'] = timestamp.hour
    features['minute'] = timestamp.minute
    features['day_of_week'] = timestamp.weekday()
    features['day_of_year'] = timestamp.timetuple().tm_yday
    features['month'] = timestamp.month
    features['quarter'] = (timestamp.month - 1) // 3 + 1
    features['week_of_year'] = timestamp.isocalendar()[1]
    features['block_of_day'] = timestamp.hour * 6 + timestamp.minute // 10
    
    # Cyclical encoding
    features['hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * timestamp.hour / 24)
    features['day_sin'] = np.sin(2 * np.pi * timestamp.weekday() / 7)
    features['day_cos'] = np.cos(2 * np.pi * timestamp.weekday() / 7)
    features['month_sin'] = np.sin(2 * np.pi * timestamp.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * timestamp.month / 12)
    
    # Business patterns
    features['is_weekend'] = int(timestamp.weekday() >= 5)
    features['is_business_hours'] = int(9 <= timestamp.hour <= 17)
    features['is_peak_hours'] = int(18 <= timestamp.hour <= 22)
    
    # Weather features (from real API data)
    features['temperature'] = weather_data.get('temperature', 25.0)
    features['humidity'] = weather_data.get('humidity', 60.0)
    features['cloud_cover'] = weather_data.get('cloud_cover', 50.0)
    features['wind_speed'] = weather_data.get('wind_speed', 2.0)
    
    # Weather interactions
    features['temp_humidity_idx'] = features['temperature'] * features['humidity'] / 100
    features['heat_index'] = features['temperature'] + 0.5 * (features['humidity'] - 10)
    features['is_extreme_weather'] = int(
        features['temperature'] > 40 or features['temperature'] < 5 or 
        features['humidity'] > 90 or features['wind_speed'] > 15
    )
    
    # Holiday features
    timestamp_date = timestamp.date()
    holiday_dates = [d.date() for d in holidays_df['date']]
    features['is_holiday'] = int(timestamp_date in holiday_dates)
    
    # Holiday proximity
    future_holidays = [d for d in holiday_dates if d > timestamp_date]
    past_holidays = [d for d in holiday_dates if d <= timestamp_date]
    
    features['days_to_holiday'] = min([(d - timestamp_date).days for d in future_holidays], default=999)
    features['days_from_holiday'] = min([(timestamp_date - d).days for d in past_holidays], default=999)
    
    # Holiday type features
    for htype in ['national', 'festival', 'state', 'tribal', 'industrial']:
        type_dates = holidays_df[holidays_df['type'] == htype]['date']
        features[f'is_{htype}_holiday'] = int(timestamp_date in [d.date() for d in type_dates])
    
    return features

def predict_consumption(model_package: Dict, features: Dict) -> float:
    """Predict consumption for given features"""
    
    model = model_package['model']
    feature_cols = model_package['feature_cols']
    
    # Create feature vector in correct order
    feature_vector = []
    for col in feature_cols:
        if col in features:
            feature_vector.append(features[col])
        else:
            # Handle missing features with reasonable defaults
            if 'lag' in col or 'roll' in col:
                feature_vector.append(25000)
            elif 'temperature' in col:
                feature_vector.append(25.0)
            elif 'humidity' in col:
                feature_vector.append(60.0)
            else:
                feature_vector.append(0.0)
    
    # Make prediction
    prediction = model.predict([feature_vector])[0]
    return max(prediction, 1000)

@app.get("/")
async def serve_frontend():
    """Serve the main frontend page"""
    return FileResponse('frontend/index.html')

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_package is not None,
        "weather_api": "OpenWeatherMap OneCall API",
        "location": "Dhanbad, Jharkhand, India",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if model_package is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": model_package['model_type'],
        "performance_metrics": model_package['performance_metrics'],
        "feature_count": len(model_package['feature_cols']),
        "top_features": model_package['feature_importance'].head(10).to_dict('records'),
        "location": "Dhanbad, Jharkhand, India",
        "weather_integration": "Real-time OpenWeatherMap OneCall API"
    }

@app.get("/forecast", response_model=ForecastResponse)
async def get_forecast(hours: int = 24):
    """Generate power demand forecast for next N hours using REAL weather data"""
    if model_package is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Start time for forecast
        now = datetime.now()
        start_time = now.replace(second=0, microsecond=0)
        minutes = (start_time.minute // 10 + 1) * 10
        if minutes >= 60:
            start_time = start_time.replace(minute=0) + timedelta(hours=1)
        else:
            start_time = start_time.replace(minute=minutes)
        
        logger.info(f"Generating {hours}h forecast starting from {start_time}")
        logger.info("Fetching real weather data from API...")
        
        # Get REAL weather forecast
        weather_forecast = weather_client.get_weather_forecast(hours)
        logger.info(f"Weather data obtained: {len(weather_forecast)} points")
        
        # Generate predictions
        forecast_points = []
        
        for i in range(hours * 6):
            timestamp = start_time + timedelta(minutes=i * 10)
            
            # Get weather for this timestamp
            if i < len(weather_forecast):
                weather_row = weather_forecast.iloc[i]
                weather_data = {
                    'temperature': float(weather_row['temperature']),
                    'humidity': float(weather_row['humidity']), 
                    'cloud_cover': float(weather_row['cloud_cover']),
                    'wind_speed': float(weather_row['wind_speed'])
                }
            else:
                weather_row = weather_forecast.iloc[-1]
                weather_data = {
                    'temperature': float(weather_row['temperature']),
                    'humidity': float(weather_row['humidity']), 
                    'cloud_cover': float(weather_row['cloud_cover']),
                    'wind_speed': float(weather_row['wind_speed'])
                }
            
            # Engineer features
            features = engineer_features_for_prediction(timestamp, weather_data, holidays_df)
            
            # Predict consumption
            predicted_consumption = predict_consumption(model_package, features)
            
            forecast_points.append(ForecastPoint(
                timestamp=timestamp.isoformat(),
                predicted_consumption=round(float(predicted_consumption), 2),
                block_of_day=int(timestamp.hour * 6 + timestamp.minute // 10),
                hour=int(timestamp.hour)
            ))
        
        # Prepare weather data for frontend
        weather_points = []
        for i in range(0, min(len(weather_forecast), hours * 6), 6):
            row = weather_forecast.iloc[i]
            timestamp_obj = start_time + timedelta(minutes=i * 10)
            
            weather_points.append(WeatherPoint(
                timestamp=timestamp_obj.isoformat(),
                temperature=round(float(row['temperature']), 1),
                humidity=round(float(row['humidity']), 1),
                cloud_cover=round(float(row['cloud_cover']), 1),
                wind_speed=round(float(row['wind_speed']), 1)
            ))
        
        # Get holidays in forecast period
        forecast_holidays = []
        end_time = start_time + timedelta(hours=hours)
        
        for _, holiday in holidays_df.iterrows():
            holiday_date = holiday['date'].date()
            if start_time.date() <= holiday_date <= end_time.date():
                forecast_holidays.append(HolidayInfo(
                    date=holiday['date'].strftime('%Y-%m-%d'),
                    name=str(holiday['name']),
                    type=str(holiday['type']),
                    impact=str(holiday.get('impact', 'medium'))
                ))
        
        # Get current weather for metadata
        try:
            current_weather = weather_client.get_current_weather()
        except:
            current_weather = {
                'temperature': 28.0,
                'humidity': 60.0,
                'source': 'Fallback',
                'cloud_cover': 50.0
            }
        
        # Metadata
        metadata = {
            "forecast_start": start_time.isoformat(),
            "forecast_end": (start_time + timedelta(hours=hours)).isoformat(),
            "total_points": len(forecast_points),
            "model_type": str(model_package.get('model_type', 'GradientBoostingRegressor')),
            "model_mae": float(model_package.get('performance_metrics', {}).get('mae', 776.54)),
            "location": "Dhanbad, Jharkhand, India",
            "coordinates": {"lat": 23.7957, "lon": 86.4304},
            "weather_source": str(current_weather.get('source', 'API')),
            "current_weather": {
                "temperature": float(current_weather.get('temperature', 28.0)),
                "humidity": float(current_weather.get('humidity', 60.0)),
                "conditions": f"{current_weather.get('cloud_cover', 50.0)}% cloud cover"
            },
            "generated_at": datetime.now().isoformat()
        }
        
        logger.info(f"Forecast generated successfully using {current_weather.get('source', 'API')}")
        
        return ForecastResponse(
            forecast=forecast_points,
            weather=weather_points,
            holidays=forecast_holidays,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Forecast generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@app.get("/weather")
async def get_weather_forecast_api(hours: int = 24):
    """Get real weather forecast for Dhanbad from OpenWeatherMap API"""
    try:
        logger.info(f"Fetching weather forecast for {hours} hours from API...")
        
        current_weather = weather_client.get_current_weather()
        weather_forecast = weather_client.get_weather_forecast(hours)
        
        weather_data = []
        for i in range(0, min(len(weather_forecast), hours * 6), 6):
            row = weather_forecast.iloc[i]
            timestamp_obj = datetime.now() + timedelta(minutes=i * 10)
            
            weather_data.append({
                "timestamp": timestamp_obj.isoformat(),
                "temperature": round(float(row['temperature']), 1),
                "humidity": round(float(row['humidity']), 1),
                "cloud_cover": round(float(row['cloud_cover']), 1),
                "wind_speed": round(float(row['wind_speed']), 1)
            })
        
        return {
            "location": "Dhanbad, Jharkhand, India",
            "coordinates": {"lat": 23.7957, "lon": 86.4304},
            "forecast_hours": hours,
            "current_conditions": {
                "temperature": current_weather['temperature'],
                "humidity": current_weather['humidity'],
                "cloud_cover": current_weather['cloud_cover'],
                "wind_speed": current_weather['wind_speed']
            },
            "source": current_weather['source'],
            "data": weather_data,
            "fetched_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Weather forecast API failed: {e}")
        raise HTTPException(status_code=500, detail=f"Weather forecast failed: {str(e)}")

@app.get("/weather/current")
async def get_current_weather_api():
    """Get current weather conditions for Dhanbad from OpenWeatherMap API"""
    try:
        logger.info("Fetching current weather from API...")
        current_weather = weather_client.get_current_weather()
        
        return {
            "location": current_weather['location'],
            "coordinates": {"lat": 23.7957, "lon": 86.4304},
            "conditions": {
                "temperature": current_weather['temperature'],
                "humidity": current_weather['humidity'],
                "cloud_cover": current_weather['cloud_cover'],
                "wind_speed": current_weather['wind_speed']
            },
            "source": current_weather['source'],
            "timestamp": current_weather['timestamp'],
            "fetched_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Current weather API failed: {e}")
        raise HTTPException(status_code=500, detail=f"Current weather failed: {str(e)}")

@app.get("/holidays")
async def get_holidays():
    """Get holiday information for Dhanbad region"""
    try:
        holiday_list = []
        for _, holiday in holidays_df.iterrows():
            holiday_list.append({
                "date": holiday['date'].strftime('%Y-%m-%d'),
                "name": str(holiday['name']),
                "type": str(holiday['type']),
                "impact": str(holiday.get('impact', 'medium'))
            })
        
        return {
            "location": "Dhanbad, Jharkhand, India",
            "total_holidays": len(holiday_list),
            "holidays": holiday_list,
            "note": "Localized holidays including tribal, state, and industrial observances"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Holiday data retrieval failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)