from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from catboost import CatBoostRegressor
import numpy as np
from fastapi_login import LoginManager
from fastapi.security import OAuth2PasswordRequestForm
from enum import Enum
from typing import Literal

app = FastAPI()

# Initialize model with different parameters
model = CatBoostRegressor(
    iterations=500,          # Increased iterations
    learning_rate=0.03,      # Reduced learning rate for better precision
    depth=6,
    loss_function='RMSE',    # Appropriate for regression
    verbose=False
)

# Create training data with more variation
np.random.seed(42)  # For reproducibility

# Generate more varied features
n_samples = 1000  # More samples
X = np.zeros((n_samples, 7))

# Generate features with realistic ranges and variation
X[:, 0] = np.random.uniform(10, 100, n_samples)  # ts: 10-100
X[:, 1] = np.random.uniform(30, 120, n_samples)  # tr: 30-120
X[:, 2] = np.random.uniform(5, 50, n_samples)    # x: 5-50
X[:, 3] = np.random.uniform(5, 50, n_samples)    # y: 5-50
X[:, 4] = np.random.uniform(0, 2, n_samples)     # e: 0-2 (will be rounded)
X[:, 5] = np.random.uniform(50, 200, n_samples)  # n: 50-200
X[:, 6] = np.random.uniform(0.8, 3, n_samples)   # v: 0.8-3 m/s (more realistic speed range)

# Round exit direction to 0, 1, or 2
X[:, 4] = np.round(X[:, 4])

# Generate target variable based on actual egress time calculations
y = np.zeros(n_samples)
for i in range(n_samples):
    distance = 0
    if X[i, 4] == 0:  # Short exit
        distance = np.minimum(X[i, 2], X[i, 3])  # min(x, y)
    elif X[i, 4] == 1:  # Long exit
        distance = np.maximum(X[i, 2], X[i, 3])  # max(x, y)
    else:  # Hypotenusal exit
        distance = np.sqrt(X[i, 2]**2 + X[i, 3]**2)  # sqrt(x^2 + y^2)
    
    # Calculate travel time without any additional factors
    y[i] = distance / X[i, 6]  # pure distance/speed calculation

# Train the model
model.fit(X, y)

# Save the model properly
model.save_model('catboost_model.cbm')  # Changed from .pkl to .cbm

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://soffit-egress.vercel.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication
SECRET = "your-secret-key"
manager = LoginManager(SECRET, '/login')

# Data models
class ExitDirection(str, Enum):
    SHORT = "0"
    LONG = "1"
    HYPOTENUSAL = "2"

class PredictionInput(BaseModel):
    ts: float  # Sensing Time (seconds)
    tr: float  # Response Time (seconds)
    x: float   # Length X (metre)
    y: float   # Length Y (metre)
    e: Literal["0", "1", "2"]  # Exit Direction (0=Short, 1=Long, 2=Hypotenusal)
    n: float   # Number of Personnel
    v: float   # Speed of personnel (m/s)

    class Config:
        schema_extra = {
            "example": {
                "ts": 50.0,
                "tr": 60.0,
                "x": 25.0,
                "y": 25.0,
                "e": "0",
                "n": 100.0,
                "v": 3.0
            }
        }

class PredictionOutput(BaseModel):
    tp: float
    rset: float

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Validate exit direction
        if input_data.e not in ["0", "1", "2"]:
            raise HTTPException(
                status_code=400, 
                detail="Exit direction must be '0' (Short), '1' (Long), or '2' (Hypotenusal)"
            )

        # Convert input data to numpy array
        features = np.array([[
            input_data.ts,
            input_data.tr,
            input_data.x,
            input_data.y,
            float(input_data.e),  # Convert string to float for the model
            input_data.n,
            input_data.v
        ]])

        # Make prediction
        tp = float(model.predict(features)[0])
        rset = tp + input_data.ts + input_data.tr

        # Handle NaN values
        if np.isnan(tp) or np.isnan(rset):
            return {
                "tp": str(0.0),
                "rset": str(0.0)
            }

        return {
            "tp": str(round(tp, 1)),
            "rset": str(round(rset, 1))
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)