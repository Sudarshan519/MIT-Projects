"""FastAPI application entry point."""

import warnings
warnings.filterwarnings("ignore")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS
from app.models.database import init_db
from app.routers import auth, dashboard, transactions


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    init_db()
    print("✅ Database initialized")

    # Try to load ML models
    from app.services.scoring import load_models
    load_models()
    print("✅ ML models loaded")

    yield
    # Shutdown
    print("👋 Shutting down")


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "INFO812 Group Assignment — Beyond Accuracy: Cost-Sensitive Evaluation "
        "of J48 and Logistic Regression for Credit Card Fraud Detection"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(transactions.router, prefix="/api/transactions", tags=["Transactions"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "project": "INFO812 Fraud Detection"}
