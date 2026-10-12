"""SQLAlchemy database models and connection."""

from datetime import datetime

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TransactionLog(Base):
    """Log of all scored transactions."""
    __tablename__ = "transaction_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    # Transaction data
    time_val = Column(Float)
    amount = Column(Float)
    # Key features for display
    v17 = Column(Float)
    v14 = Column(Float)
    v12 = Column(Float)
    v10 = Column(Float)
    v4 = Column(Float)
    v11 = Column(Float)
    # Scoring results
    logreg_probability = Column(Float)
    j48_prediction = Column(Integer)
    j48_rule_path = Column(Text, nullable=True)
    decision = Column(String(20))  # allow, flag, block
    risk_level = Column(String(20))
    fraud_probability = Column(Float)
    # Cost
    cost_estimate_fn = Column(Float, default=0.0)
    cost_estimate_fp = Column(Float, default=0.0)
    # Status tracking
    status = Column(String(20), default="new")
    reviewer = Column(String(100), nullable=True)
    review_notes = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    # Actual label (if known, for evaluation)
    actual_label = Column(Integer, nullable=True)
    # Timestamps
    scored_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditTrail(Base):
    """Audit trail for status changes."""
    __tablename__ = "audit_trail"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    transaction_id = Column(Integer, index=True)
    action = Column(String(50))  # status_change, review, etc.
    old_status = Column(String(20), nullable=True)
    new_status = Column(String(20))
    performed_by = Column(String(100))
    notes = Column(Text, nullable=True)
    performed_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency to get DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
