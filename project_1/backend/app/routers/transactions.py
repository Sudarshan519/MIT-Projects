"""Transaction scoring and management router."""

import csv
import io
import json
import asyncio
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.models.database import TransactionLog, AuditTrail, get_db
from app.models.schemas import (
    AlertStatus, ScoringResult, StatusUpdate, TransactionInput,
)
from app.routers.auth import get_current_user
from app.models.schemas import UserOut
from app.services.scoring import score_transaction, score_batch_transactions

router = APIRouter()


def _create_transaction_log(transaction: dict, result: ScoringResult) -> TransactionLog:
    """Create a TransactionLog object without committing."""
    return TransactionLog(
        time_val=transaction.get("time", 0),
        amount=transaction.get("amount", 0),
        v17=transaction.get("v17", 0),
        v14=transaction.get("v14", 0),
        v12=transaction.get("v12", 0),
        v10=transaction.get("v10", 0),
        v4=transaction.get("v4", 0),
        v11=transaction.get("v11", 0),
        logreg_probability=result.logreg_probability,
        j48_prediction=result.j48_prediction,
        j48_rule_path=result.j48_rule_path,
        decision=result.decision,
        risk_level=result.risk_level.value,
        fraud_probability=result.fraud_probability,
        cost_estimate_fn=result.cost_estimate_fn,
        cost_estimate_fp=result.cost_estimate_fp,
        status=result.status.value,
        scored_at=datetime.utcnow(),
    )


@router.post("/score", response_model=ScoringResult)
async def score_single(
    transaction: TransactionInput,
    db: Session = Depends(get_db),
    current_user: UserOut = Depends(get_current_user),
):
    """Score a single transaction for fraud probability."""
    tx_dict = transaction.model_dump()
    result = score_transaction(tx_dict)
    log = _create_transaction_log(tx_dict, result)
    db.add(log)
    db.commit()
    db.refresh(log)
    
    result.id = log.id
    result.scored_at = log.scored_at
    return result


@router.post("/score/batch")
async def score_batch(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: UserOut = Depends(get_current_user),
):
    """Score a batch of transactions from CSV upload."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    content = await file.read()
    text = content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))

    results = []
    logs = []
    tx_dicts = []
    
    for row in reader:
        tx_dict = {}
        for key, value in row.items():
            clean_key = key.strip().lower()
            try:
                tx_dict[clean_key] = float(value)
            except (ValueError, TypeError):
                tx_dict[clean_key] = 0.0
        tx_dicts.append(tx_dict)

    batch_results = score_batch_transactions(tx_dicts)

    for tx_dict, result in zip(tx_dicts, batch_results):
        log = _create_transaction_log(tx_dict, result)
        logs.append(log)
        
        result.id = 0  # Placeholder, since bulk insert does not return IDs immediately
        result.scored_at = log.scored_at
        results.append(result.model_dump())

    # Bulk save to DB for massive performance boost
    db.add_all(logs)
    db.commit()

    return {
        "total": len(results),
        "flagged": sum(1 for r in results if r["decision"] in ("flag", "block")),
        "results": results,
    }


@router.post("/score/batch/stream")
async def score_batch_stream(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: UserOut = Depends(get_current_user),
):
    """Score a batch of transactions and stream progress via SSE."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    content = await file.read()
    text = content.decode("utf-8")
    reader = list(csv.DictReader(io.StringIO(text)))
    total_rows = len(reader)
    
    async def event_generator():
        # Step 1: Parsing
        yield f"data: {json.dumps({'status': 'Parsing CSV Data...', 'progress': 10, 'total': total_rows})}\n\n"
        await asyncio.sleep(0.1)

        tx_dicts = []
        for row in reader:
            tx_dict = {}
            for key, value in row.items():
                clean_key = key.strip().lower()
                try:
                    tx_dict[clean_key] = float(value)
                except (ValueError, TypeError):
                    tx_dict[clean_key] = 0.0
            tx_dicts.append(tx_dict)
            
        # Step 2: ML Inference
        yield f"data: {json.dumps({'status': 'Running AI Inference (LogReg & J48)...', 'progress': 40, 'total': total_rows})}\n\n"
        await asyncio.sleep(0.1)

        batch_results = await asyncio.to_thread(score_batch_transactions, tx_dicts)

        # Step 3: Preparing Database Logs
        yield f"data: {json.dumps({'status': 'Preparing Database Logs...', 'progress': 70, 'total': total_rows})}\n\n"
        await asyncio.sleep(0.1)

        def prepare_logs():
            _logs = []
            _flagged = 0
            for tx_dict, result in zip(tx_dicts, batch_results):
                log = _create_transaction_log(tx_dict, result)
                _logs.append(log)
                if result.decision in ("flag", "block"):
                    _flagged += 1
            return _logs, _flagged

        logs, flagged = await asyncio.to_thread(prepare_logs)

        # Step 4: Bulk Database Commit
        yield f"data: {json.dumps({'status': 'Saving to Database...', 'progress': 90, 'total': total_rows})}\n\n"
        await asyncio.sleep(0.1)

        def save_db():
            db.add_all(logs)
            db.commit()
            
        await asyncio.to_thread(save_db)

        # Step 5: Complete
        yield f"data: {json.dumps({'status': 'Complete!', 'progress': 100, 'total': total_rows, 'flagged': flagged})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("")
async def list_transactions(
    status: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: UserOut = Depends(get_current_user),
):
    """List scored transactions with filtering and pagination."""
    query = db.query(TransactionLog)

    if status:
        query = query.filter(TransactionLog.status == status)
    if risk_level:
        query = query.filter(TransactionLog.risk_level == risk_level)

    total = query.count()
    items = (
        query.order_by(TransactionLog.scored_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
        "items": [
            {
                "id": item.id,
                "time": item.time_val,
                "amount": item.amount,
                "fraud_probability": item.fraud_probability,
                "decision": item.decision,
                "risk_level": item.risk_level,
                "j48_rule_path": item.j48_rule_path,
                "j48_prediction": item.j48_prediction,
                "status": item.status,
                "scored_at": item.scored_at.isoformat() if item.scored_at else None,
                "reviewer": item.reviewer,
                "v17": item.v17,
                "v14": item.v14,
                "v12": item.v12,
                "v10": item.v10,
                "logreg_probability": item.logreg_probability,
                "cost_estimate_fn": item.cost_estimate_fn,
                "cost_estimate_fp": item.cost_estimate_fp,
            }
            for item in items
        ],
    }


@router.get("/{transaction_id}")
async def get_transaction(
    transaction_id: int,
    db: Session = Depends(get_db),
    current_user: UserOut = Depends(get_current_user),
):
    """Get a single transaction with full details."""
    item = db.query(TransactionLog).filter(TransactionLog.id == transaction_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Transaction not found")

    # Get audit trail
    audits = (
        db.query(AuditTrail)
        .filter(AuditTrail.transaction_id == transaction_id)
        .order_by(AuditTrail.performed_at.desc())
        .all()
    )

    return {
        "id": item.id,
        "time": item.time_val,
        "amount": item.amount,
        "fraud_probability": item.fraud_probability,
        "logreg_probability": item.logreg_probability,
        "j48_prediction": item.j48_prediction,
        "j48_rule_path": item.j48_rule_path,
        "decision": item.decision,
        "risk_level": item.risk_level,
        "status": item.status,
        "cost_estimate_fn": item.cost_estimate_fn,
        "cost_estimate_fp": item.cost_estimate_fp,
        "scored_at": item.scored_at.isoformat() if item.scored_at else None,
        "reviewer": item.reviewer,
        "review_notes": item.review_notes,
        "reviewed_at": item.reviewed_at.isoformat() if item.reviewed_at else None,
        "key_features": {
            "V17": item.v17, "V14": item.v14, "V12": item.v12,
            "V10": item.v10, "V4": item.v4, "V11": item.v11,
            "Amount": item.amount,
        },
        "audit_trail": [
            {
                "action": a.action,
                "old_status": a.old_status,
                "new_status": a.new_status,
                "performed_by": a.performed_by,
                "notes": a.notes,
                "performed_at": a.performed_at.isoformat() if a.performed_at else None,
            }
            for a in audits
        ],
    }


@router.patch("/{transaction_id}/status")
async def update_status(
    transaction_id: int,
    update: StatusUpdate,
    db: Session = Depends(get_db),
    current_user: UserOut = Depends(get_current_user),
):
    """Update transaction alert status."""
    item = db.query(TransactionLog).filter(TransactionLog.id == transaction_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Transaction not found")

    old_status = item.status
    item.status = update.status.value
    item.reviewer = update.reviewer or current_user.username
    item.review_notes = update.notes
    item.reviewed_at = datetime.utcnow()

    # Create audit trail entry
    audit = AuditTrail(
        transaction_id=transaction_id,
        action="status_change",
        old_status=old_status,
        new_status=update.status.value,
        performed_by=current_user.username,
        notes=update.notes,
    )
    db.add(audit)
    db.commit()

    return {"message": "Status updated", "old_status": old_status, "new_status": update.status.value}
