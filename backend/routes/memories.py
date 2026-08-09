"""
Memories routes — memory selection UI and forget/retain dataset creation.

GET  /api/memories        — list all training records
POST /api/memories/forget-set — create forget/retain split from selected IDs
POST /api/memories/load-builtin — load built-in synthetic dataset
"""

from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    MemoryRecord, MemoriesResponse,
    ForgetSetRequest, ForgetSetResponse, StatusResponse,
)
from backend.models.database import (
    get_training_records,
    get_training_records_by_ids,
    get_training_records_excluding_ids,
    save_training_records,
)
from training.dataset import DatasetManager
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["memories"])


@router.get("/memories", response_model=MemoriesResponse)
async def list_memories():
    """Return all training data records for the memory selection UI."""
    records = get_training_records()
    categories = sorted(set(r.get("category", "general") for r in records))

    return MemoriesResponse(
        records=[
            MemoryRecord(
                id=r["id"],
                question=r["question"],
                answer=r["answer"],
                category=r.get("category", "general"),
            )
            for r in records
        ],
        categories=categories,
        total=len(records),
    )


@router.post("/memories/forget-set", response_model=ForgetSetResponse)
async def create_forget_set(request: ForgetSetRequest):
    """Create forget and retain datasets from selected record IDs."""
    try:
        forget_records = get_training_records_by_ids(request.forget_ids)
        retain_records = get_training_records_excluding_ids(request.forget_ids)

        if not forget_records:
            raise HTTPException(status_code=404, detail="No records found for the given IDs.")

        # Build text arrays
        forget_texts = [
            f"Question: {r['question']}\nAnswer: {r['answer']}"
            for r in forget_records
        ]
        retain_texts = [
            f"Question: {r['question']}\nAnswer: {r['answer']}"
            for r in retain_records
        ]

        logger.info("Forget set: %d | Retain set: %d", len(forget_texts), len(retain_texts))

        return ForgetSetResponse(
            forget_count=len(forget_texts),
            retain_count=len(retain_texts),
            forget_texts=forget_texts,
            retain_texts=retain_texts,
            message=f"Created forget set ({len(forget_texts)} items) and retain set ({len(retain_texts)} items).",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create forget set: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memories/load-builtin", response_model=StatusResponse)
async def load_builtin_data():
    """Load built-in synthetic dataset into the database."""
    try:
        manager = DatasetManager()
        records = manager.load_builtin()
        train, val, test = manager.split_and_save(records)

        all_records = []
        for r in train:
            r["split"] = "train"
            all_records.append(r)
        for r in val:
            r["split"] = "val"
            all_records.append(r)
        for r in test:
            r["split"] = "test"
            all_records.append(r)

        save_training_records(all_records)

        return StatusResponse(
            status="ok",
            message=f"Loaded {len(records)} built-in records (train={len(train)}, val={len(val)}, test={len(test)}).",
        )
    except Exception as e:
        logger.error("Failed to load built-in data: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
