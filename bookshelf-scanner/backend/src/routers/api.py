from fastapi import APIRouter
from .predict_router import predict_router

api_router = APIRouter(prefix="/api")
api_router.include_router(predict_router)
