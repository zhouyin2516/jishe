from fastapi import FastAPI
from routers import internal_storage
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="Cloud Storage Gateway",
    description="Internal microservice for federated learning to decouple Huawei OBS access via ECS Agency STS Tokens.",
    version="1.0.0"
)

# Optional CORS middleware if accessed by internal browser dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include standard routes
app.include_router(internal_storage.router)

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
