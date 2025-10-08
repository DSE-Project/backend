
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from main import app

import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from main import app  # adjust if it's from app.main instead

@pytest_asyncio.fixture
async def client():
    """Provide an AsyncClient that talks to the FastAPI app in memory."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

