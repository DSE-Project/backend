import pytest
import warnings
from httpx import AsyncClient
from main import app
from httpx._transports.asgi import ASGITransport  # to call the app directly

@pytest.mark.asyncio
async def test_generate_report():
    url_to_render = "https://example.com"
    filename = "test_report.pdf"

    # Suppress DeprecationWarnings for this test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/generate-report?url={url_to_render}&filename={filename}")

            # Check HTTP status
            assert response.status_code == 200

            # Check content type
            assert response.headers["content-type"] == "application/pdf"

            # Check content disposition header
            assert f"filename={filename}" in response.headers["content-disposition"]

            # Optionally: check that response has some content
            assert len(response.content) > 0
