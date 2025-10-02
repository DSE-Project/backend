from io import BytesIO
from playwright.sync_api import sync_playwright


def render_url_to_pdf_sync(url: str) -> bytes:
    print("🔹 Starting PDF render:", url)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="load")
        page.wait_for_timeout(2000)
        pdf_bytes = page.pdf(
            format="A4",
            print_background=True,
            margin={"top": "20mm", "bottom": "20mm", "left": "15mm", "right": "15mm"},
            scale=0.65,   
    
        )
        browser.close()
    print("🔹 Finished PDF render, size:", len(pdf_bytes))
    return pdf_bytes

