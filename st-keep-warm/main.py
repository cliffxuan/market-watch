import asyncio
import os

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

# Streamlit app URL from environment variable (or default)
STREAMLIT_URL = os.environ.get(
    "STREAMLIT_APP_URL", "https://market-watch.streamlit.app/"
)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--window-size=1920,1080",
            ],
        )
        context = await browser.new_context(viewport={"width": 1920, "height": 1080})
        page = await context.new_page()
        try:
            await page.goto(STREAMLIT_URL, timeout=15000)
            print(f"Opened {STREAMLIT_URL}")

            try:
                # Wait for the button and click it
                button = await page.wait_for_selector(
                    "button:has-text('Yes, get this app back up')",
                    timeout=15000,
                    state="visible",
                )
                print("Wake-up button found. Clicking...")
                await button.click()

                # Wait for button to disappear
                try:
                    await page.wait_for_selector(
                        "button:has-text('Yes, get this app back up')",
                        state="detached",
                        timeout=10000,
                    )
                    print("Button clicked and disappeared ✅ (app should be waking up)")
                except PlaywrightTimeoutError:
                    print(
                        "Button was clicked but did NOT disappear ❌ (possible failure)"
                    )
                    exit(1)

            except PlaywrightTimeoutError:
                # No button at all -> app is assumed awake
                print("No wake-up button found. Assuming app is already awake ✅")

        except Exception as e:
            print(f"Unexpected error: {e}")
            exit(1)
        finally:
            await context.close()
            await browser.close()
            print("Script finished.")


if __name__ == "__main__":
    asyncio.run(main())
