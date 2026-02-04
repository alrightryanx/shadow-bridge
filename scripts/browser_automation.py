import sys
import json
import asyncio
import os
import base64
from pathlib import Path

try:
    from playwright.async_api import async_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

class BrowserAutomation:
    def __init__(self):
        self.browser = None
        self.context = None
        self.page = None

    async def setup(self):
        """Install playwright and browsers if missing."""
        if not HAS_PLAYWRIGHT:
            print("Installing Playwright...")
            os.system(f"{sys.executable} -m pip install playwright")
            os.system(f"{sys.executable} -m playwright install chromium")
            return {"success": True, "message": "Playwright installed. Please restart the command."}
        
        # Try to launch to verify
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                await browser.close()
            return {"success": True, "message": "Browser environment ready"}
        except Exception as e:
            print(f"Browser setup failed: {e}")
            os.system(f"{sys.executable} -m playwright install chromium")
            return {"success": True, "message": "Attempted to fix browser installation."}

    async def search_and_summarize(self, query):
        """Search Google and summarize the first few results."""
        if not HAS_PLAYWRIGHT:
            return {"success": False, "error": "Playwright not installed"}

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            try:
                # Search Google
                await page.goto(f"https://www.google.com/search?q={query}")
                
                # Get the first few results (titles and snippets)
                results = []
                # Google search result selectors (might change, but usually these work)
                items = await page.query_selector_all("div.g")
                for item in items[:3]:
                    title_el = await item.query_selector("h3")
                    snippet_el = await item.query_selector("div.VwiC3b")
                    
                    if title_el and snippet_el:
                        results.append({
                            "title": await title_el.inner_text(),
                            "snippet": await snippet_el.inner_text()
                        })

                if not results:
                    return {"success": False, "error": "No results found"}

                summary = f"Search results for '{query}':\n"
                for i, r in enumerate(results, 1):
                    summary += f"{i}. {r['title']}: {r['snippet']}\n"

                return {
                    "success": True, 
                    "summary": summary,
                    "results": results
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
            finally:
                await browser.close()

    async def get_page_content(self, url):
        """Navigate to URL and return text content."""
        if not HAS_PLAYWRIGHT:
            return {"success": False, "error": "Playwright not installed"}

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                await page.goto(url, wait_until="networkidle")
                content = await page.inner_text("body")
                return {"success": True, "content": content[:5000]} # Limit for now
            except Exception as e:
                return {"success": False, "error": str(e)}
            finally:
                await browser.close()

def get_browser_service():
    return BrowserAutomation()

if __name__ == "__main__":
    # Internal CLI for debugging/testing
    import argparse
    parser = argparse.ArgumentParser(description="Browser Automation Service")
    parser.add_argument("command", choices=["search", "summarize", "setup"])
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--url", help="URL to summarize")
    
    args = parser.parse_args()
    service = get_browser_service()
    
    loop = asyncio.get_event_loop()
    if args.command == "setup":
        print(json.dumps(loop.run_until_complete(service.setup())))
    elif args.command == "search":
        print(json.dumps(loop.run_until_complete(service.search_and_summarize(args.query))))
    elif args.command == "summarize":
        print(json.dumps(loop.run_until_complete(service.get_page_content(args.url))))
