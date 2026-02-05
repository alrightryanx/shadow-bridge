import os
import sys
import asyncio
import argparse
import json
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig

async def main():
    parser = argparse.ArgumentParser(description="Shadow Agent - Browser Use Wrapper")
    parser.add_argument("--task", required=True, help="The task for the agent to perform")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"], help="Model provider")
    parser.add_argument("--model", help="Specific model name (optional)")
    parser.add_argument("--api-key", required=True, help="API Key for the provider")
    parser.add_argument("--connect-url", help="CDP URL to connect to existing browser (e.g., http://localhost:9222)")
    
    args = parser.parse_args()

    # Set environment variable for the provider
    if args.provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = args.api_key
        llm = ChatAnthropic(model_name=args.model or "claude-3-5-sonnet-20240620")
    elif args.provider == "openai":
        os.environ["OPENAI_API_KEY"] = args.api_key
        llm = ChatOpenAI(model=args.model or "gpt-4o")
    else:
        print(json.dumps({"status": "error", "message": "Unsupported provider"}))
        return

    # Configure Browser
    browser = None
    if args.connect_url:
        # Direct connection to existing browser
        browser = Browser(
            config=BrowserConfig(
                chrome_instance_path=None, # Not needed when connecting
                cdp_url=args.connect_url,
            )
        )
    else:
        # Launch new headless browser
        browser = Browser(
            config=BrowserConfig(
                headless=True,
            )
        )

    try:
        agent = Agent(
            task=args.task,
            llm=llm,
            browser=browser,
        )

        history = await agent.run()
        
        # Extract the final result usually found in the last message or structured output
        # For this wrapper, we'll try to get the agent's final thought/action result
        result_message = history.final_result() 
        
        print(json.dumps({
            "status": "success", 
            "result": result_message,
            "steps": len(history.history)
        }))

    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
    finally:
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
