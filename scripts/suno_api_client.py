"""
Suno API Client - Suno AI Music Generation API
Handles API calls, song generation, and audio download
"""

import asyncio
import json
import logging
import os
from typing import Optional, Dict, Any
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)


class SunoAPIClient:
    """Client for Suno AI Music Generation API"""

    def __init__(self, api_key: str):
        """
        Initialize Suno API client

        Args:
            api_key: Suno API key
        """
        self.api_key = api_key
        self.base_url = "https://api.suno.ai/v1"
        self.session = aiohttp.ClientSession()

        logger.info(f"SunoAPIClient initialized with API key")

    async def close(self):
        """Close HTTP session"""
        await self.session.close()

    async def generate_song(
        self,
        prompt: str,
        duration: int = 120,
        instrumental: bool = False,
        tags: Optional[list] = None,
    ) -> str:
        """
        Generate a song using Suno API

        Args:
            prompt: Description of the song
            duration: Duration in seconds
            instrumental: Generate instrumental only
            tags: Optional list of genre/style tags

        Returns:
            Song ID (string)
        """
        url = f"{self.base_url}/generate"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {"prompt": prompt, "duration": duration, "instrumental": instrumental}

        if tags:
            payload["tags"] = tags

        logger.info(f"Generating song with Suno API: {prompt}")

        try:
            async with self.session.post(
                url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()

                    if "song_id" in data:
                        song_id = data["song_id"]
                        logger.info(f"Song generation started: {song_id}")
                        return song_id
                    elif "id" in data:
                        song_id = data["id"]
                        logger.info(f"Song generation started: {song_id}")
                        return song_id
                    else:
                        logger.error(f"Unexpected response: {data}")
                        raise Exception("No song ID in response")
                else:
                    error_text = await response.text()
                    logger.error(f"Suno API error {response.status}: {error_text}")
                    raise Exception(f"Suno API request failed: {response.status}")

        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error: {str(e)}")
            raise Exception(f"HTTP client error: {str(e)}")
        except Exception as e:
            logger.error(f"Sono API error: {str(e)}")
            raise

    async def get_song_status(self, song_id: str) -> Dict[str, Any]:
        """
        Get the status of a song generation

        Args:
            song_id: Song ID from generate_song

        Returns:
            Dict with status information
        """
        url = f"{self.base_url}/songs/{song_id}"

        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Error getting song status {response.status}: {error_text}"
                    )
                    raise Exception(f"Failed to get song status: {response.status}")

        except Exception as e:
            logger.error(f"Error getting song status: {str(e)}")
            raise

    async def download_song(
        self, song_id: str, output_dir: Optional[str] = None
    ) -> str:
        """
        Download generated song audio

        Args:
            song_id: Song ID to download
            output_dir: Optional output directory

        Returns:
            Path to downloaded audio file
        """
        url = f"{self.base_url}/songs/{song_id}/download"

        headers = {"Authorization": f"Bearer {self.api_key}"}

        if not output_dir:
            output_dir = os.path.expanduser("~/shadow_music")
        else:
            output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir) / f"{song_id}.wav"

        logger.info(f"Downloading song {song_id} to {output_path}")

        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    # Stream download
                    with open(output_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(1024):
                            f.write(chunk)

                    logger.info(f"Song downloaded successfully: {output_path}")
                    return str(output_path)
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Error downloading song {response.status}: {error_text}"
                    )
                    raise Exception(f"Failed to download song: {response.status}")

        except Exception as e:
            logger.error(f"Error downloading song: {str(e)}")
            raise

    async def generate_and_download(
        self,
        prompt: str,
        duration: int = 120,
        instrumental: bool = False,
        tags: Optional[list] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Generate song and wait for completion, then download

        Args:
            prompt: Description of the song
            duration: Duration in seconds
            instrumental: Generate instrumental only
            tags: Optional list of genre/style tags
            output_dir: Optional output directory

        Returns:
            Path to downloaded audio file
        """
        # Generate song
        song_id = await self.generate_song(
            prompt=prompt, duration=duration, instrumental=instrumental, tags=tags
        )

        # Poll for completion
        max_attempts = 60  # 5 minutes max
        attempt = 0

        while attempt < max_attempts:
            await asyncio.sleep(5)  # Wait 5 seconds between checks

            status = await self.get_song_status(song_id)

            if status.get("status") == "complete":
                logger.info(f"Song generation complete: {song_id}")
                break
            elif status.get("status") == "failed":
                error_msg = status.get("error", "Unknown error")
                logger.error(f"Song generation failed: {error_msg}")
                raise Exception(f"Song generation failed: {error_msg}")

            attempt += 1
            logger.info(f"Waiting for completion... ({attempt}/{max_attempts})")

        if attempt >= max_attempts:
            raise Exception("Song generation timed out")

        # Download song
        return await self.download_song(song_id, output_dir)


# Convenience function for standalone use
async def main():
    """Main function for standalone script execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate music using Suno AI API")

    parser.add_argument("--api-key", required=True, help="Suno API key")

    parser.add_argument("--prompt", required=True, help="Song description/prompt")

    parser.add_argument(
        "--duration", type=int, default=120, help="Duration in seconds (default: 120)"
    )

    parser.add_argument(
        "--instrumental", action="store_true", help="Generate instrumental only"
    )

    parser.add_argument("--tags", nargs="+", help="Genre/style tags")

    parser.add_argument(
        "--output", default=None, help="Output directory for downloaded audio"
    )

    args = parser.parse_args()

    client = SunoAPIClient(args.api_key)

    try:
        audio_path = await client.generate_and_download(
            prompt=args.prompt,
            duration=args.duration,
            instrumental=args.instrumental,
            tags=args.tags,
            output_dir=args.output,
        )

        print(f"Success! Song downloaded to: {audio_path}")
        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    finally:
        await client.close()


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
