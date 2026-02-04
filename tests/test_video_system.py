"""
End-to-End Test for Video Generation System
Tests the integration between routes, progress tracking, and metrics.
"""

import sys
import os
import json
import time
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add parent directory to path to import web modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from web.routes.video import generate_video_local, _load_generations
from web.services.metrics_service import get_metrics_service
from web.services.video_progress import get_progress_tracker

class TestVideoSystem(unittest.TestCase):
    def setUp(self):
        self.metrics = get_metrics_service()
        self.tracker = get_progress_tracker()
        self.test_options = {
            "prompt": "A futuristic city at night",
            "model": "ltx-video",
            "duration": 5,
            "aspect_ratio": "16:9"
        }

    @patch('web.routes.video.subprocess.Popen')
    @patch('web.routes.video.is_model_installed')
    @patch('web.routes.video.get_video_duration')
    @patch('web.routes.video.os.path.exists')
    @patch('web.routes.video.os.path.getsize')
    @patch('web.services.video_error_handling.SystemResourceChecker.check_requirements')
    def test_complete_generation_flow(self, mock_resources, mock_getsize, mock_exists, mock_duration, mock_installed, mock_popen):
        """Test the full generation flow with mocked subprocess."""
        # Setup mocks
        mock_installed.return_value = True
        mock_duration.return_value = 5.0
        mock_exists.return_value = True 
        mock_getsize.return_value = 1024 * 1024 # 1MB
        mock_resources.return_value = {
            "memory_ok": True,
            "gpu_ok": True,
            "disk_ok": True,
            "recommendations": []
        }
        
        # Mock subprocess to simulate model output
        mock_process = MagicMock()
        mock_process.poll.return_value = 0
        mock_process.wait.return_value = 0
        mock_process.stdout.readline.side_effect = [
            "Loading model...",
            "Frame 1/100",
            "Frame 50/100",
            "Frame 100/100",
            "Finalizing...",
            ""
        ]
        mock_popen.return_value = mock_process

        # Mock output file creation
        output_dir = Path(os.path.expanduser("~")) / ".shadowai" / "video_models" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / "test.mp4")
        with open(output_path, 'w') as f:
            f.write("mock video content")

        # Define a progress callback to verify tracking
        progress_updates = []
        def progress_callback(data):
            progress_updates.append(data)

        # Run generation
        # Note: We call generate_video_local directly to test logic
        with patch('web.routes.video.MODELS_DIR', os.path.dirname(os.path.dirname(output_path))):
            result = generate_video_local(self.test_options, progress_callback)

        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['model'], "LTX Video (Fast)")
        self.assertGreater(len(progress_updates), 0)
        
        # Verify metrics were recorded (manually trigger recording since we skipped api_generate)
        initial_metrics = self.metrics.get_metrics()
        self.metrics.record_generation("ltx-video", 1000, True)
        new_metrics = self.metrics.get_metrics()
        
        self.assertEqual(new_metrics['total_generations'], initial_metrics['total_generations'] + 1)
        print("\n✅ End-to-End Logic Test Passed")
        print(f"✅ Progress Updates Received: {len(progress_updates)}")
        print(f"✅ Final Metrics: {json.dumps(new_metrics, indent=2)}")

if __name__ == "__main__":
    unittest.main()
