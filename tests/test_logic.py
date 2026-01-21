import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Add src to sys.path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sr import sr
from conf import conf

class TestDeepSeseLogic(unittest.TestCase):

    def test_sr_coordinates(self):
        """
        Test if SR tiling coordinates are calculated correctly.
        This runs on CPU and needs no model.
        """
        print("\n[TEST] Verifying SR Tiling Logic...")
        coords = sr.get_tile_coordinates()
        
        # Check if we have 4 tiles
        self.assertEqual(len(coords), 4)
        
        # Check Tile names
        names = [c[0] for c in coords]
        self.assertListEqual(names, ["TL", "TR", "BL", "BR"])
        
        # Check overlap logic (TR x position should be 1920 - 1024 = 896)
        tr_x = coords[1][1]
        self.assertEqual(tr_x, 896)
        print("[PASS] SR Coordinates logic is valid.")

    @patch('t2i.t2i.StableDiffusionXLPipeline')
    @patch('t2i.t2i.AutoPipelineForText2Image')
    @patch('t2i.t2i.AutoPipelineForImage2Image')
    def test_pipeline_flow(self, mock_img2img, mock_txt2img, mock_sdxl):
        """
        Mock the heavy model loading and verify the code flow.
        """
        print("\n[TEST] Simulating Pipeline Flow (Mocked)...")
        
        # Setup Mock objects to avoid loading 6GB files
        mock_pipe_instance = MagicMock()
        mock_sdxl.from_single_file.return_value = mock_pipe_instance
        
        # Mock image generation output
        mock_image = MagicMock()
        # Mocking the return structure: pipe().images[0]
        mock_pipe_instance.return_value.images = [mock_image] 
        
        # Import main logic
        from t2i import t2i
        
        # Run a "fake" task
        # This will execute all logic in run_task BUT will not touch GPU
        try:
            # We override path checks since we don't have model files in CI
            with patch('os.path.exists') as mock_exists:
                mock_exists.return_value = True 
                t2i.run_task(num_images=1)
        except Exception as e:
            self.fail(f"Pipeline crashed during mock run: {e}")
            
        print("[PASS] Pipeline flow executed successfully without GPU.")

if __name__ == '__main__':
    unittest.main()