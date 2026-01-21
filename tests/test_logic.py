import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# --- FIX: Adjust PYTHONPATH ---
# 1. get cwd (tests/)
current_test_dir = os.path.dirname(os.path.abspath(__file__))
# 2. get project dir (DeepSese/)
project_root = os.path.dirname(current_test_dir)

# 3. Add the project root directory to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.sr import sr
from src.conf import conf
from src.t2i import t2i

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

    # FIX: Patch 路径也必须加上 'src.' 前缀，因为我们现在是从根目录加载的
    @patch('src.t2i.t2i.StableDiffusionXLPipeline')
    @patch('src.t2i.t2i.AutoPipelineForText2Image')
    @patch('src.t2i.t2i.AutoPipelineForImage2Image')
    def test_pipeline_flow(self, mock_img2img, mock_txt2img, mock_sdxl):
        """
        Mock the heavy model loading and verify the code flow.
        """
        print("\n[TEST] Simulating Pipeline Flow (Mocked)...")
        
        # Setup Mock objects to avoid loading 6GB files
        mock_pipe_instance = MagicMock()
        mock_sdxl.from_single_file.return_value = mock_pipe_instance
        
        # Mock image generation output
        # pipe() returns a result object, which has .images attribute
        mock_image = MagicMock()
        mock_pipe_result = MagicMock()
        mock_pipe_result.images = [mock_image]
        mock_pipe_instance.return_value = mock_pipe_result

        # Also need to mock the scheduler config access
        mock_pipe_instance.scheduler.config = {}

        # Run a "fake" task
        # This will execute all logic in run_task BUT will not touch GPU
        try:
            # We override path checks since we don't have model files in CI
            with patch('os.path.exists') as mock_exists:
                # We need exists to return True so it enters the logic
                mock_exists.return_value = True 
                
                # IMPORTANT: Mock makedirs so we don't actually create folders in CI
                with patch('os.makedirs'):
                    t2i.run_task(num_images=1)
                    
        except Exception as e:
            # Print full trace for debugging if it fails
            import traceback
            traceback.print_exc()
            self.fail(f"Pipeline crashed during mock run: {e}")
            
        print("[PASS] Pipeline flow executed successfully without GPU.")

if __name__ == '__main__':
    unittest.main()