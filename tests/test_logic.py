import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# --- FIX: Adjust PYTHONPATH ---
current_test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_test_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------

from src.sr import sr
from src.conf import conf
from src.t2i import t2i

# ==========================================
# 定义伪造的类 (Dummy Classes)
# 用于替换 diffusers 的真实类，解决 isinstance 报错问题
# ==========================================

class DummyOutput:
    """模拟 diffusers 的输出结果，包含 images 属性"""
    def __init__(self):
        # 返回一个伪造的图像对象
        self.images = [MagicMock()] 

class DummyPipeBase:
    """所有 Pipeline 的基类模拟"""
    def __init__(self, *args, **kwargs):
        self.scheduler = MagicMock()
        self.scheduler.config = {}
        self.vae = MagicMock()
    
    def enable_model_cpu_offload(self): pass
    def enable_slicing(self): pass
    def enable_tiling(self): pass
    
    def __call__(self, *args, **kwargs):
        # 模拟生成过程，返回包含图片的结果
        return DummyOutput()

class DummySDXL(DummyPipeBase):
    """模拟 StableDiffusionXLPipeline"""
    @classmethod
    def from_single_file(cls, *args, **kwargs):
        return cls()

class DummyT2I(DummyPipeBase):
    """模拟 AutoPipelineForText2Image"""
    @classmethod
    def from_pipe(cls, pipe):
        return cls()

class DummyI2I(DummyPipeBase):
    """模拟 AutoPipelineForImage2Image"""
    @classmethod
    def from_pipe(cls, pipe):
        return cls()

# ==========================================
# 测试逻辑
# ==========================================

class TestDeepSeseLogic(unittest.TestCase):

    def test_sr_coordinates(self):
        """
        Test if SR tiling coordinates are calculated correctly.
        """
        print("\n[TEST] Verifying SR Tiling Logic...")
        coords = sr.get_tile_coordinates()
        
        self.assertEqual(len(coords), 4)
        names = [c[0] for c in coords]
        self.assertListEqual(names, ["TL", "TR", "BL", "BR"])
        
        tr_x = coords[1][1]
        self.assertEqual(tr_x, 896)
        print("[PASS] SR Coordinates logic is valid.")

    # 使用 new=... 将源代码中的类替换为我们可以控制的 Dummy 类
    # 这样 isinstance(pipe, DummyT2I) 就是合法的语法
    @patch('src.t2i.t2i.StableDiffusionXLPipeline', new=DummySDXL)
    @patch('src.t2i.t2i.AutoPipelineForText2Image', new=DummyT2I)
    @patch('src.t2i.t2i.AutoPipelineForImage2Image', new=DummyI2I)
    def test_pipeline_flow(self):
        """
        Mock the heavy model loading and verify the code flow.
        """
        print("\n[TEST] Simulating Pipeline Flow (Mocked)...")
        
        # 注意：因为使用了 new=，这里不再需要接受 mock 参数
        
        try:
            # 模拟文件存在检查
            with patch('os.path.exists') as mock_exists:
                mock_exists.return_value = True 
                
                # 模拟文件夹创建，防止在 CI 环境报错
                with patch('os.makedirs'):
                    # 运行核心任务
                    t2i.run_task(num_images=1)
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"Pipeline crashed during mock run: {e}")
            
        print("[PASS] Pipeline flow executed successfully without GPU.")

if __name__ == '__main__':
    unittest.main()