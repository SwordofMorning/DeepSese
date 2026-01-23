# src/main.py

import argparse
import sys
import os

def setup_env():
    """
    Ensure the project root is in sys.path so that absolute imports
    like 'from src.conf import ...' work correctly.
    """
    # Get the directory where main.py is located (e.g., .../DeepSese/src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the parent directory (Project Root, e.g., .../DeepSese)
    project_root = os.path.dirname(current_dir)
    
    # Add project root to sys.path if not present
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Apply environment setup immediately
setup_env()

def main():
    # Initialize Argument Parser
    parser = argparse.ArgumentParser(description="DeepSese Image Generation Framework")
    
    # Define arguments
    parser.add_argument("--t2i", action="store_true", help="Run Text-to-Image generation task")
    parser.add_argument("--nums", type=int, default=None, help="Number of images to generate (T2I)")
    
    parser.add_argument("--sr", action="store_true", help="Run Super-Resolution task")
    parser.add_argument("--file", type=str, help="Single file path for SR")
    parser.add_argument("--folder", type=str, help="Folder path for SR")

    args = parser.parse_args()

    # Dispatch Logic
    # Note: Now we use absolute imports (src.xxx) to be consistent and safe
    if args.t2i:
        print("[MAIN] Mode selected: Text-to-Image (T2I)")
        try:
            from src.t2i import t2i
            t2i.run_task(num_images=args.nums)
        except ImportError as e:
            print(f"[ERROR] Failed to import T2I module: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    elif args.sr:
        print("[MAIN] Mode selected: Super-Resolution (SR)")
        try:
            from src.sr import sr
            sr.run_task(file_path=args.file, folder_path=args.folder)
        except ImportError as e:
            print(f"[ERROR] Failed to import SR module: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
    else:
        print("[MAIN] No valid mode selected.")
        print("Usage T2I: python src/main.py --t2i --nums 10")
        print("Usage SR : python src/main.py --sr --file 'path/to/img.png'")
        parser.print_help()

if __name__ == "__main__":
    main()