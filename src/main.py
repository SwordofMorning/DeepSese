# src/main.py

import argparse
import sys
import os

def main():
    # Initialize Argument Parser
    parser = argparse.ArgumentParser(description="DeepSese Image Generation Framework")
    
    # Define arguments
    parser.add_argument("--t2i", action="store_true", help="Run Text-to-Image generation task")
    parser.add_argument("--nums", type=int, default=1, help="Number of images to generate")
    
    # Future SR argument placeholder
    # parser.add_argument("--sr", action="store_true", help="Run Super-Resolution task")

    args = parser.parse_args()

    # Dispatch Logic
    if args.t2i:
        print("[MAIN] Mode selected: Text-to-Image (T2I)")
        try:
            from t2i import t2i
            t2i.run_task(num_images=args.nums)
        except ImportError as e:
            print(f"[ERROR] Failed to import T2I module: {e}")
            sys.exit(1)
            
    # elif args.sr:
    #     print("[MAIN] Mode selected: Super-Resolution (SR)")
    #     # Call SR module here in the future
        
    else:
        print("[MAIN] No valid mode selected.")
        print("Usage: python src/main.py --t2i --nums 10")
        parser.print_help()

if __name__ == "__main__":
    main()