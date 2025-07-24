# -*- coding: utf-8 -*-
"""
Test script for SIFT model integration in LibFewShot
"""
import sys
sys.dont_write_bytecode = True

import torch
import os
from core.config import Config
from core import Trainer


def test_sift_integration():
    """Test SIFT model integration"""
    try:
        # Test importing SIFT
        from core.model.metric import SIFT
        print("✓ SIFT import successful")
        
        # Test basic model creation
        model = SIFT(
            way_num=5,
            shot_num=1,
            query_num=15,
            mode='dc',
            classifier_method='metric'
        )
        print("✓ SIFT model creation successful")
        
        return True
    except Exception as e:
        print(f"✗ SIFT integration test failed: {e}")
        return False


def main(rank, config):
    """Main training function"""
    trainer = Trainer(rank, config)
    trainer.train_loop(rank)


if __name__ == "__main__":
    # First test the integration
    if test_sift_integration():
        print("SIFT integration test passed! You can now use SIFT in LibFewShot.")
        
        # Example: Run SIFT with configuration
        try:
            config = Config("./config/sift.yaml").get_config_dict()
            print("✓ SIFT configuration loaded successfully")
            
            if config["n_gpu"] > 1:
                os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
                torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
            else:
                main(0, config)
                
        except Exception as e:
            print(f"Note: Configuration test failed (this is expected if headers are missing): {e}")
            print("You can create a complete configuration based on existing examples.")
    else:
        print("SIFT integration test failed. Please check the installation.")
