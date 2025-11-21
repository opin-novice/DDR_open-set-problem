"""
Simple test to verify DDR dataset integration with the existing OSR framework
"""

import sys
sys.path.append(".")

from datasets import DDR
import torchvision.transforms as transforms

def test_simple_ddr():
    print("Testing DDR dataset integration with OSR framework...")
    
    # Define simple transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Smaller size for faster testing
        transforms.ToTensor(),
    ])
    
    try:
        # Test if we can instantiate the DDR dataset
        print("Creating DDR training dataset...")
        trainset = DDR(root='./DDR dataset',  # This should point to the actual DDR dataset path
                      train=True, 
                      transform=transform,
                      train_class_num=3,      # Use 3 classes for training (0, 1, 2)
                      test_class_num=5,       # Test on total 5 classes (0, 1, 2, 3, 4)
                      includes_all_train_class=True)
        
        print(f"✓ Training set created successfully!")
        print(f"  - Number of samples: {len(trainset)}")
        print(f"  - Classes: {trainset.classes}")
        
        # Test if we can get a sample
        if len(trainset) > 0:
            sample_img, sample_target = trainset[0]
            print(f"  - Sample image shape: {sample_img.shape}")
            print(f"  - Sample target: {sample_target}")
        
        print("\nCreating DDR testing dataset...")
        testset = DDR(root='./DDR dataset',  # This should point to the actual DDR dataset path
                     train=False,
                     transform=transform,
                     train_class_num=3,      # Same as training
                     test_class_num=5,       # Same as training
                     includes_all_train_class=True)

        print(f"✓ Testing set created successfully!")
        print(f"  - Number of samples: {len(testset)}")
        print(f"  - Classes: {testset.classes}")

        # Test if we can get a sample
        if len(testset) > 0:
            sample_img, sample_target = testset[0]
            print(f"  - Sample image shape: {sample_img.shape}")
            print(f"  - Sample target: {sample_target}")

        print(f"\n✓ SUCCESS: DDR dataset successfully integrated with OSR framework!")
        print(f"  - Training uses {trainset.train_class_num} known classes")
        print(f"  - Testing includes {testset.test_class_num} total classes with open set functionality")
        if hasattr(testset, 'openness'):
            print(f"  - Test openness: {testset.openness:.3f}")

        return True
        
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        print("Note: Make sure the DDR dataset path is correct and the DR_grading.csv file exists.")
        return False

if __name__ == "__main__":
    success = test_simple_ddr()
    if success:
        print("\n🎉 DDR dataset integration verified successfully!")
    else:
        print("\n❌ DDR dataset integration failed!")