import torch
from models.resnet_cbam import resnet50_cbam

def test_cbam_structure():
    print("Testing ResNet50-CBAM Structure...")
    
    # 1. Instantiate
    try:
        model = resnet50_cbam(pretrained=True)
        print("✅ Model instantiated successfully.")
    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
        return

    # 2. Check for CBAM modules
    has_cbam = False
    for name, module in model.named_modules():
        if 'ca' in name and 'sa' in name: # Check for both in a block? No, names are separate
            pass
        if isinstance(module, torch.nn.Module):
            if 'BottleneckCBAM' in str(type(module)):
                has_cbam = True
                break
    
    # Check by printing first layer
    print("\nFirst Bottleneck Block:")
    print(model.layer1[0])
    
    if hasattr(model.layer1[0], 'ca') and hasattr(model.layer1[0], 'sa'):
        print("✅ CBAM modules (ca, sa) found in Bottleneck.")
    else:
        print("❌ CBAM modules NOT found in Bottleneck.")

    # 3. Forward Pass
    print("\nTesting Forward Pass...")
    x = torch.randn(2, 3, 224, 224)
    try:
        out = model(x)
        print(f"✅ Forward pass successful. Output shape: {out.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")

if __name__ == "__main__":
    test_cbam_structure()
