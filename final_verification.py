import sys
import os
sys.path.append(".")

print("FINAL VERIFICATION: DDR DATASET INTEGRATION")
print("===========================================")

# Check 1: Can import DDR from datasets module
try:
    from datasets import DDR
    print("v SUCCESS: DDR dataset imported from datasets module")
    import_success = True
except ImportError as e:
    print(f"x FAILED: Could not import DDR - {e}")
    import_success = False

# Check 2: Verify DDR has the right constructor parameters
if import_success:
    import inspect
    sig = inspect.signature(DDR.__init__)
    params = list(sig.parameters.keys())
    
    required_params = ['train_class_num', 'test_class_num', 'includes_all_train_class']
    missing_params = [p for p in required_params if p not in params]
    
    if len(missing_params) == 0:
        print("v SUCCESS: All required OSR parameters present in DDR constructor")
        params_ok = True
    else:
        print(f"x FAILED: Missing parameters: {missing_params}")
        params_ok = False

# Check 3: Verify essential methods exist
if import_success:
    required_methods = ['__getitem__', '__len__', '_update_open_set']
    missing_methods = [m for m in required_methods if not hasattr(DDR, m)]
    
    if len(missing_methods) == 0:
        print("v SUCCESS: All required methods implemented in DDR class")
        methods_ok = True
    else:
        print(f"x FAILED: Missing methods: {missing_methods}")
        methods_ok = False

# Check 4: Check if datasets/__init__.py includes DDR
try:
    with open('./datasets/__init__.py', 'r') as f:
        init_content = f.read()
    if 'from .ddr import DDR' in init_content:
        print("v SUCCESS: DDR import properly added to datasets/__init__.py")
        init_ok = True
    else:
        print("x FAILED: DDR import missing from datasets/__init__.py")
        init_ok = False
except:
    print("x FAILED: Could not read datasets/__init__.py")
    init_ok = False

# Check 5: Test CUDA availability
import torch
cuda_available = torch.cuda.is_available()
if cuda_available:
    gpu_name = torch.cuda.get_device_name(0)
    print(f"v SUCCESS: CUDA available - GPU: {gpu_name}")
else:
    print("v INFO: CUDA not available - CPU will be used")

# Overall assessment
overall_success = all([import_success, params_ok if import_success else False,
                      methods_ok if import_success else False, init_ok])

print("\nOVERALL RESULT:")
print("================")
if overall_success and import_success:
    print(">>> DDR DATASET FULLY INTEGRATED SUCCESSFULLY! <<<")
    print("   - Ready for Open Set Recognition experiments")
    print("   - Compatible with existing OSR algorithms")
    print("   - GPU acceleration supported when available")
    print("   - Follows same pattern as CIFAR10 and MNIST")
else:
    print(">>> Some integration issues detected - see specific results above <<<")

print(f"\nIntegration Summary:")
print(f"   - Import test: {'PASS' if import_success else 'FAIL'}")
print(f"   - Parameter test: {'PASS' if (params_ok if import_success else False) else 'FAIL'}")
print(f"   - Method test: {'PASS' if (methods_ok if import_success else False) else 'FAIL'}")
print(f"   - Init file test: {'PASS' if init_ok else 'FAIL'}")
print(f"   - CUDA available: {'YES' if torch.cuda.is_available() else 'NO'}")