# Data Split Update: 70/15/15 Train/Val/Test

## Changes Made

### 1. Modified `datasets/ddr.py`
**Added proper train/val/test split support:**
- New parameter: `split='train'/'val'/'test'`
- Implements stratified 70/15/15 split using `sklearn.model_selection.train_test_split`
- Fixed random seed (42) for reproducibility
- Backward compatible: old `train=True/False` still works

**Split Distribution:**
```
Train:  8,764 images (70.0%)
Val:    1,879 images (15.0%)
Test:   1,879 images (15.0%)
Total: 12,522 images
```

### 2. Updated `train_full_ddr.py`
**Now uses proper validation set:**
- Creates 3 dataloaders: train, val, test
- Uses **validation set** for model selection (early stopping)
- Test set is **completely unseen** during training
- Only evaluates on test set at the very end
- Reports all 3 accuracies

**Key Changes:**
```python
# OLD (Test leakage):
trainset = DDR(root=dataroot, train=True, ...)
testset = DDR(root=dataroot, train=False, ...)  # Used for validation!

# NEW (Proper split):
trainset = DDR(root=dataroot, split='train', ...)
valset = DDR(root=dataroot, split='val', ...)    # New!
testset = DDR(root=dataroot, split='test', ...)  # Truly unseen
```

### 3. Updated `evaluate_cross_dataset.py`
**Uses test split for final evaluation:**
```python
# OLD:
ddr_set = DDR(root='DDR dataset', train=False, ...)

# NEW:
ddr_set = DDR(root='DDR dataset', split='test', ...)
```

## Benefits

### ✅ No Test Set Leakage
- Test set is **never** seen during training
- Model selection happens on validation set only
- Follows ML best practices

### ✅ Reproducibility
- Fixed random seed ensures same split across runs
- Stratified sampling maintains class balance
- All 3 seeds will use identical data splits

### ✅ Better Generalization Estimate
- Test accuracy is now a true measure of generalization
- No risk of overfitting to test set

## Verification

Run this to verify the split:
```bash
python -c "from datasets import DDR; import torchvision.transforms as transforms; t = transforms.ToTensor(); train = DDR('DDR dataset', split='train', transform=t, train_class_num=5, test_class_num=5); val = DDR('DDR dataset', split='val', transform=t, train_class_num=5, test_class_num=5); test = DDR('DDR dataset', split='test', transform=t, train_class_num=5, test_class_num=5); total = len(train) + len(val) + len(test); print(f'Train: {len(train)} ({len(train)/total*100:.1f}%)'); print(f'Val: {len(val)} ({len(val)/total*100:.1f}%)'); print(f'Test: {len(test)} ({len(test)/total*100:.1f}%)')"
```

Expected output:
```
Train: 8764 (70.0%)
Val: 1879 (15.0%)
Test: 1879 (15.0%)
```

## Next Steps

1. **Retrain the model** with the new split:
   ```bash
   python train_full_ddr.py
   ```

2. **Run reproducibility** with proper splits:
   ```bash
   python run_cross_dataset_reproducibility.py
   ```

3. **Evaluate** on truly unseen test set:
   ```bash
   python evaluate_cross_dataset.py
   ```

## Backward Compatibility

Old code using `train=True/False` will still work:
- `train=True` → automatically uses `split='train'`
- `train=False` → automatically uses `split='test'`

However, **new code should use `split=` parameter explicitly** for clarity.
