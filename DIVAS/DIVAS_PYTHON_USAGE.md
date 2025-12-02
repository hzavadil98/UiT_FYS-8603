# PyDIVAS - Python Wrapper for DIVAS R Package

Documentation for using the DIVAS (Distinctive and Indistinctive Variation Analysis System) R package from Python.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Data Format Convention](#data-format-convention)
4. [API Reference](#api-reference)
5. [Results Structure](#results-structure)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Python packages:
```bash
pip install rpy2 numpy
```

Requires rpy2 >= 3.4 for modern conversion API.

### R packages:
```R
# In R console
install.packages("DIVAS")
install.packages("R.utils")  # Required for keymapname generation
```

---

## Quick Start

```python
from src.py_divas import PyDIVAS
import numpy as np

# Initialize
divas = PyDIVAS()

# Prepare data blocks: (n_samples, n_features)
# All blocks must have the same number of samples
data_blocks = {
    'View1': np.random.randn(100, 50),  # 100 samples, 50 features
    'View2': np.random.randn(100, 30),  # 100 samples, 30 features
}

# Run DIVAS
results = divas.run_divas(
    datablock=data_blocks,
    nsim=400,
    colCent=True,
    seed=42
)

# Access results directly
print(f"Estimated ranks: {results['rBars']}")
print(f"Joint patterns: {results['keymapname']}")

# Access joint structures by pattern ID
joint_scores = results['jointBasisMap']['3']  # Pattern ID '3' (binary 11 = all blocks)
loadings_block1 = results['matLoadings'][0]['3']  # Loadings for first block, pattern '3'
```

---

## Data Format Convention

### Matrix Shape Convention

**Python convention (your input)**: `(n_samples, n_features)`
- Rows = samples/observations
- Columns = features/variables

**R DIVAS convention (internal)**: `(n_features, n_samples)`
- Rows = features/variables  
- Columns = samples/observations

**The PyDIVAS wrapper automatically transposes your matrices** between conventions. Always provide data in standard Python format `(n_samples, n_features)`.

```python
# ✅ CORRECT: Provide data in Python convention
data = {
    'Block1': np.random.randn(100, 50),  # 100 samples, 50 features
    'Block2': np.random.randn(100, 30),  # 100 samples, 30 features
}

# ❌ WRONG: Don't transpose yourself
data = {
    'Block1': np.random.randn(50, 100),  # Backwards!
}
```

**Requirements:**
- All data blocks must have **same number of rows** (samples)
- Data blocks can have **different numbers of columns** (features)

---

## API Reference

### `PyDIVAS.__init__()`

Initialize the R environment and load the DIVAS package.

```python
divas = PyDIVAS()
```

---

### `PyDIVAS.run_divas()`

Run DIVAS analysis on multiple data blocks.

```python
results = divas.run_divas(
    datablock,
    nsim=400,
    iprint=False,
    colCent=False,
    rowCent=False,
    figdir=None,
    seed=None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `datablock` | `Dict[str, np.ndarray]` | Required | Dictionary of data blocks. Keys are block names, values are 2D arrays in `(n_samples, n_features)` format |
| `nsim` | `int` | 400 | Number of simulations for signal extraction. Higher = more accurate but slower |
| `iprint` | `bool` | False | Print detailed R output during execution |
| `colCent` | `bool` | False | Center columns (features). Usually `True` to remove feature means |
| `rowCent` | `bool` | False | Center rows (samples). Usually `False` |
| `figdir` | `str` | None | Directory to save diagnostic figures. `None` = no figures saved |
| `seed` | `int` | None | Random seed for reproducibility. Recommended to set |

**Returns:** `Dict` containing DIVAS decomposition results (see [Results Structure](#results-structure))

**Recommendations:**

| Scenario | nsim | colCent | rowCent |
|----------|------|---------|---------|
| Standard analysis | 400 | True | False |
| Quick test | 100-200 | True | False |
| High accuracy | 500-1000 | True | False |

---

### `PyDIVAS.transform()`

Transform new data into the DIVAS joint space using learned loadings.

```python
transformed = PyDIVAS.transform(
    input,
    results,
    block_identifier,
    data_block_n
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `np.ndarray` | New data matrix in Python convention `(n_samples, n_features)` |
| `results` | `Dict` | DIVAS results dictionary from `run_divas()` |
| `block_identifier` | `str` | Pattern identifier (e.g., `'7'` for joint across all blocks for 3 blocks) |
| `data_block_n` | `int` | Data block index (1-based) to get loadings from |

**Returns:** `np.ndarray` of shape `(n_samples, n_components)` - transformed data in joint space

**Example:**

```python
# Train DIVAS on training data
results = divas.run_divas(train_data, nsim=400, colCent=True, seed=42)

# Transform test data using block 1 loadings for pattern '7'
test_transformed = PyDIVAS.transform(
    test_data['Block1'],  # (n_test_samples, n_features)
    results,
    block_identifier='7',  # Joint pattern across all blocks
    data_block_n=1         # Use first block's loadings
)
# Result: (n_test_samples, n_joint_components)
```

---

## Results Structure

The `results` dictionary contains the full DIVAS decomposition:

### Key Components

| Key | Type | Description |
|-----|------|-------------|
| `rBars` | `np.ndarray` | Estimated ranks for each data block |
| `keymapname` | `dict` | Human-readable pattern names, e.g., `{'7': 'Block1+Block2+Block3'}` |
| `keyIdxMap` | `dict` | Index mapping for patterns |
| `jointBasisMap` | `dict` | **Joint scores** for each pattern. Keys are pattern IDs (e.g., `'7'`, `'3'`), values are `(n_samples, n_components)` arrays |
| `matLoadings` | `list` | **Loadings** for each block. `matLoadings[i][pattern_id]` gives loadings matrix `(n_features, n_components)` |
| `VBars` | `list` | Right singular vectors for each block |
| `UBars` | `list` | Left singular vectors for each block |
| `phiBars` | `list` | Column space singular values |
| `psiBars` | `list` | Row space singular values |
| `individual` | `dict` | Individual structures per block |
| `noise` | `dict` | Estimated noise per block |

### Understanding Pattern IDs

DIVAS identifies joint and individual structures using **binary pattern encoding**:

**For 3 data blocks:**
- Pattern `111` (decimal 7) = Joint across **all three** blocks
- Pattern `011` (decimal 3) = Joint across **blocks 1 and 2** only  
- Pattern `001` (decimal 1) = **Individual** to block 1 only

The `keymapname` dictionary provides human-readable labels:
```python
results['keymapname']
# Output: {'7': 'Block1+Block2+Block3', '3': 'Block1+Block2', '1': 'Block1'}
```

### Accessing Structures

```python
# Get joint scores across all blocks (pattern '7')
all_joint_scores = results['jointBasisMap']['7']  # Shape: (n_samples, n_joint_components)

# Get loadings for first block, pattern '7'
block1_loadings = results['matLoadings'][0]['7']  # Shape: (n_features_block1, n_joint_components)

# Reconstruct joint structure for block 1
reconstructed_joint = all_joint_scores @ block1_loadings.T  # Shape: (n_samples, n_features_block1)

# Get individual structure for first block
individual_block1 = results['individual']['Block1']  # Shape: (n_features_block1, n_samples) - R format
```

---

## Examples

### Example 1: Multi-View Mammography Analysis

```python
from src.py_divas import PyDIVAS
import numpy as np

# Initialize
divas = PyDIVAS()

# Features from different mammography views
features_cc = np.random.randn(500, 128)    # CC view: (500 patients, 128 features)
features_mlo = np.random.randn(500, 128)   # MLO view: (500 patients, 128 features)
features_mirai = np.random.randn(500, 64)  # MIRAI risk: (500 patients, 64 features)

data = {
    'CC_View': features_cc,
    'MLO_View': features_mlo,
    'MIRAI_Risk': features_mirai
}

# Run DIVAS
results = divas.run_divas(
    datablock=data,
    nsim=500,
    colCent=True,
    seed=42,
    iprint=True
)

# Analyze results
print(f"Estimated ranks: {results['rBars']}")
print(f"Joint patterns: {results['keymapname']}")

# Access joint structure across all views
if '7' in results['jointBasisMap']:
    joint_all = results['jointBasisMap']['7']
    print(f"Joint structure across all views: {joint_all.shape}")
    
# Get loadings for CC view, joint pattern
cc_loadings = results['matLoadings'][0]['7']
print(f"CC view loadings: {cc_loadings.shape}")
```

### Example 2: Transform Test Data

```python
# Train on training set
train_data = {
    'View1': train_features1,  # (n_train, n_features1)
    'View2': train_features2   # (n_train, n_features2)
}

results = divas.run_divas(train_data, nsim=400, colCent=True, seed=42)

# Transform test set using learned joint structure
test_features1 = test_data['View1']  # (n_test, n_features1)

test_joint = PyDIVAS.transform(
    test_features1,
    results,
    block_identifier='3',  # Joint pattern ID
    data_block_n=1         # Use View1 loadings
)

# Use transformed features for prediction
# test_joint shape: (n_test, n_joint_components)
predictions = classifier.predict(test_joint)
```

### Example 3: Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot estimated ranks
plt.figure(figsize=(10, 6))
plt.bar(range(len(results['rBars'])), results['rBars'])
plt.xlabel('Block Index')
plt.ylabel('Estimated Rank')
plt.title('DIVAS Estimated Ranks')
plt.xticks(range(len(data.keys())), data.keys())
plt.show()

# Heatmap of joint structure
if '7' in results['jointBasisMap']:
    joint = results['jointBasisMap']['7']
    plt.figure(figsize=(10, 8))
    sns.heatmap(joint[:50, :], cmap='coolwarm', center=0)
    plt.title('Joint Structure (first 50 samples)')
    plt.xlabel('Joint Components')
    plt.ylabel('Samples')
    plt.show()
```

---

## Troubleshooting

### Shape Mismatch Error

```python
# ❌ Problem: Different number of samples
data = {
    'Block1': np.random.randn(100, 50),  # 100 samples
    'Block2': np.random.randn(150, 30),  # 150 samples - WRONG!
}

# ✅ Solution: Same number of rows
data = {
    'Block1': np.random.randn(100, 50),  # 100 samples
    'Block2': np.random.randn(100, 30),  # 100 samples
}
```

### Failed to Load DIVAS R Package

```R
# In R console:
install.packages("DIVAS")
install.packages("R.utils")
```

### R Not Accessible from Python

Set `R_HOME` environment variable:
```bash
export R_HOME=/Library/Frameworks/R.framework/Resources  # macOS
export R_HOME=/usr/lib/R                                 # Linux
```

### Memory Issues with Large Datasets

- Reduce `nsim` (e.g., 200 instead of 400)
- Apply dimensionality reduction before DIVAS (e.g., PCA)
- Process data in batches

### Transform Method Errors

```python
# ❌ ValueError: Input shape mismatch
# Make sure input has correct number of features
input_data.shape[1] == results['matLoadings'][block_idx][pattern_id].shape[0]

# ❌ KeyError: Pattern ID not found
# Check available patterns first
print(results['keymapname'].keys())
```

---

## Testing

Run the validation script:

```bash
cd /Users/jazav7774/UiT/FYS-8603
python test_pydivas.py
```

Expected output:
```
============================================================
Testing PyDIVAS Wrapper
============================================================

1. Initializing PyDIVAS...
   ✓ PyDIVAS initialized successfully

2. Creating synthetic data...
   ✓ Synthetic data created

3. Running DIVAS analysis...
   ✓ DIVAS analysis completed

✓ All tests completed successfully!
```

---

## Summary

PyDIVAS provides a simple Python interface to the DIVAS R package:

**Basic workflow:**
1. Initialize: `divas = PyDIVAS()`
2. Prepare data in `(n_samples, n_features)` format
3. Run: `results = divas.run_divas(data, nsim=400, colCent=True, seed=42)`
4. Access results directly via dictionary keys
5. Transform new data: `PyDIVAS.transform(new_data, results, pattern_id, block_n)`

**Key features:**
- ✅ Automatic matrix transpose handling (Python ↔ R conventions)
- ✅ Modern rpy2 API (no deprecation warnings)
- ✅ Direct access to all DIVAS outputs
- ✅ Transform method for applying learned structures to new data

---

**Last Updated**: November 2025  
**Version**: 1.2
