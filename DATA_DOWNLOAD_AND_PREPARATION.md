# 🌊 ONC Data Download and Preparation

This document covers how to prepare ONC hydrophone data for machine learning.

## 📋 Table of Contents

- [ Downloading Data](#-downloading-data)
- [🏷️ Interactive Spectrogram Labeling Tool](#️-interactive-spectrogram-labeling-tool)
- [🗂️ Creating HDF5 Datasets](#️-creating-hdf5-datasets)
- [🔧 Advanced Options](#-advanced-options)
- [🛠️ Troubleshooting](#️-troubleshooting)

---

## 📥 Downloading Data

**Data downloading functionality has been moved to a dedicated repository:**

� **[onc-hydrophone-data](https://github.com/Spiffical/onc-hydrophone-data)**

This repository provides tools for:
- Downloading spectrograms from Ocean Networks Canada (ONC)
- Downloading FLAC audio files
- Generating custom spectrograms from audio files
- Deployment validation and device discovery

### Installation

```bash
# Install from PyPI
pip install onc-hydrophone-data

# Or clone for development
git clone https://github.com/Spiffical/onc-hydrophone-data.git
cd onc-hydrophone-data
pip install -e .
```

---

## 🏷️ Interactive Spectrogram Labeling Tool

For efficient manual annotation of spectrograms, use the interactive Dash-based labeling application:

### ✨ Features
- **Visual spectrogram display** with integrated audio playback
- **Customizable labeling categories** for anomaly detection  
- **Pagination and navigation** for large datasets
- **Automatic audio-spectrogram matching** based on timestamps
- **Label persistence** with JSON export
- **Dual view modes** (grid and detailed modal)
- **Intelligent caching** for optimal performance

### 🚀 Quick Start
```bash
# Configure paths in config.yaml, then run:
cd tools/labeling
python run.py
```

**For complete setup and usage instructions, see: [tools/labeling/README.md](tools/labeling/README.md)**

---

## 🗂️ Creating HDF5 Datasets

Convert downloaded spectrograms into ML-ready HDF5 datasets with flexible labeling.

```bash
# Basic usage
python scripts/create_h5_dataset.py --h5_filename datasets/hydrophone_data.h5 --data_folders data/ICLISTENHF6020/

# Multiple devices
python scripts/create_h5_dataset.py --h5_filename datasets/multi_device.h5 --data_folders data/DEVICE1/ data/DEVICE2/
```

### 📂 Supported Structures
- **Enhanced**: `data/mat/DEVICE/METHOD_DATES/processed/*.mat` (recommended)
- **Flat**: `folder/*.mat` + `folder/labels.json`
- **Nested**: Any structure with `.mat` files

### 🏷️ Labels and Classification

#### 📍 Label File Placement
Place `labels.json` files at any level (checked in order):
1. **Method**: `data/DEVICE/METHOD_DATES/labels.json` (highest priority)
2. **Device**: `data/DEVICE/labels.json`  
3. **Folder**: `your_folder/labels.json`
4. **Automatic**: Folder-based rules (if no JSON entry found)

#### 📝 Label Format
```json
{
  "spectrogram_20210115_120000.mat": ["ship_noise", "anomaly"],
  "spectrogram_20210118_140000.mat": ["normal"],
  "spectrogram_20210120_100000.mat": ["whale_call", "bio_acoustic"]
}
```

#### 🤖 Automatic Labeling

**When automatic labeling is used:**
- Files **without** entries in any `labels.json` file
- Files in folders without any `labels.json` file present

**Automatic rules:**
- `processed/` folders → "normal"
- `rejects/` folders → "anomaly" 
- All other locations → "normal"

**Note:** If a file has an entry in any `labels.json` file (even an empty list `[]`), automatic labeling is **not** applied.

**💡 Labeling App**: Use the interactive labeling tool in `tools/labeling/` for manual annotation!

---

## 🔧 Advanced Options

```bash
# Custom batch size and dimensions
python scripts/create_h5_dataset.py --h5_filename datasets/custom.h5 --data_folders data/ --batch_size 50 --target_dim 256 256

# Multiple structures together
python scripts/create_h5_dataset.py --h5_filename datasets/mixed.h5 --data_folders data/enhanced/ data/flat/
```

### 📊 HDF5 Output
- **`spectrograms`**: Image arrays
- **`labels`**: Multi-hot encoded vectors
- **`sources`**: Original .mat filenames
- **`label_strings`**: Human-readable labels

### 🔄 Complete Workflow
```bash
# 1. Download data using onc-hydrophone-data repo
#    See: https://github.com/Spiffical/onc-hydrophone-data

# 2. Label your data using the interactive tool (recommended)
cd tools/labeling && python run.py

# 3. Create HDF5 dataset
python scripts/create_h5_dataset.py --h5_filename datasets/my_data.h5 --data_folders data/DEVICE/
```

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| No .mat files found | Verify folder structure |
| Labels not loading | Check JSON syntax |
| Memory errors | Reduce `--batch_size` |

**💡 For download-related issues, see the [onc-hydrophone-data](https://github.com/Spiffical/onc-hydrophone-data) repository.**