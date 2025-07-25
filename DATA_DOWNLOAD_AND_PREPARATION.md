# 🌊 ONC Data Download and Preparation

Complete guide for downloading Ocean Networks Canada spectrograms, FLAC audio files, and preparing ML-ready HDF5 datasets.

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [⚙️ Setup](#️-setup)
- [📥 Downloading Spectrograms](#-downloading-spectrograms)
- [🎵 Downloading FLAC Audio Files](#-downloading-flac-audio-files)
- [🎶 Custom Spectrogram Generation](#-custom-spectrogram-generation)
- [🏷️ Interactive Spectrogram Labeling Tool](#️-interactive-spectrogram-labeling-tool)
- [🗂️ Creating HDF5 Datasets](#️-creating-hdf5-datasets)
- [🔧 Advanced Options](#-advanced-options)
- [🛠️ Troubleshooting](#️-troubleshooting)

## 🚀 Quick Start

```bash
# 1. Interactive download (recommended for beginners) - uses sampling strategy
#    Now includes option to download FLAC files!
python scripts/download_spectrograms.py

# 2. Direct download with custom batch size
python scripts/download_spectrograms.py --mode sampling --device ICLISTENHF6020 --start-date 2021 1 1 --threshold 500 --spectrograms-per-batch 12 --check-deployments

# 3. Download spectrograms WITH corresponding FLAC audio files
python scripts/download_spectrograms.py --mode sampling --device ICLISTENHF6020 --start-date 2021 1 1 --threshold 500 --spectrograms-per-batch 6 --download-flac

# 4. Generate custom spectrograms from FLAC files (NEW!)
python scripts/generate_spectrograms.py --input-dir data/ICLISTENHF6020/flac/ --win-dur 2.0

# 5. Label your spectrograms using the interactive tool (recommended)
cd tools/labeling && python run.py

# 6. Create HDF5 dataset
python scripts/create_h5_dataset.py data/mat/ICLISTENHF6020/ --output datasets/hydrophone_data.h5
```

## ✨ Key Features

- **🤖 Smart Interactive Mode**: Guided setup that uses the intelligent sampling strategy and includes FLAC audio option
- **🎵 FLAC Audio Download**: Download corresponding raw audio files alongside spectrograms
- **🎶 Custom Spectrogram Generation**: Create spectrograms with any duration/parameters from FLAC files
- **🚀 Deployment Validation**: Ensures hydrophones were deployed during requested periods  
- **📊 Device Discovery**: Browse available hydrophones with deployment information
- **⏰ Date Validation**: Checks dates fall within active deployment periods
- **💾 Efficient Caching**: Minimizes API calls through intelligent caching
- **🔧 Multiple Modes**: Sampling, range, specific times, and deployment checking
- **📁 Universal Folder Support**: Works with enhanced, flat, and nested folder structures
- **🗂️ HDF5 Dataset Creation**: Convert downloaded spectrograms into ML-ready datasets

## ⚙️ Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure ONC API token:**
   Create/edit `.env` file:
   ```
   ONC_TOKEN=your_actual_onc_token_here
   DATA_DIR=./data
   ```

## 📥 Downloading Spectrograms

### 🎯 Usage Modes

| Mode | Description | Example |
|------|-------------|---------|
| **Interactive** | Guided setup using **sampling strategy** (recommended) | `python scripts/download_spectrograms.py` |
| **Sampling** | Smart sampling from date range | `--mode sampling --threshold 1000` |
| **Range** | All spectrograms in date range | `--mode range --start-date 2021 1 1 --end-date 2021 1 7` |
| **Specific** | Exact timestamps from JSON | `--mode specific --config times.json` |
| **Check** | View deployment info | `--mode check-deployments` |

**📌 Note**: **Interactive mode** is simply a guided way to set up the intelligent sampling strategy. It prompts you for device, dates, threshold, and spectrograms per batch, then uses the same smart sampling algorithm described below.

### 🧠 Intelligent Sampling Strategy

The **sampling mode** (including **interactive mode**) uses a smart algorithm to efficiently distribute downloads across your date range:

**How it works:**
1. **Data Availability Check**: Queries ONC API to find which days have data available
2. **Request Calculation**: Determines number of requests needed based on `spectrograms_per_batch`:
   ```
   total_requests = ceil(threshold_num / spectrograms_per_batch)
   ```
3. **Optimal Day Spacing**: Distributes requests evenly across available days
4. **Random Time Distribution**: Uses random hours (0-23) and minutes (0-59) for maximum temporal diversity
5. **Duplicate Prevention**: Automatically skips dates where files already exist
6. **Adaptive Sampling**: Handles both sparse sampling across many days and multiple requests per day

**Benefits:**
- **Even temporal coverage** across your entire date range
- **Full 24-hour sampling** with random start times for maximum diversity
- **Efficient API usage** by checking availability first
- **Resume-friendly** by skipping existing downloads

### 📊 Spectrograms Per Batch

Control how many 5-minute spectrograms are downloaded per request with `--spectrograms-per-batch`:

| Batch Size | Duration |
|------------|----------|
| `1` | 5 minutes |
| `6` | 30 minutes (default) |
| `12` | 1 hour |
| `36` | 3 hours |

```bash
# Custom batch size example
python scripts/download_spectrograms.py --mode sampling --spectrograms-per-batch 12
```

### 🚀 Deployment Validation

Ensures hydrophones were active during requested periods. Add `--check-deployments` to verify:
- ✅ Deployment coverage for your dates
- 📍 Exact locations and coordinates  
- 🔍 Data availability verification
- 💡 Alternative suggestions if needed

```bash
python scripts/download_spectrograms.py --check-deployments
```

### 🎛️ Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--mode` | Download mode | Interactive prompt |
| `--device` | Hydrophone device code | Interactive selection |
| `--spectrograms-per-batch` | Number of 5-min spectrograms per request | 6 |
| `--download-flac` | Also download FLAC audio files | False |
| `--check-deployments` | Validate deployment periods | Recommended |
| `--start-date` | Start date (YYYY MM DD) | Prompted |
| `--end-date` | End date (YYYY MM DD) | Prompted |
| `--threshold` | Number of spectrograms | Prompted |

### 📁 File Organization

Downloads are organized by device, method, and date range:

```
data/
└── DEVICE/
    └── sampling_YYYY-MM-DD_to_YYYY-MM-DD/
        ├── mat/
        │   ├── processed/     # Downloaded spectrograms
        │   └── rejects/       # Quality-filtered files
        └── flac/              # FLAC audio files (if --download-flac used)
```

**Example:** `data/ICLISTENHF6020/sampling_2021-01-01_to_2021-01-31/`

### 📝 Specific Times Config

For exact timestamps, create a JSON file:
```json
{
  "ICLISTENHF6020": [
    [2021, 1, 15, 12, 0, 0],
    [2021, 1, 15, 18, 30, 0]
  ]
}
```
Format: `[Year, Month, Day, Hour, Minute, Second]`

## 🎵 Downloading FLAC Audio Files

FLAC files contain raw hydrophone audio recordings. Add `--download-flac` to any command or use interactive mode (which now prompts for FLAC preference):

```bash
# Interactive mode (prompts for FLAC)
python scripts/download_spectrograms.py

# Any mode with FLAC
python scripts/download_spectrograms.py --mode sampling --download-flac
```

**Use Cases**: Audio analysis, custom spectrograms, ML training on raw audio
**File Organization**: FLAC files saved in `flac/` subdirectory alongside spectrograms  
**Performance**: 10-50x larger than spectrograms; start with small downloads (--threshold 5-10)

## 🎶 Custom Spectrogram Generation

Generate custom spectrograms from your downloaded FLAC/WAV audio files with configurable parameters. This functionality translates MATLAB spectrogram code to Python, allowing you to create spectrograms with different durations, frequency ranges, and analysis parameters.

### ✨ Features
- **Multiple audio formats**: FLAC, WAV, MP3, M4A support
- **Configurable parameters**: Window duration, overlap, frequency limits
- **MATLAB compatibility**: Outputs .mat files with same structure as MATLAB
- **High-quality plots**: PNG visualizations with customizable colormaps
- **Batch processing**: Process entire directories efficiently
- **Project integration**: Works seamlessly with downloaded FLAC files

### 🚀 Quick Start

```bash
# Interactive mode (recommended)
python scripts/generate_spectrograms.py

# Process FLAC files from ONC downloads
python scripts/generate_spectrograms.py --input-dir data/ICLISTENHF6020/flac/

# Custom parameters for longer spectrograms  
python scripts/generate_spectrograms.py \
  --input-dir data/DEVICE/flac/ \
  --win-dur 2.0 \
  --overlap 0.75 \
  --freq-min 5 \
  --freq-max 20000
```

### 🎛️ Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--win-dur` | Window duration in seconds | 1.0 |
| `--overlap` | Overlap ratio (0-1) | 0.5 |
| `--freq-min` | Minimum frequency (Hz) | 10 |
| `--freq-max` | Maximum frequency (Hz) | 10000 |
| `--colormap` | Matplotlib colormap | turbo |
| `--clim-min` | Color scale minimum (dB) | -60 |
| `--clim-max` | Color scale maximum (dB) | 0 |

### 📁 Output Structure

Custom spectrograms are saved parallel to FLAC directories:

```
data/
└── DEVICE/
    └── sampling_YYYY-MM-DD_to_YYYY-MM-DD/
        ├── flac/                    # Downloaded audio files
        └── custom_spectrograms/     # Generated spectrograms
            ├── audio1.mat           # MATLAB data files
            ├── audio1.png           # PNG visualizations
            ├── audio2.mat
            └── audio2.png
```

### ⚙️ Configuration

Parameters can be configured in `config/dataset_config.yaml`:

```yaml
custom_spectrograms:
  window_duration: 1.0     # Window duration in seconds
  overlap: 0.5             # Overlap ratio (0-1)
  frequency_limits:
    min: 10                # Minimum frequency (Hz)
    max: 10000             # Maximum frequency (Hz)
  colormap: "turbo"        # Matplotlib colormap
  color_limits:
    min: -60               # Color scale minimum (dB)
    max: 0                 # Color scale maximum (dB)
  log_frequency: true      # Use log frequency scale
```

### 🔬 Use Cases

**Different Analysis Requirements:**
- **High time resolution**: `--win-dur 0.5 --overlap 0.75` for transient events
- **High frequency resolution**: `--win-dur 4.0 --overlap 0.9` for tonal analysis  
- **Low-frequency focus**: `--freq-min 1 --freq-max 1000` for whale calls
- **Wideband analysis**: `--freq-min 1 --freq-max 50000` for full spectrum

**Custom Duration Spectrograms:**
Unlike ONC's fixed 5-minute spectrograms, you can create any duration by adjusting window parameters to analyze longer or shorter audio segments.

### 💻 Programmatic Usage

```python
from utils.audio import SpectrogramGenerator

# Create generator with custom parameters
generator = SpectrogramGenerator(
    win_dur=2.0,           # 2 second windows
    overlap=0.75,          # 75% overlap
    freq_lims=(5, 20000),  # 5 Hz to 20 kHz
    colormap='viridis'
)

# Process a directory
results = generator.process_directory(
    input_dir="data/DEVICE/flac/",
    save_dir="data/DEVICE/custom_spectrograms/",
    save_mat=True,
    save_plot=True
)
```

See `examples/generate_custom_spectrograms_example.py` for complete examples.

## 🏷️ Interactive Spectrogram Labeling Tool

For efficient manual annotation of spectrograms, use the interactive Dash-based labeling application:

### ✨ Features
- **Visual spectrogram display** with integrated audio playbook
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

## 🗂️ Creating HDF5 Datasets

Convert downloaded spectrograms into ML-ready HDF5 datasets with flexible labeling.

```bash
# Basic usage
python scripts/create_h5_dataset.py data/ICLISTENHF6020/ --output datasets/hydrophone_data.h5

# Multiple devices
python scripts/create_h5_dataset.py data/DEVICE1/ data/DEVICE2/ --output datasets/multi_device.h5
```

### 📂 Supported Structures
- **Enhanced**: `data/DEVICE/METHOD_DATES/mat/processed/*.mat` (recommended)
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

## 🔧 Advanced Options

```bash
# Custom batch size and dimensions
python scripts/create_h5_dataset.py data/ --output datasets/custom.h5 --batch-size 50 --target-dim 256 256

# Multiple structures together
python scripts/create_h5_dataset.py data/enhanced/ data/flat/ --output datasets/mixed.h5

# Download with custom settings
python scripts/download_spectrograms.py --mode sampling --device ICLISTENHF6020 --spectrograms-per-batch 12 --threshold 200 --check-deployments
```

### 📊 HDF5 Output
- **`spectrograms`**: Image arrays
- **`labels`**: Multi-hot encoded vectors
- **`filenames`**: Original .mat names
- **`label_strings`**: Human-readable labels

### 🔄 Complete Workflow
```bash
# 1. Download with custom batch size
python scripts/download_spectrograms.py --mode sampling --spectrograms-per-batch 12 --check-deployments

# 2. Label your data using the interactive tool (recommended)
cd tools/labeling && python run.py
# OR manually create labels file:
# echo '{"spec_001.mat": ["ship_noise"]}' > data/DEVICE/METHOD/labels.json

# 3. Create HDF5 dataset
python scripts/create_h5_dataset.py data/DEVICE/ --output datasets/my_data.h5
```

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Invalid ONC Token | Check `.env` file |
| No Deployment Coverage | Use `--check-deployments` |
| No .mat files found | Verify folder structure |
| Labels not loading | Check JSON syntax |
| Memory errors | Reduce `--batch-size` |
| FLAC download fails | Check network connection and storage space |
| Large FLAC files | Monitor disk space, start with small downloads |

**💡 Pro Tip**: Always use `--check-deployments` to ensure active deployment periods!