# Spectrogram Labeling Tool

This is a Dash application for visualizing and labeling spectrograms with integrated audio playback and hierarchical labeling system.

## Features
- **Display & Navigation**: Load spectrograms from `.mat` files with pagination controls
- **Audio Playback**: Automatically match and play corresponding audio files (`.flac`)
- **Hierarchical Labeling**: Standardized taxonomy with 5 main categories (Anthropophony, Biophony, Geophony, Instrumentation, Other)
- **Smart Selection**: Collapsible tree structure, search functionality, selection at any hierarchy level
- **Display Options**: Toggle colormap (Viridis/hydrophone) and Y-axis scaling (linear/log)
- **Performance**: Image caching and background preloading for smooth navigation

## Requirements

- Python 3.7+
- Dash, NumPy, SciPy, OpenCV, Matplotlib, Cachetools, Plotly, PyYAML

## Installation

1. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage
```bash
python run.py
```

### With Custom Configuration
```bash
python run.py --config my_config.yaml
```

### Key Command Line Options
- `--folder`: Spectrogram files path
- `--audio_folder`: Audio files path (.flac)
- `--output_file`: Labels output JSON file
- `--specs_per_page`: Spectrograms per page
- `--enable_audio` / `--disable_audio`: Audio playback toggle

## Configuration

Configure via `config.yaml`:

```yaml
labels:
  use_hierarchical: true           # Enable hierarchical labeling
  hierarchical:
    allow_partial_paths: true      # Allow selection at any level
    search:
      enable: true                 # Enable search functionality
```

## Hierarchical Labeling

### Label Format
```json
{
  "file.mat": [
    "Biophony > Marine mammal > Cetacean > Blue whale",
    "Anthropophony > Vessel > Cargo ship"
  ]
}
```

### Usage Tips
1. **Navigate**: Click ▶ arrows to expand categories
2. **Search**: Type keywords like "whale", "vessel", "rain"
3. **Select Wisely**: Choose broader categories if uncertain
4. **Multiple Labels**: Select multiple categories for mixed sounds
5. **Remove**: Click × on badges to remove labels

### Examples
- **Blue whale call**: Biophony > Marine mammal > Cetacean > Baleen whale > Blue whale
- **Unknown whale**: Biophony > Marine mammal > Cetacean (partial path is fine)
- **Mixed sounds**: Select both vessel noise + rain categories

## Backward Compatibility

- Existing flat label files automatically supported
- Old labels converted to hierarchical format where possible
- Legacy mode available: `use_hierarchical: false`

## Dataset Integration

```bash
# Create datasets with hierarchical labels
python scripts/create_h5_dataset.py data/DEVICE/ --output datasets/hierarchical_data.h5
```

Both old and new label formats supported automatically.

## Troubleshooting

- **Labels not saving**: Check output file permissions
- **Slow performance**: Reduce `specs_per_page` in config
- **Legacy mode**: Set `use_hierarchical: false` for old interface