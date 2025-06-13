# Spectrogram Labeling Tool

This is a Dash application for visualizing and labeling spectrograms with integrated audio playback. You can use it to display a specified number of spectrograms per page, label them based on customizable categories, and listen to the corresponding audio files.

## Features
- **Display Spectrogram**: Load and display spectrograms from `.mat` files in a specified folder.
- **Audio Playback**: Automatically match and play corresponding audio files (`.flac`) for each spectrogram.
- **Pagination**: Dynamically specify the number of spectrograms to display per page with navigation controls.
- **Labeling**: Choose from a list of customizable labels to assign to each spectrogram.
- **Dual View Modes**: 
  - Grid view with audio players under each spectrogram
  - Detailed modal view with enhanced audio controls when clicking on spectrograms
- **Intelligent Audio Matching**: Automatically matches audio files to spectrograms based on timestamps in filenames.
- **Caching Mechanism**:
    - Limited Cache Size: Caches up to 400 images to optimize performance.
    - Background Preloading: Preloads images for the next page in the background to enhance user experience.
- **Label Persistence**: Automatically saves labeled data to a specified output JSON file (creates a new file or updates an existing one).

## Requirements

- Python 3.7 or higher
- Dash
- NumPy
- SciPy
- OpenCV
- Matplotlib
- Cachetools
- Plotly
- PyYAML

## Installation

1. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The application can be configured in two ways:

### 1. Configuration File (Recommended)

Create or edit the `config.yaml` file to set up all parameters. This is the recommended approach for persistent settings:

```yaml
# Data paths
data:
  folder: "/path/to/spectrograms"  # Path to folder containing spectrogram files
  audio_folder: "/path/to/audio"   # Path to folder containing audio files (.flac)
  output_file: "labels.json"        # Path to output file for saving labeled filenames

# Display settings
display:
  target_dim: [512, 512]           # Target dimensions [height, width] for reshaping the data
  specs_per_page: 50               # Number of spectrograms to display per page

# Audio settings
audio:
  enable: true                     # Whether to enable audio playback
  auto_match: true                 # Whether to automatically match audio files to spectrograms

# Labeling options
labels:
  available:                       # Labels available for selection
    - "Rain"
    - "Engine Noise"
    - "Unknown Features"

# Cache settings
cache:
  max_size: 400                    # Maximum number of images to cache
  preload_next_page: true          # Whether to preload images for the next page
```

### 2. Command-Line Arguments (Optional Overrides)

You can override specific settings from the config file using command-line arguments. Command-line arguments take precedence over the config file:

```bash
python run.py --folder <path_to_spectrogram_folder> --audio_folder <path_to_audio_folder> --output_file <path_to_output_file> [--target_dim HEIGHT WIDTH] [--specs_per_page NUMBER] [--available_labels LABEL1 LABEL2 ...]
```

#### Arguments:

- `--folder`: Path to the folder containing spectrogram files.
- `--audio_folder`: Path to the folder containing audio files (.flac).
- `--output_file`: Path to the output file for saving labeled filenames.
- `--target_dim`: Target dimensions (height, width) for reshaping the data. Default: `(512, 512)`.
- `--specs_per_page`: Number of spectrograms to display per page. Default: `50`.
- `--available_labels`: Labels available for selection. Default: `["Rain", "Engine Noise", "Unknown Features"]`.
- `--enable_audio`: Enable audio playback (overrides config file).
- `--disable_audio`: Disable audio playback (overrides config file).

## Audio Functionality

### File Matching
The application automatically matches audio files to spectrograms based on timestamps in the filenames:

- **Spectrogram files**: `ICLISTENHF6406_20240523T061507.000Z_20240523T062007.000Z-spect_plotRes.mat`
- **Audio files**: `ICLISTENHF6406_20240523T061507.000Z.flac`

The system finds audio files that fall within the time range of each spectrogram (with a tolerance of ±5 minutes).

### Audio Players
- **Grid View**: Simple audio controls under each spectrogram thumbnail
- **Modal View**: Enhanced audio player with additional information when viewing individual spectrograms
- **Format Support**: Optimized for FLAC files (browser compatibility may vary)

### Disabling Audio
If you don't have audio files or want to disable audio functionality:
```yaml
audio:
  enable: false
```

Or use the command line flag: `--disable_audio`

## Usage

### Using Configuration File Only

1. Edit the `config.yaml` file to set your desired parameters:
   ```yaml
   data:
     folder: "/data/ICLISTENHF6406/sampling_2023-10-10_to_2024-08-07/mat/processed"
     audio_folder: "/data/ICLISTENHF6406/sampling_2023-10-10_to_2024-08-07/flac"
     output_file: "/data/ICLISTENHF6406/sampling_2023-10-10_to_2024-08-07/labels.json"
   ```

2. Run the application:
    ```bash
    python run.py
    ```

### Using Command-Line Arguments to Override Config

```bash
# Override just the folder paths
python run.py --folder /new/path/to/spectrograms --audio_folder /new/path/to/audio

# Override multiple settings
python run.py --folder /new/path/to/spectrograms --audio_folder /new/path/to/audio --output_file new_labels.json --specs_per_page 25

# Disable audio for this session
python run.py --disable_audio
```

## Example Configuration

Here's an example configuration for labeling ONC hydrophone data with audio playback:

```yaml
data:
  folder: "/data/ICLISTENHF6406/sampling_2023-10-10_to_2024-08-07/mat/processed"
  audio_folder: "/data/ICLISTENHF6406/sampling_2023-10-10_to_2024-08-07/flac"
  output_file: "/data/ICLISTENHF6406/sampling_2023-10-10_to_2024-08-07/labels.json"

display:
  target_dim: [512, 512]
  specs_per_page: 25

audio:
  enable: true
  auto_match: true

labels:
  available:
    - "Rain"
    - "Engine Noise"
    - "Unknown Features"
    - "Biological Sounds"
    - "Ambient Noise"
```

## Troubleshooting

### Audio Issues
- **No audio players appear**: Check that `audio_folder` is correctly set and contains `.flac` files
- **Audio won't play**: Some browsers have limited FLAC support. Files will still be available for download
- **No matching audio**: The timestamp matching may need adjustment. Check the tolerance settings in the code

### Performance
- **Slow loading**: Reduce `specs_per_page` or adjust `cache.max_size`
- **Memory issues**: Lower the cache size or disable `preload_next_page`