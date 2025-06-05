# Spectrogram Download Script

This script utilizes the `SpectrogramDownloader` class to download spectrograms from Ocean Networks Canada (ONC) with improved user-friendly messaging and progress tracking.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your ONC API token:**
   Edit the `.env` file and replace `your_onc_api_token_here` with your actual ONC API token:
   ```
   ONC_TOKEN=your_actual_onc_token_here
   DATA_DIR=./data
   ```

## Features

- **Clean Progress Tracking**: Clear progress indicators with emoji status icons
- **Smart Warning Suppression**: ONC API warnings are suppressed by default for cleaner output
- **Batch Progress**: Shows progress for each download batch and file processing
- **Error Handling**: Comprehensive error handling with helpful suggestions
- **Verbose Mode**: Optional detailed output including all ONC API messages

## Usage

The script supports three different download modes:

### 1. Sampling Schedule Mode
Downloads spectrograms based on an intelligent sampling schedule to reach a target number of files:

```bash
python scripts/download_spectrograms.py --mode sampling --device ICLISTENHF6020 --start-date 2020 10 2 --threshold 1000
```

**Parameters:**
- `--device`: Device code (e.g., ICLISTENHF6020)
- `--start-date`: Start date as three integers (YEAR MONTH DAY)
- `--threshold`: Target number of spectrograms to download
- `--num-days`: (Optional) Number of days to consider
- `--filetype`: File type to download (`png` or `mat`, default: `mat`)
- `--verbose`: (Optional) Show detailed ONC API messages

### 2. Specific Times Mode
Downloads spectrograms for specific timestamps defined in a JSON configuration file:

```bash
# First, create an example configuration file
python scripts/download_spectrograms.py --create-example-config

# Edit the generated example_specific_times.json file, then run:
python scripts/download_spectrograms.py --mode specific --config example_specific_times.json
```

**Configuration file format:**
```json
{
  "ICLISTENHF6020": [
    [2020, 10, 2, 12, 0, 0],
    [2020, 10, 2, 18, 30, 0],
    [2020, 10, 3, 6, 15, 0]
  ],
  "ANOTHER_DEVICE": [
    [2020, 10, 5, 9, 0, 0],
    [2020, 10, 5, 15, 45, 0]
  ]
}
```

Each timestamp is specified as: `[Year, Month, Day, Hour, Minute, Second]`

### 3. Date Range Mode
Downloads all available spectrograms within a specified date range:

```bash
python scripts/download_spectrograms.py --mode range --device ICLISTENHF6020 --start-date 2020 10 2 --end-date 2020 10 5
```

**Parameters:**
- `--device`: Device code
- `--start-date`: Start date as three integers (YEAR MONTH DAY)
- `--end-date`: End date as three integers (YEAR MONTH DAY)
- `--filetype`: File type to download (`png` or `mat`, default: `mat`)
- `--verbose`: (Optional) Show detailed ONC API messages

## Output Messages

The script provides clear, color-coded status messages:

- 🔄 **Progress**: Ongoing operations
- ✅ **Success**: Completed operations
- ⚠️ **Warning**: Non-critical issues
- ❌ **Error**: Critical errors
- ℹ️ **Info**: General information

### Verbose Mode

By default, ONC API warnings are suppressed for cleaner output. Use the `--verbose` flag to see all API messages:

```bash
python scripts/download_spectrograms.py --mode sampling --device ICLISTENHF6020 --start-date 2020 10 2 --threshold 25 --verbose
```

## File Organization

The script automatically creates the following directory structure:

```
data/
├── mat/  (or png/)
│   ├── processed/     # Successfully processed spectrograms
│   ├── rejects/       # Anomalous or problematic files
│   └── [temp files]   # Temporary downloads (cleaned up after processing)
```

## Output Files

- **Spectrograms**: Saved in the `processed/` directory
- **Anomaly logs**: 
  - `anomalous_files.txt`: List of files with anomalies
  - `anomalous_file_summary.txt`: Detailed anomaly descriptions

**Anomaly Detection**: Spectrograms are flagged as anomalous if any row has pixel intensity sums < 500 (too dark) or > 568,000 (too bright), indicating corrupted or saturated data.

## Common Device Codes

Some common ONC device codes you might use:
- `ICLISTENHF6020`: Hydrophone at specific location
- (Add other device codes as needed)

## Troubleshooting

1. **Invalid ONC Token**: Make sure your `.env` file contains a valid ONC API token
2. **Data Restrictions**: Some devices may have restricted data - contact datastewardship@oceannetworks.ca
3. **Network Issues**: The script will retry failed downloads automatically
4. **Disk Space**: Ensure you have sufficient disk space for the downloads
5. **Device Code**: Verify the device code exists and is accessible with your ONC account

## Examples

```bash
# Download 500 MAT spectrograms using sampling schedule (clean output)
python scripts/download_spectrograms.py --mode sampling --device ICLISTENHF6020 --start-date 2020 10 1 --threshold 500

# Download with verbose output to see all API messages
python scripts/download_spectrograms.py --mode sampling --device ICLISTENHF6020 --start-date 2020 10 1 --threshold 500 --verbose

# Download PNG files for specific times
python scripts/download_spectrograms.py --mode specific --config my_times.json --filetype png

# Download all spectrograms in October 2020
python scripts/download_spectrograms.py --mode range --device ICLISTENHF6020 --start-date 2020 10 1 --end-date 2020 10 31
```

## Sample Output

```
============================================================
 ONC SPECTROGRAM DOWNLOADER
============================================================

--- Loading Configuration ---
ℹ️ Data Directory: ./data
✅ ONC Token: ✓ Loaded
ℹ️ Verbose mode OFF - ONC warnings suppressed for cleaner output
ℹ️ Use --verbose flag to see detailed API messages

--- Initializing Downloader ---
✅ SpectrogramDownloader initialized

============================================================
 SAMPLING SCHEDULE MODE
============================================================
ℹ️ Device Code: ICLISTENHF6020
ℹ️ Start Date: 2020-10-2
ℹ️ Target Files: 25
ℹ️ File Type: MAT

--- Starting Download Process ---
🔄 Setting up directories...
🔄 Calculating sampling schedule...
✅ Found 5 time slots to download

--- Downloading Files ---
🔄 Progress: 6/25 files downloaded
🔄 Downloading batch 1/5: 2020-10-02 00:00:00
🔄 Processing downloaded files...
✅ Added 6 new files
```