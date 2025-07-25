#!/usr/bin/env python3
"""
Example: Custom Spectrogram Generation

This example demonstrates how to use the custom spectrogram generation functionality
to create spectrograms from FLAC audio files downloaded from ONC.

The script shows both programmatic usage and integration with the existing project structure.
"""

import sys
import os
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.audio import SpectrogramGenerator
from utils.data.config_utils import DatasetConfig

def example_basic_usage():
    """Basic example of generating spectrograms from a directory of audio files."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)
    
    # Create spectrogram generator with default parameters (matching MATLAB code)
    generator = SpectrogramGenerator(
        win_dur=1.0,         # 1 second window
        overlap=0.5,         # 50% overlap
        freq_lims=(10, 10000),  # 10 Hz to 10 kHz
        colormap='turbo',    # Turbo colormap (like MATLAB)
        clim=(-60, 0),       # Color limits in dB
        log_freq=True        # Logarithmic frequency scale
    )
    
    # Example: Process a directory of FLAC files
    input_dir = "data/ICLISTENHF6020/sampling_2021-01-01_to_2021-01-31/flac"
    output_dir = "data/ICLISTENHF6020/sampling_2021-01-01_to_2021-01-31/custom_spectrograms"
    
    if Path(input_dir).exists():
        print(f"Processing audio files from: {input_dir}")
        print(f"Saving spectrograms to: {output_dir}")
        
        # Process all audio files in the directory
        results = generator.process_directory(
            input_dir=input_dir,
            save_dir=output_dir,
            save_mat=True,    # Save MATLAB .mat files
            save_plot=True    # Save PNG plots
        )
        
        print(f"Processed {len(results)} files")
        
        # Show some results
        successful = [r for r in results if 'error' not in r]
        errors = [r for r in results if 'error' in r]
        
        print(f"Successful: {len(successful)}")
        print(f"Errors: {len(errors)}")
        
        if successful:
            print(f"First result sample rate: {successful[0]['sample_rate']} Hz")
            print(f"First result duration: {successful[0]['duration']:.2f} seconds")
    else:
        print(f"Directory not found: {input_dir}")
        print("This example requires FLAC files from ONC downloads.")

def example_custom_parameters():
    """Example with custom spectrogram parameters for different analysis needs."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Parameters")
    print("=" * 60)
    
    # High-resolution spectrogram for detailed analysis
    high_res_generator = SpectrogramGenerator(
        win_dur=0.5,         # Shorter window for better time resolution
        overlap=0.75,        # Higher overlap for smoother spectrogram
        freq_lims=(1, 50000), # Wider frequency range
        colormap='viridis',   # Different colormap
        clim=(-80, 0),       # Wider dynamic range
        log_freq=True
    )
    
    # Low-frequency focused analysis (for whale calls, etc.)
    low_freq_generator = SpectrogramGenerator(
        win_dur=4.0,         # Longer window for better frequency resolution
        overlap=0.9,         # Very high overlap
        freq_lims=(1, 1000), # Focus on low frequencies
        colormap='plasma',
        clim=(-60, 0),
        log_freq=True
    )
    
    print("High-resolution generator configuration:")
    print(f"  Window duration: {high_res_generator.win_dur}s")
    print(f"  Overlap: {high_res_generator.overlap}")
    print(f"  Frequency range: {high_res_generator.freq_lims} Hz")
    
    print("\nLow-frequency generator configuration:")
    print(f"  Window duration: {low_freq_generator.win_dur}s")
    print(f"  Overlap: {low_freq_generator.overlap}")
    print(f"  Frequency range: {low_freq_generator.freq_lims} Hz")

def example_single_file():
    """Example of processing a single audio file."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Single File Processing")
    print("=" * 60)
    
    generator = SpectrogramGenerator()
    
    # Find any FLAC file in the project
    data_dir = Path("data")
    flac_files = list(data_dir.rglob("*.flac"))
    
    if flac_files:
        audio_file = flac_files[0]
        output_dir = audio_file.parent / "example_spectrograms"
        
        print(f"Processing single file: {audio_file}")
        print(f"Output directory: {output_dir}")
        
        try:
            result = generator.process_single_file(
                audio_path=audio_file,
                save_dir=output_dir,
                save_plot=True,
                save_mat=True
            )
            
            print("Processing successful!")
            print(f"Sample rate: {result['sample_rate']} Hz")
            print(f"Duration: {result['duration']:.2f} seconds")
            print(f"Frequency bins: {len(result['frequencies'])}")
            print(f"Time frames: {len(result['times'])}")
            
            if 'mat_file' in result:
                print(f"MATLAB file: {result['mat_file']}")
            if 'png_file' in result:
                print(f"PNG plot: {result['png_file']}")
                
        except Exception as e:
            print(f"Error processing file: {e}")
    else:
        print("No FLAC files found in project data directory.")

def example_config_integration():
    """Example of loading parameters from the project configuration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Configuration Integration")
    print("=" * 60)
    
    try:
        # Load configuration from project config file
        config = DatasetConfig()
        
        # Extract custom spectrogram settings if they exist
        if 'custom_spectrograms' in config.config:
            spec_config = config.config['custom_spectrograms']
            
            # Create generator from config
            generator = SpectrogramGenerator(
                win_dur=spec_config.get('window_duration', 1.0),
                overlap=spec_config.get('overlap', 0.5),
                freq_lims=(
                    spec_config.get('frequency_limits', {}).get('min', 10),
                    spec_config.get('frequency_limits', {}).get('max', 10000)
                ),
                colormap=spec_config.get('colormap', 'turbo'),
                clim=(
                    spec_config.get('color_limits', {}).get('min', -60),
                    spec_config.get('color_limits', {}).get('max', 0)
                ),
                log_freq=spec_config.get('log_frequency', True)
            )
            
            print("Loaded configuration from config/dataset_config.yaml:")
            print(f"  Window duration: {generator.win_dur}s")
            print(f"  Overlap: {generator.overlap}")
            print(f"  Frequency limits: {generator.freq_lims} Hz")
            print(f"  Colormap: {generator.colormap}")
            print(f"  Color limits: {generator.clim} dB")
            print(f"  Log frequency scale: {generator.log_freq}")
            
        else:
            print("Custom spectrogram configuration not found in config file.")
            
    except Exception as e:
        print(f"Error loading configuration: {e}")

def main():
    """Run all examples."""
    print("Custom Spectrogram Generation Examples")
    print("Based on MATLAB code translation")
    
    # Run examples
    example_basic_usage()
    example_custom_parameters()
    example_single_file()
    example_config_integration()
    
    print("\n" + "=" * 60)
    print("USAGE SUMMARY")
    print("=" * 60)
    print("To generate custom spectrograms:")
    print()
    print("1. Interactive mode (recommended):")
    print("   python scripts/generate_spectrograms.py")
    print()
    print("2. Process a directory:")
    print("   python scripts/generate_spectrograms.py --input-dir data/DEVICE/flac/")
    print()
    print("3. Custom parameters:")
    print("   python scripts/generate_spectrograms.py \\")
    print("     --input-dir data/DEVICE/flac/ \\")
    print("     --win-dur 2.0 --overlap 0.75 \\")
    print("     --freq-min 5 --freq-max 20000")
    print()
    print("4. Single file:")
    print("   python scripts/generate_spectrograms.py --input-file audio.flac")

if __name__ == "__main__":
    main() 