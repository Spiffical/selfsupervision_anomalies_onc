#!/usr/bin/env python3
"""
Custom spectrogram generation from audio files.
Translates MATLAB spectrogram functionality to Python.
"""

import os
import numpy as np
import scipy.io
import scipy.signal
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import soundfile as sf
import librosa
from pathlib import Path
from typing import Union, Tuple, List, Optional
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from functools import lru_cache

# Thread lock for matplotlib operations (shared across instances)
_plot_lock = threading.Lock()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrintLogger:
    def info(self, msg, *args, **kwargs):
        print(msg % args if args else msg)
    def warning(self, msg, *args, **kwargs):
        print('WARNING:', msg % args if args else msg)
    def error(self, msg, *args, **kwargs):
        print('ERROR:', msg % args if args else msg)

class SpectrogramGenerator:
    """
    Generate spectrograms from audio files with configurable parameters.
    Based on MATLAB spectrogram computation with normalization and dB conversion.
    """
    
    def __init__(self, 
                 win_dur: float = 1.0,
                 overlap: float = 0.5,
                 freq_lims: Tuple[float, float] = (10, 10000),
                 colormap: str = 'turbo',
                 clim: Tuple[float, float] = (-60, 0),
                 log_freq: bool = True,
                 max_duration: Optional[float] = None,
                 clip_start: Optional[float] = None,
                 clip_end: Optional[float] = None,
                 use_gpu: bool = False,
                 quiet: bool = False,
                 use_logging: bool = True):
        """
        Initialize spectrogram generator with parameters from MATLAB code.
        
        Args:
            win_dur: Window duration in seconds (controls FFT size: NFFT = win_dur * fs)
            overlap: Overlap ratio between adjacent windows (0-1), higher = smoother time axis
            freq_lims: Frequency limits for plotting [Hz]
            colormap: Matplotlib colormap name
            clim: Color axis limits [dB]
            log_freq: Whether to use logarithmic frequency scale
            max_duration: Maximum duration to process in seconds (None = full file)
            clip_start: Optional start time (seconds) to trim from beginning of audio
            clip_end: Optional end time (seconds) to stop processing; must be > clip_start
            use_gpu: If True and CUDA is available, compute spectrogram on GPU using torch.stft
            quiet: If True, suppress logger noise (only minimal prints for progress bar)
            use_logging: If False, fall back to stdout printing (avoids notebook logging friction)
        """
        self.win_dur = win_dur
        self.overlap = overlap
        self.freq_lims = freq_lims
        self.colormap = colormap
        self.clim = clim
        self.log_freq = log_freq
        self.max_duration = max_duration
        self.clip_start = clip_start
        self.clip_end = clip_end
        self.use_gpu = use_gpu
        self.quiet = quiet
        self.log = logger if use_logging else PrintLogger()
        
    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load audio file supporting multiple formats.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        try:
            # Try soundfile first (supports FLAC, WAV, etc.)
            audio_data, sample_rate = sf.read(str(audio_path))
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
                
        except Exception as e:
            try:
                # Fallback to librosa for other formats
                audio_data, sample_rate = librosa.load(str(audio_path), sr=None)
            except Exception as e2:
                raise RuntimeError(f"Could not load audio file {audio_path}: {e}, {e2}")
        
        # Optional clipping window
        total_samples = len(audio_data)
        original_duration = total_samples / sample_rate
        start_idx = 0
        end_idx = total_samples
        if self.clip_start is not None:
            start_idx = max(0, int(self.clip_start * sample_rate))
        if self.clip_end is not None:
            end_idx = min(total_samples, int(self.clip_end * sample_rate))
        if end_idx <= start_idx:
            raise ValueError(f"Invalid clip window: start={self.clip_start}s end={self.clip_end}s for {audio_path.name}")
        if start_idx != 0 or end_idx != total_samples:
            audio_data = audio_data[start_idx:end_idx]
            original_duration = len(audio_data) / sample_rate
            if not self.quiet:
                self.log.info(f"Clipped audio to {self.clip_start or 0:.2f}s–{self.clip_end or original_duration:.2f}s ({audio_path.name})")

        # Truncate to max_duration if specified
        if self.max_duration is not None and original_duration > self.max_duration:
            max_samples = int(self.max_duration * sample_rate)
            audio_data = audio_data[:max_samples]
            if not self.quiet:
                self.log.info(f"Truncated audio from {original_duration:.2f}s to {self.max_duration:.2f}s")

        if not self.quiet:
            self.log.info(f"Loaded audio: {audio_path.name}, duration: {len(audio_data)/sample_rate:.2f}s, sr: {sample_rate}Hz")
        return audio_data, sample_rate
    
    def compute_spectrogram(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram following MATLAB implementation.
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate in Hz
            
        Returns:
            Tuple of (frequencies, times, power_spectrogram, normalized_db)
        """
        # Calculate NFFT size based on window duration and sample rate
        nfft = int(self.win_dur * sample_rate)
        
        # Calculate overlap in samples
        noverlap = int(self.overlap * nfft)
        
        # Create Hann window
        window = scipy.signal.windows.hann(nfft)

        use_torch = False
        torch = None
        if self.use_gpu:
            try:
                import torch  # type: ignore
                use_torch = torch.cuda.is_available()
            except Exception:
                use_torch = False
        
        if use_torch:
            # GPU path using torch.stft
            device = torch.device('cuda')
            audio_t = torch.from_numpy(audio_data.astype(np.float32, copy=False)).to(device)
            win_t = torch.from_numpy(window.astype(np.float32, copy=False)).to(device)
            hop_length = nfft - noverlap
            spec = torch.stft(
                audio_t,
                n_fft=nfft,
                hop_length=hop_length,
                window=win_t,
                center=False,
                return_complex=True,
                onesided=True,
            )
            power = (spec.real**2 + spec.imag**2).cpu().numpy()
            # Torch stft returns shape [freq, time]; build freq/time axes to match scipy
            frequencies = np.linspace(0, sample_rate/2, power.shape[0])
            times = np.arange(power.shape[1]) * (hop_length / sample_rate)
            Sxx = power
        else:
            # CPU path using scipy (equivalent to MATLAB spectrogram with 'psd')
            frequencies, times, Sxx = scipy.signal.spectrogram(
                audio_data,
                fs=sample_rate,
                window=window,
                noverlap=noverlap,
                nfft=nfft,
                scaling='density',  # Power spectral density
                mode='psd'
            )
        
        # Normalize and convert to dB (following MATLAB: 10*log10(abs(P./max(P,[],'all'))))
        max_power = np.max(np.abs(Sxx))
        if max_power > 0:
            normalized_power = np.abs(Sxx) / max_power
            # Avoid log(0) by setting minimum value
            normalized_power = np.maximum(normalized_power, 1e-10)
            power_db_norm = 10 * np.log10(normalized_power)
        else:
            power_db_norm = np.full_like(Sxx, -100.0)  # Very low dB value
        
        if not self.quiet:
            self.log.info(f"Spectrogram computed: {frequencies.shape[0]} freq bins, {times.shape[0]} time frames")
        return frequencies, times, Sxx, power_db_norm
    
    def plot_spectrogram(self, frequencies: np.ndarray, times: np.ndarray, 
                        power_db_norm: np.ndarray, title: str = "Spectrogram",
                        save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Plot spectrogram following MATLAB visualization.
        
        Args:
            frequencies: Frequency array
            times: Time array
            power_db_norm: Normalized power in dB
            title: Plot title
            save_path: Optional path to save plot
            
        Returns:
            matplotlib Figure object
        """
        # Use thread lock for all matplotlib operations
        with _plot_lock:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create meshgrid for pcolor
            T, F = np.meshgrid(times, frequencies)
            
            # Plot spectrogram
            pcm = ax.pcolor(T, F, power_db_norm, 
                           cmap=self.colormap, 
                           shading='auto',
                           vmin=self.clim[0], 
                           vmax=self.clim[1])
            
            # Set frequency limits
            ax.set_ylim(self.freq_lims)
            
            # Set log scale if requested
            if self.log_freq:
                ax.set_yscale('log')
            
            # Labels and title
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Frequency [Hz]')
            ax.set_title(title)
            
            # Colorbar
            cbar = plt.colorbar(pcm, ax=ax)
            cbar.set_label('PSD re max [dB]')
            
            # Save if requested
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if not self.quiet:
                self.log.info(f"Spectrogram plot saved: {save_path}")
        
        return fig
    
    def save_matlab_format(self, frequencies: np.ndarray, times: np.ndarray,
                          power_spectrogram: np.ndarray, power_db_norm: np.ndarray,
                          save_path: Union[str, Path]) -> None:
        """
        Save spectrogram data in MATLAB format.
        
        Args:
            frequencies: Frequency array
            times: Time array  
            power_spectrogram: Raw power spectrogram
            power_db_norm: Normalized power in dB
            save_path: Path to save .mat file
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save in MATLAB format (following original MATLAB code)
        scipy.io.savemat(
            save_path,
            {
                'F': frequencies,
                'T': times,
                'P': power_spectrogram,
                'PdB_norm': power_db_norm
            }
        )
        if not self.quiet:
            self.log.info(f"MATLAB data saved: {save_path}")
    
    def process_single_file(self, audio_path: Union[str, Path], 
                           save_dir: Union[str, Path],
                           save_plot: bool = True,
                           save_mat: bool = True) -> dict:
        """
        Process a single audio file to generate spectrogram.
        
        Args:
            audio_path: Path to input audio file
            save_dir: Directory to save outputs
            save_plot: Whether to save PNG plot
            save_mat: Whether to save MAT file
            
        Returns:
            Dictionary with processing results
        """
        audio_path = Path(audio_path)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load audio
        audio_data, sample_rate = self.load_audio(audio_path)
        
        # Compute spectrogram
        frequencies, times, power_spec, power_db_norm = self.compute_spectrogram(audio_data, sample_rate)
        
        # Create output filenames
        base_name = audio_path.stem
        mat_path = save_dir / f"{base_name}.mat"
        png_path = save_dir / f"{base_name}.png"
        
        # Save outputs
        results = {
            'audio_file': str(audio_path),
            'frequencies': frequencies,
            'times': times,
            'power_spectrogram': power_spec,
            'power_db_norm': power_db_norm,
            'sample_rate': sample_rate,
            'duration': len(audio_data) / sample_rate
        }
        
        if save_mat:
            self.save_matlab_format(frequencies, times, power_spec, power_db_norm, mat_path)
            results['mat_file'] = str(mat_path)
        
        if save_plot:
            fig = self.plot_spectrogram(frequencies, times, power_db_norm, 
                                       title=f"Spectrogram - {base_name}",
                                       save_path=png_path)
            plt.close(fig)  # Close to free memory
            results['png_file'] = str(png_path)
        
        return results
    
    def process_directory(self, input_dir: Union[str, Path], 
                         save_dir: Union[str, Path],
                         file_extensions: List[str] = ['.wav', '.flac', '.mp3', '.m4a'],
                         save_plot: bool = True,
                         save_mat: bool = True) -> List[dict]:
        """
        Process all audio files in a directory.
        
        Args:
            input_dir: Directory containing audio files
            save_dir: Directory to save outputs
            file_extensions: List of audio file extensions to process
            save_plot: Whether to save PNG plots
            save_mat: Whether to save MAT files
            
        Returns:
            List of processing results for each file
        """
        input_dir = Path(input_dir)
        save_dir = Path(save_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all audio files
        audio_files = []
        for ext in file_extensions:
            audio_files.extend(input_dir.glob(f"*{ext}"))
            audio_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        if not audio_files:
            logger.warning(f"No audio files found in {input_dir} with extensions {file_extensions}")
            return []
        
        if not self.quiet:
            self.log.info(f"Processing {len(audio_files)} audio files from {input_dir}")
        
        results = []
        max_workers = min(8, os.cpu_count() or 4)
        total = len(audio_files)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_single_file, audio_file, save_dir, save_plot, save_mat): audio_file
                for audio_file in audio_files
            }
            completed = 0
            for future in as_completed(future_to_file):
                audio_file = future_to_file[future]
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                    if not self.quiet:
                        self.log.info(f"Processed {completed}/{total}: {audio_file.name}")
                except Exception as e:
                    if not self.quiet:
                        self.log.error(f"Error processing {audio_file}: {e}")
                    results.append({
                        'audio_file': str(audio_file),
                        'error': str(e)
                    })
                # Lightweight progress bar to stdout
                bar_len = 20
                filled = int(bar_len * completed / total)
                bar = '#' * filled + '-' * (bar_len - filled)
                sys.stdout.write(f"\rSpectrograms: [{bar}] {completed}/{total}")
                sys.stdout.flush()
            sys.stdout.write("\n")
        
        if not self.quiet:
            self.log.info(f"Completed processing {len(results)} files")
        return results 
