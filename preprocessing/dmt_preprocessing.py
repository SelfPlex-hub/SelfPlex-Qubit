# -*- coding: utf-8 -*-
"""
DMT EEG Preprocessing Pipeline
==============================

Publication-quality preprocessing for the Pallavicini et al. (2021) inhaled DMT dataset.

Dataset: doi.org/10.5281/zenodo.3992359

STATUS: Work in progress - requires manual inspection steps
        Originally developed as Colab notebook, being adapted for local use

Publication standards (from original paper):
- Bandpass: 1-90 Hz (we use 0.5-45 Hz for TDA focus)
- Notch: 47.5-52.5 Hz (50 Hz mains)
- Bad channels: auto-detect + manual review, then interpolate
- Bad segments: auto-detect + manual review, then reject
- ICA: Infomax, manual component selection with documented reasons
- Subject exclusion if >30% channels bad or excessive epoch rejection

Workflow:
1. Load raw BDF
2. Drop non-EEG channels
3. View all channels, mark bad channels
4. Apply filters (bandpass + notch)
5. Apply montage + average reference
6. Interpolate bad channels
7. Mark bad segments
8. Run ICA, inspect and reject with documentation
9. Before/after overlay comparison
10. Free crop to segment of interest
11. Save + log
12. Auto-advance to next file
"""

# =============================================================================
# CONFIGURATION - EDIT THESE PATHS FOR YOUR SYSTEM
# =============================================================================

from pathlib import Path

# Data paths - UPDATE THESE FOR YOUR SYSTEM
DATA_ROOT = Path(r"C:\Users\bwilk\Downloads\dmt_inhalation_study3992359")  # Where raw BDF files are
OUTPUT_DIR = DATA_ROOT / "cleaned"  # Where cleaned files will be saved

# Alternative: Use relative paths
# DATA_ROOT = Path("./data/dmt_raw")
# OUTPUT_DIR = Path("./data/dmt_cleaned")

# =============================================================================
# FILTER SETTINGS (publication standard)
# =============================================================================

HP_CUTOFF = 0.5        # Hz highpass
LP_CUTOFF = 45.0       # Hz lowpass
NOTCH_FREQ = 50.0      # Hz (EU mains - change to 60.0 for US)
NOTCH_WIDTH = 2.5      # Hz (47.5-52.5 Hz notch)

# Montage
MONTAGE_NAME = "standard_1005"  # includes mastoids

# Publication-worthy rejection criteria
BAD_CHANNEL_CRITERIA = {
    'flat_threshold': 0.5e-6,      # V² - variance below this = flat
    'flat_duration': 1.0,           # seconds
    'high_amplitude': 150e-6,       # V - amplitude above this = bad
    'neighbor_correlation': 0.4,    # correlation below this = bad
}

BAD_SEGMENT_CRITERIA = {
    'amplitude_threshold': 100e-6,  # V - mark if exceeded
    'channel_percentage': 0.10,     # 10% of channels bad = mark segment
    'min_duration': 0.2,            # seconds minimum segment to mark
}

# ICA
ICA_N_COMPONENTS = 0.99  # variance explained
ICA_METHOD = "infomax"   # same as original paper
ICA_RANDOM_STATE = 97

# File pattern - process DMT files (not EC/EO baselines)
FILE_PATTERN = "*DMT*.bdf"

# =============================================================================
# IMPORTS
# =============================================================================

import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from mne.preprocessing import ICA
import pandas as pd
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

mne.set_log_level('WARNING')

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("DMT Preprocessing Pipeline")
print("=" * 70)
print(f"MNE version: {mne.__version__}")
print(f"Data directory: {DATA_ROOT}")
print(f"Output directory: {OUTPUT_DIR}")

# =============================================================================
# FIND ALL FILES TO PROCESS
# =============================================================================

all_files = sorted(DATA_ROOT.glob(FILE_PATTERN))

# Check for already processed files
processed = []
to_process = []

for f in all_files:
    clean_file = OUTPUT_DIR / f"{f.stem}_clean_raw.fif"
    if clean_file.exists():
        processed.append(f.name)
    else:
        to_process.append(f)

print(f"\nTotal DMT files found: {len(all_files)}")
print(f"Already processed: {len(processed)}")
print(f"To process: {len(to_process)}")

if to_process:
    print(f"\nFiles to process:")
    for i, f in enumerate(to_process):
        print(f"  {i+1}. {f.name}")
else:
    print("\n*** All files already processed! ***")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def plot_all_channels(raw, title="EEG", start=0, duration=30, scalings=50e-6):
    """
    Plot all channels in a scrollable figure.
    Uses MNE's plot but with better defaults.
    """
    fig = raw.plot(
        start=start,
        duration=duration,
        n_channels=len(raw.ch_names),
        scalings={'eeg': scalings},
        title=title,
        show=True,
        block=True
    )
    return fig


def plot_channels_grid(raw, start=0, duration=10, title="EEG Channels"):
    """
    Plot all channels in a static grid - easier to screenshot.
    """
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    n_channels = len(ch_names)

    start_idx = int(start * sfreq)
    end_idx = int((start + duration) * sfreq)
    end_idx = min(end_idx, data.shape[1])

    data_segment = data[:, start_idx:end_idx]
    times = np.arange(data_segment.shape[1]) / sfreq + start

    # Calculate grid size
    n_cols = 4
    n_rows = int(np.ceil(n_channels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 1.5), sharex=True)
    axes = axes.flatten()

    for i in range(n_channels):
        ax = axes[i]
        ax.plot(times, data_segment[i] * 1e6, 'k-', linewidth=0.5)  # Convert to µV
        ax.set_ylabel(ch_names[i], fontsize=8, rotation=0, ha='right')
        ax.set_ylim([-150, 150])  # µV
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(n_channels, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"{title} | Start: {start}s | Duration: {duration}s", fontsize=12)
    plt.tight_layout()
    plt.show()

    return fig


def auto_detect_bad_channels(raw, criteria):
    """
    Automatically detect bad channels based on criteria.
    Returns list of bad channel names and reasons.
    """
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names

    bad_channels = {}

    for i, ch in enumerate(ch_names):
        reasons = []
        ch_data = data[i]

        # Check for flat channel
        window_samples = int(criteria['flat_duration'] * sfreq)
        for start in range(0, len(ch_data) - window_samples, window_samples):
            window = ch_data[start:start + window_samples]
            if np.var(window) < criteria['flat_threshold']:
                reasons.append('flat')
                break

        # Check for high amplitude
        if np.max(np.abs(ch_data)) > criteria['high_amplitude']:
            reasons.append('high_amplitude')

        # Check neighbor correlation (simplified - compare to mean of all)
        other_data = np.delete(data, i, axis=0)
        mean_other = np.mean(other_data, axis=0)
        corr = np.corrcoef(ch_data, mean_other)[0, 1]
        if corr < criteria['neighbor_correlation']:
            reasons.append(f'low_correlation({corr:.2f})')

        if reasons:
            bad_channels[ch] = reasons

    return bad_channels


def auto_detect_bad_segments(raw, criteria):
    """
    Automatically detect bad time segments.
    Returns list of (start, end) tuples in seconds.
    """
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    n_channels = len(raw.ch_names)
    
    threshold = criteria['amplitude_threshold']
    min_bad_channels = int(n_channels * criteria['channel_percentage'])
    min_samples = int(criteria['min_duration'] * sfreq)
    
    # Find samples where threshold is exceeded
    bad_mask = np.abs(data) > threshold
    bad_channel_count = np.sum(bad_mask, axis=0)
    bad_samples = bad_channel_count >= min_bad_channels
    
    # Find contiguous bad segments
    segments = []
    in_segment = False
    start_idx = 0
    
    for i, is_bad in enumerate(bad_samples):
        if is_bad and not in_segment:
            start_idx = i
            in_segment = True
        elif not is_bad and in_segment:
            if i - start_idx >= min_samples:
                segments.append((start_idx / sfreq, i / sfreq))
            in_segment = False
    
    # Handle segment at end
    if in_segment and len(bad_samples) - start_idx >= min_samples:
        segments.append((start_idx / sfreq, len(bad_samples) / sfreq))
    
    return segments


def plot_before_after(raw_before, raw_after, start=0, duration=30, channels=None):
    """
    Overlay before/after ICA for comparison.
    """
    if channels is None:
        channels = raw_before.ch_names[:8]
    
    fig, axes = plt.subplots(len(channels), 1, figsize=(15, 2*len(channels)), sharex=True)
    
    sfreq = raw_before.info['sfreq']
    start_idx = int(start * sfreq)
    end_idx = int((start + duration) * sfreq)
    times = np.arange(end_idx - start_idx) / sfreq + start
    
    for i, ch in enumerate(channels):
        ch_idx = raw_before.ch_names.index(ch)
        
        data_before = raw_before.get_data()[ch_idx, start_idx:end_idx] * 1e6
        data_after = raw_after.get_data()[ch_idx, start_idx:end_idx] * 1e6
        
        axes[i].plot(times, data_before, 'r-', alpha=0.5, linewidth=1, label='Before ICA')
        axes[i].plot(times, data_after, 'b-', alpha=0.7, linewidth=1, label='After ICA')
        axes[i].set_ylabel(ch, fontsize=10)
        axes[i].set_ylim([-100, 100])
        axes[i].grid(True, alpha=0.3)
        
        if i == 0:
            axes[i].legend(loc='upper right')
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Before (red) vs After (blue) ICA', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("NOTE: This script requires interactive inspection.")
    print("For full preprocessing, run in Jupyter/IPython or use the Colab notebook.")
    print("=" * 70)
    
    if not to_process:
        print("\nNo files to process.")
    else:
        print(f"\nReady to process {len(to_process)} files.")
        print("Import this module in Jupyter and call the processing functions.")
