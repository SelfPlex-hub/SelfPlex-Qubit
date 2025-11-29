# -*- coding: utf-8 -*-
"""
TDA Engine - Topological Data Analysis for EEG
==============================================

Antientropic Repeller Theory of Consciousness

STATUS: Work in progress - core functions implemented, full pipeline needs testing

Purpose: Extract topological signatures from cleaned EEG

Pipeline:
1. Takens embedding (m=10, τ derived via AMI)
2. Vietoris-Rips filtration → Persistent homology H1
3. B(c,t) = sum of 1-cycle lifetimes
4. E_B(c,t) = B(c,t) / log₁₊P(c,t) — topological efficiency
5. Lock_E_B(c|s) = mean(E_B) / (1 + var(E_B) + jitter(E_B)) — topological locking

Output: Per-channel, per-window topological features ready for consciousness analysis
"""

# =============================================================================
# CONFIGURATION - EDIT THESE PATHS FOR YOUR SYSTEM
# =============================================================================

from pathlib import Path

# Data paths - UPDATE THESE FOR YOUR SYSTEM
DATA_ROOT = Path('./data/cleaned')      # Where cleaned EEG files are
OUTPUT_ROOT = Path('./data/tda_features')  # Where TDA features will be saved

# Create output directory
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# =============================================================================
# TDA PARAMETERS
# =============================================================================

TDA_PARAMS = {
    'window_length': 5.0,       # seconds
    'window_overlap': 0.5,      # 50% overlap
    'embedding_dim': 10,        # m = 10 (Takens dimension)
    'tau_method': 'ami',        # 'ami' for delay selection
    'max_tau': 50,              # Maximum delay to search
    'ripser_maxdim': 1,         # Compute H0 and H1 (we focus on H1)
    'ripser_thresh': 2.0,       # Maximum filtration value
    'min_lifetime': 0.01,       # Minimum persistence to count
    'n_jobs': -1                # Parallel processing (-1 = all cores)
}

# Channels of interest (will compute for all, but highlight these)
ROI_CHANNELS = ['CP6', 'CP5', 'P4', 'TP8', 'C4']  # Right centro-parietal ROI
TEMPORAL_PORTS = ['TP9', 'TP10']  # Temporal write ports

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# MNE for EEG loading
import mne

# Ripser for persistent homology
from ripser import ripser
from persim import plot_diagrams

# Parallel processing
from joblib import Parallel, delayed
from tqdm.auto import tqdm

print("TDA Engine loaded")
print(f"Data root: {DATA_ROOT}")
print(f"Output root: {OUTPUT_ROOT}")

# =============================================================================
# TAKENS EMBEDDING FUNCTIONS
# =============================================================================

def compute_ami(x, max_lag=50):
    """
    Compute Average Mutual Information to find optimal delay τ.

    Parameters
    ----------
    x : array
        Time series
    max_lag : int
        Maximum lag to test

    Returns
    -------
    tau_opt : int
        Optimal delay (first minimum of AMI)
    ami_values : array
        AMI values for each lag
    """
    ami_values = []

    for lag in range(1, max_lag + 1):
        # Create delayed versions
        x1 = x[:-lag]
        x2 = x[lag:]

        # Compute 2D histogram
        bins = int(np.sqrt(len(x1)))
        hist_2d, _, _ = np.histogram2d(x1, x2, bins=bins)

        # Normalize to get probabilities
        p_xy = hist_2d / np.sum(hist_2d)
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)

        # Compute mutual information
        mi = 0
        for i in range(len(p_x)):
            for j in range(len(p_y)):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))

        ami_values.append(mi)

    ami_values = np.array(ami_values)

    # Find first local minimum
    derivatives = np.diff(ami_values)
    minima = np.where((derivatives[:-1] < 0) & (derivatives[1:] > 0))[0] + 1

    if len(minima) > 0:
        tau_opt = minima[0] + 1  # +1 because lag starts at 1
    else:
        # If no minimum found, use lag where AMI drops to 1/e of initial
        threshold = ami_values[0] / np.e
        tau_opt = np.where(ami_values < threshold)[0][0] + 1 if np.any(ami_values < threshold) else 10

    return tau_opt, ami_values


def takens_embedding(x, m, tau):
    """
    Create Takens time-delay embedding.

    Parameters
    ----------
    x : array
        Time series
    m : int
        Embedding dimension
    tau : int
        Time delay

    Returns
    -------
    embedded : array, shape (n_points, m)
        Embedded point cloud
    """
    n = len(x) - (m - 1) * tau
    embedded = np.zeros((n, m))

    for i in range(m):
        embedded[:, i] = x[i * tau:i * tau + n]

    return embedded


def false_nearest_neighbors(x, m=6, tau=10, rtol=15.0, atol=2.0):
    """
    Compute False Nearest Neighbors to verify embedding dimension.

    Parameters
    ----------
    x : array
        Time series
    m : int
        Embedding dimension
    tau : int
        Time delay
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance for distance increase

    Returns
    -------
    fnn_pct : float
        Percentage of false nearest neighbors
    """
    # Create m-dimensional embedding
    n = len(x) - (m - 1) * tau
    embedded_m = np.zeros((n, m))
    for i in range(m):
        embedded_m[:, i] = x[i * tau:i * tau + n]

    # Create (m+1)-dimensional embedding
    n_plus = len(x) - m * tau
    embedded_m_plus = np.zeros((n_plus, m + 1))
    for i in range(m + 1):
        embedded_m_plus[:, i] = x[i * tau:i * tau + n_plus]

    # Truncate m-dimensional to match
    embedded_m = embedded_m[:n_plus]

    false_neighbors = 0
    total_neighbors = 0

    # For each point, find nearest neighbor in m dimensions
    for i in range(min(500, len(embedded_m))):  # Limit for speed
        # Compute distances in m dimensions
        distances_m = np.sqrt(np.sum((embedded_m - embedded_m[i])**2, axis=1))
        distances_m[i] = np.inf  # Exclude self

        # Find nearest neighbor
        nn_idx = np.argmin(distances_m)
        dist_m = distances_m[nn_idx]

        # Compute distance in (m+1) dimensions
        dist_m_plus = np.sqrt(np.sum((embedded_m_plus[i] - embedded_m_plus[nn_idx])**2))

        # Check if false neighbor
        if dist_m > 0:
            increase_ratio = (dist_m_plus - dist_m) / dist_m
            if increase_ratio > rtol or dist_m_plus > atol:
                false_neighbors += 1
            total_neighbors += 1

    fnn_pct = 100.0 * false_neighbors / total_neighbors if total_neighbors > 0 else 0

    return fnn_pct


# =============================================================================
# PERSISTENT HOMOLOGY FUNCTIONS
# =============================================================================

def compute_persistent_homology(point_cloud, maxdim=1, thresh=2.0):
    """
    Compute persistent homology using Ripser.

    Parameters
    ----------
    point_cloud : array, shape (n_points, n_dimensions)
        Embedded point cloud
    maxdim : int
        Maximum homological dimension (1 = compute H0 and H1)
    thresh : float
        Maximum filtration value

    Returns
    -------
    result : dict
        Ripser output containing persistence diagrams
    """
    result = ripser(
        point_cloud,
        maxdim=maxdim,
        thresh=thresh
    )

    return result


def extract_betti1_lifetimes(dgm, min_lifetime=0.01):
    """
    Extract 1-cycle lifetimes from persistence diagram.

    Parameters
    ----------
    dgm : array, shape (n_features, 2)
        Persistence diagram for H1 (birth, death pairs)
    min_lifetime : float
        Minimum persistence to include

    Returns
    -------
    lifetimes : array
        Array of lifetimes (death - birth) for each 1-cycle
    total_persistence : float
        Sum of all lifetimes (B(c,t) metric)
    n_cycles : int
        Number of significant 1-cycles
    """
    if len(dgm) == 0:
        return np.array([]), 0.0, 0

    lifetimes = dgm[:, 1] - dgm[:, 0]
    
    # Filter by minimum lifetime
    significant = lifetimes >= min_lifetime
    lifetimes = lifetimes[significant]

    total_persistence = np.sum(lifetimes)
    n_cycles = len(lifetimes)

    return lifetimes, total_persistence, n_cycles


def compute_topological_efficiency(B_ct, P_ct, epsilon=1e-10):
    """
    Compute topological efficiency E_B = B(c,t) / log(1 + P(c,t))

    Parameters
    ----------
    B_ct : float
        Total Betti-1 persistence (sum of 1-cycle lifetimes)
    P_ct : float
        Spectral power in the window

    Returns
    -------
    E_B : float
        Topological efficiency
    """
    E_B = B_ct / (np.log(1 + P_ct) + epsilon)
    return E_B


def compute_topological_locking(E_B_series):
    """
    Compute topological locking metric.
    
    Lock_E_B = mean(E_B) / (1 + var(E_B) + jitter(E_B))
    
    where jitter = variance of the derivative of E_B

    Parameters
    ----------
    E_B_series : array
        Time series of E_B values across windows

    Returns
    -------
    Lock_E_B : float
        Topological locking value
    jitter : float
        Jitter (variance of dE_B/dt)
    """
    if len(E_B_series) < 3:
        return 0.0, 0.0

    mean_E_B = np.mean(E_B_series)
    var_E_B = np.var(E_B_series)
    
    # Jitter = variance of derivative
    dE_B = np.diff(E_B_series)
    jitter = np.var(dE_B)

    Lock_E_B = mean_E_B / (1 + var_E_B + jitter)

    return Lock_E_B, jitter


# =============================================================================
# WINDOW PROCESSING
# =============================================================================

def process_single_window(window_data, sfreq, params):
    """
    Process a single time window: embedding → TDA → features.

    Parameters
    ----------
    window_data : array
        EEG data for this window (1D)
    sfreq : float
        Sampling frequency
    params : dict
        TDA parameters

    Returns
    -------
    features : dict
        Extracted features for this window
    """
    # Compute optimal tau via AMI
    tau_opt, _ = compute_ami(window_data, max_lag=params['max_tau'])

    # Takens embedding
    embedded = takens_embedding(
        window_data,
        m=params['embedding_dim'],
        tau=tau_opt
    )

    # Subsample if too many points (for computational efficiency)
    max_points = 1000
    if len(embedded) > max_points:
        indices = np.linspace(0, len(embedded)-1, max_points, dtype=int)
        embedded = embedded[indices]

    # Compute persistent homology
    result = compute_persistent_homology(
        embedded,
        maxdim=params['ripser_maxdim'],
        thresh=params['ripser_thresh']
    )

    # Extract H1 features
    dgm_h1 = result['dgms'][1]  # H1 diagram
    lifetimes, B_ct, n_cycles = extract_betti1_lifetimes(
        dgm_h1,
        min_lifetime=params['min_lifetime']
    )

    # Compute spectral power
    P_ct = np.var(window_data)  # Simple proxy; could use bandpower

    # Compute topological efficiency
    E_B = compute_topological_efficiency(B_ct, P_ct)

    # Verify embedding with FNN
    fnn_pct = false_nearest_neighbors(window_data, m=params['embedding_dim'], tau=tau_opt)

    return {
        'tau_opt': tau_opt,
        'B_ct': B_ct,
        'n_cycles': n_cycles,
        'P_ct': P_ct,
        'E_B': E_B,
        'fnn_pct': fnn_pct,
        'max_lifetime': np.max(lifetimes) if len(lifetimes) > 0 else 0,
        'mean_lifetime': np.mean(lifetimes) if len(lifetimes) > 0 else 0,
    }


def process_channel(raw, channel, params):
    """
    Process all windows for a single channel.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object
    channel : str
        Channel name
    params : dict
        TDA parameters

    Returns
    -------
    channel_features : list of dict
        Features for each window
    """
    # Get channel data
    ch_idx = raw.ch_names.index(channel)
    data = raw.get_data()[ch_idx]
    sfreq = raw.info['sfreq']

    # Window parameters
    window_samples = int(params['window_length'] * sfreq)
    step_samples = int(window_samples * (1 - params['window_overlap']))

    # Process windows
    channel_features = []
    n_windows = (len(data) - window_samples) // step_samples + 1

    for w in range(n_windows):
        start = w * step_samples
        end = start + window_samples
        window_data = data[start:end]

        features = process_single_window(window_data, sfreq, params)
        features['channel'] = channel
        features['window'] = w
        features['time_sec'] = start / sfreq

        channel_features.append(features)

    return channel_features


def process_all_channels(raw, params, channels=None):
    """
    Process all channels in parallel.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object
    params : dict
        TDA parameters
    channels : list, optional
        List of channels to process. If None, process all EEG channels.

    Returns
    -------
    all_features : pd.DataFrame
        Features for all channels and windows
    """
    if channels is None:
        channels = raw.ch_names

    print(f"Processing {len(channels)} channels...")

    # Process channels in parallel
    results = Parallel(n_jobs=params['n_jobs'])(
        delayed(process_channel)(raw, ch, params)
        for ch in tqdm(channels, desc="Channels")
    )

    # Flatten results
    all_features = []
    for channel_features in results:
        all_features.extend(channel_features)

    return pd.DataFrame(all_features)


# =============================================================================
# LOCKING METRICS
# =============================================================================

def compute_locking_per_channel(features_df):
    """
    Compute Lock_E_B for each channel from window-wise features.

    Parameters
    ----------
    features_df : pd.DataFrame
        Features with columns: channel, E_B, etc.

    Returns
    -------
    locking_df : pd.DataFrame
        Lock_E_B and jitter for each channel
    """
    locking_results = []

    for channel in features_df['channel'].unique():
        ch_data = features_df[features_df['channel'] == channel]
        E_B_series = ch_data['E_B'].values

        Lock_E_B, jitter = compute_topological_locking(E_B_series)

        locking_results.append({
            'channel': channel,
            'Lock_E_B': Lock_E_B,
            'jitter': jitter,
            'mean_E_B': np.mean(E_B_series),
            'var_E_B': np.var(E_B_series),
            'mean_B_ct': np.mean(ch_data['B_ct']),
            'n_windows': len(ch_data)
        })

    locking_df = pd.DataFrame(locking_results)
    locking_df = locking_df.sort_values('Lock_E_B', ascending=False)

    return locking_df


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_persistence_diagram(dgm, title="Persistence Diagram H1", ax=None):
    """Plot persistence diagram for H1."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    if len(dgm) > 0:
        ax.scatter(dgm[:, 0], dgm[:, 1], alpha=0.6, s=50)

        # Diagonal line
        max_val = max(np.max(dgm[:, 1]), np.max(dgm[:, 0]))
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)

        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        ax.set_title(title)
        ax.axis('equal')
    else:
        ax.text(0.5, 0.5, 'No H1 cycles detected',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)

    return ax


def plot_channel_features(features_df, channel, figsize=(15, 10)):
    """Plot topological features over time for a single channel."""
    ch_data = features_df[features_df['channel'] == channel]

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # B(c,t)
    axes[0].plot(ch_data['time_sec'], ch_data['B_ct'], 'b-', linewidth=2)
    axes[0].set_ylabel('B(c,t)\n(Betti-1 sum)', fontsize=12)
    axes[0].set_title(f'Channel {channel} - Topological Features', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)

    # P(c,t)
    axes[1].plot(ch_data['time_sec'], ch_data['P_ct'], 'r-', linewidth=2)
    axes[1].set_ylabel('P(c,t)\n(Power)', fontsize=12)
    axes[1].grid(alpha=0.3)

    # E_B(c,t)
    axes[2].plot(ch_data['time_sec'], ch_data['E_B'], 'g-', linewidth=2)
    axes[2].set_ylabel('E_B(c,t)\n(Efficiency)', fontsize=12)
    axes[2].grid(alpha=0.3)

    # Number of cycles
    axes[3].plot(ch_data['time_sec'], ch_data['n_cycles'], 'purple', linewidth=2)
    axes[3].set_ylabel('# H1 cycles', fontsize=12)
    axes[3].set_xlabel('Time (seconds)', fontsize=12)
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_locking_topography(locking_df, top_n=10, figsize=(12, 6)):
    """Plot Lock_E_B across channels."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Bar plot of top channels
    top_channels = locking_df.head(top_n)
    axes[0].barh(range(len(top_channels)), top_channels['Lock_E_B'].values)
    axes[0].set_yticks(range(len(top_channels)))
    axes[0].set_yticklabels(top_channels['channel'].values)
    axes[0].set_xlabel('Lock_E_B', fontsize=12)
    axes[0].set_title(f'Top {top_n} Channels by Topological Locking', fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)

    # Highlight ROI channels
    for i, ch in enumerate(top_channels['channel'].values):
        if ch in ROI_CHANNELS:
            axes[0].get_yticklabels()[i].set_color('red')
            axes[0].get_yticklabels()[i].set_fontweight('bold')

    # Lock_E_B vs jitter scatter
    axes[1].scatter(locking_df['jitter'], locking_df['Lock_E_B'], alpha=0.6, s=80)

    # Highlight ROI channels
    roi_data = locking_df[locking_df['channel'].isin(ROI_CHANNELS)]
    axes[1].scatter(roi_data['jitter'], roi_data['Lock_E_B'],
                    color='red', s=150, alpha=0.8, edgecolors='darkred', linewidths=2,
                    label='ROI channels')

    for _, row in roi_data.iterrows():
        axes[1].annotate(row['channel'], (row['jitter'], row['Lock_E_B']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)

    axes[1].set_xlabel('Jitter (var of dE_B/dt)', fontsize=12)
    axes[1].set_ylabel('Lock_E_B', fontsize=12)
    axes[1].set_title('Locking vs Jitter', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_tda_pipeline(raw_path, output_dir=None, plot_results=True):
    """
    Run complete TDA pipeline on a cleaned EEG file.

    Parameters
    ----------
    raw_path : str or Path
        Path to cleaned MNE Raw file (.fif)
    output_dir : str or Path, optional
        Where to save results. If None, uses OUTPUT_ROOT.
    plot_results : bool
        Whether to generate plots

    Returns
    -------
    results : dict
        All TDA results
    """
    raw_path = Path(raw_path)
    if output_dir is None:
        output_dir = OUTPUT_ROOT
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TDA Pipeline: {raw_path.name}")
    print(f"{'='*60}\n")

    # Load data
    print(f"Loading: {raw_path}")
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose='ERROR')
    print(f"Channels: {len(raw.ch_names)}, Duration: {raw.times[-1]:.1f}s")

    # Process all channels
    print("\nRunning TDA on all channels...")
    features_df = process_all_channels(raw, TDA_PARAMS)

    # Compute locking metrics
    print("\nComputing topological locking...")
    locking_df = compute_locking_per_channel(features_df)

    print(f"\nTop 5 channels by Lock_E_B:")
    print(locking_df[['channel', 'Lock_E_B', 'jitter']].head())

    # Check CP6 rank
    if 'CP6' in locking_df['channel'].values:
        cp6_row = locking_df[locking_df['channel'] == 'CP6']
        cp6_rank = locking_df.index.get_loc(cp6_row.index[0]) + 1
        print(f"\n*** CP6 rank: {cp6_rank}/{len(locking_df)} ***")
    else:
        cp6_rank = None

    # Save results
    subject_id = raw_path.stem.replace('_clean_raw', '').replace('_cleaned-raw', '')
    
    features_file = output_dir / f"{subject_id}_tda_features.csv"
    features_df.to_csv(features_file, index=False)
    print(f"\nSaved features: {features_file}")

    locking_file = output_dir / f"{subject_id}_locking_metrics.csv"
    locking_df.to_csv(locking_file, index=False)
    print(f"Saved locking: {locking_file}")

    # Plotting
    if plot_results:
        fig1 = plot_locking_topography(locking_df, top_n=10)
        plt.savefig(output_dir / f"{subject_id}_locking_topography.png", dpi=150)
        plt.show()

        top_channel = locking_df.iloc[0]['channel']
        fig2 = plot_channel_features(features_df, top_channel)
        plt.savefig(output_dir / f"{subject_id}_{top_channel}_features.png", dpi=150)
        plt.show()

    return {
        'features_df': features_df,
        'locking_df': locking_df,
        'cp6_rank': cp6_rank
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("\nTDA Engine ready.")
    print("\nExample usage:")
    print("  from tda_engine import run_tda_pipeline")
    print("  results = run_tda_pipeline('path/to/cleaned_data.fif')")
    print("\nOr import individual functions:")
    print("  from tda_engine import compute_ami, takens_embedding, compute_persistent_homology")
