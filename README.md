The SelfPlex Project: Phenomenological Superposition and a Topological Model of Consciousness
This repository contains code and experiments for a working research program on
consciousness as an antientropic "repeller" state and the SelfPlex –
a minimal, topologically stable substrate of conscious experience.
Core idea:

Conscious awareness is not just neural activity; it is the policy that controls
collapse vs. superposition over a core sensory–parietal network.
The same hardware can either:

collapse quickly to a single interpretation (attractor), or
maintain multiple interpretations in parallel (repeller / "phenomenological qubit").


We study this using:

Clinical and psychedelic EEG datasets (propofol, DMT, meditation).
Topological data analysis (TDA) – Betti numbers, persistent homology.
Dynamical metrics – curvature, intrinsic dimension, entropy, PLV.
First-person tasks – Necker cube superposition, subjective ratings.

This repo is intended to be publication-grade:
every preprocessing decision is logged, every artifact removal step is inspectable.

1. Research design
1.1 Phase 1: Triangulation of the observer
Three datasets are used to characterize the topological signature of the conscious "observer" (the SelfPlex):

Propofol anesthesia (OpenNeuro ds005620)

Awake vs. deep sedation vs. recovery.
Characterizes collapse / attractor behavior – what happens when consciousness is suppressed.


DMT inhalation (Pallavicini et al., 2021; Zenodo 3992359)

Baseline vs. DMT.
Characterizes expanded repeller states – consciousness with altered/dissolved body schema.


Meditation (e.g., focused attention vs. mind-wandering)

Characterizes a trained witness state – voluntary maintenance of superposition.



By triangulating across these three conditions, we aim to identify the invariant topological features that define the observer regardless of content.
1.2 Phase 2: Prediction
Once the observer signature is characterized, we test its predictive validity:

Propofol awakenings: Use the triangulated SelfPlex signature to predict subjective experience levels reported upon awakening from sedation.
This provides a prospective test of whether the topological model captures something real about consciousness.

1.3 Separate validation: Necker cube superposition
The Necker cube experiment is a self-contained validation study, not part of the main triangulation:

Single-subject pilot (autistic / high-functioning) with:

LEFT, RIGHT, BOTH (superposition), and control conditions.
Time-locked perturbations (tree images) to test the stability of superposition.


Tests whether the SelfPlex model can detect voluntary bistable perception in real-time.


2. Preprocessing: publication-grade, semi-automatic
All datasets are passed through explicit, logged preprocessing pipelines
before any TDA or metrics.
Example: preprocessing/dmt_preprocessing.py (inhaled DMT study).
Pipeline steps (per file):

File discovery & logging

Auto-scan for *DMT*.bdf in the dataset directory.
Maintain a preprocessing_log.csv with one row per file:
subject ID, channels dropped, ICA components removed, final duration, etc.


Drop non-EEG channels

Automatically remove EXG, Status, GSR, etc.


Raw data inspection (manual)

Interactive multi-channel plot.
User inspects for:

flat channels,
gross artifacts,
obviously broken sensors.


Screenshots can be taken as part of the paper's methods supplement.


Auto-detect + manual confirm bad channels

Heuristics:

low variance over ≥1s windows,
high amplitude spikes,
low correlation with neighbors.


User confirms / edits the suggested bad channel list.


Filtering

Bandpass: 0.5–45 Hz (TDA-friendly; avoids line noise and ultra-slow drift).
Notch: 50 Hz ± 2.5 Hz.


Montage + referencing

Apply a 10-20/10-05 montage appropriate for the system.
Set average reference.


Interpolate bad channels

Using MNE's spherical interpolation.
Bad channels are documented and then repaired.


Filtered data inspection (manual)

Second interactive plot to identify bad time segments:

movement,
electrode pops,
muscle bursts.




Auto-detect + manual confirm bad segments

Mark segments where:

≥ X% of channels exceed amplitude threshold,
duration ≥ Y ms.


User confirms / revises the segment list.
Segments stored as BAD_segment annotations.


Save pre-ICA data

_preica_raw.fif for reproducibility and backup.


ICA (Infomax)

Fit on filtered, annotated data.
Number of components chosen to explain ≥99% variance.


ICA inspection (manual)

Component topomaps.
Component timecourses.
User labels components as:

blink, saccade, muscle, heartbeat, noise, other.




Apply ICA & compare

Remove marked components.
Overlay before vs after traces for frontal channels.
User visually confirms blink/muscle removal.


Crop to segment of interest

User chooses start/end times for the "analysis window."
Ensures we analyze stable, artifact-free stretches.


Save cleaned data

_clean_raw.fif files ready for metrics / TDA analysis.
Update preprocessing_log.csv.



This flow is semi-automatic:

Files are discovered and processed in sequence.
All key judgment calls (bad channels, bad segments, ICA rejections, crop) are explicit and logged.


3. Metrics and topological analysis
After preprocessing, cleaned data are passed through a shared set of metrics:
3.1 Base time-series metrics
Per channel and window (e.g. 5 s windows, sliding):

Band-limited power.
Lempel–Ziv complexity (LZC).
Spectral entropy (H).
Permutation entropy.
Sample entropy.

3.2 Efficiency metrics
Normalize complexity by power:

E_L = LZC / log(1 + P)
E_H = H / log(1 + P)

These capture how much "informational richness" is being maintained per unit energy.
3.3 TDA and curvature
Using time-delay embedding (TDE):

Embed EEG into a high-dimensional trajectory:

dimension d (e.g. 8–10),
delay τ (e.g. 8 samples),
limited to a max number of points for computational stability.



From the embedded point cloud:

Compute persistent homology (Betti numbers):

β₀: connected components.
β₁: loops (core object for "SelfPlex" structure).
β₂: voids (higher-order structure when present).


Compute curvature proxy:

e.g. variance of stepwise distances / mean²,
reflects how sharply the trajectory bends or collapses.


Core consciousness index:

β₁ / R (loops divided by curvature) as an antientropic repeller measure.



3.4 Connectivity and synchronization
Using band-limited filters (e.g. gamma 30–50 Hz, alpha 8–12 Hz):

PLV (phase-locking value) for all channel pairs.
Directed PLV (lead/lag asymmetry).
Cluster structure in connectivity graphs:

which channels form stable loops,
which channels become isolated (e.g., CP6 decoupling in DMT).



3.5 Phenomenological tagging
Each dataset contributes different "labels" for the same metrics:

Propofol:

awake vs deep sedation vs recovery
with/without subjective experiences on awakening.


DMT:

baseline vs peak,
ego dissolution,
OBE-like ratings, etc.


Meditation:

focused attention vs diffuse awareness,
dereification / non-grasping.


Necker cube:

LEFT / RIGHT (collapsed),
BOTH (superposition),
whether superposition survived or broke after perturbation.



These labels let us test:

When does the SelfPlex behave like an attractor (collapse, low β₁, high curvature)?
When does it behave like a repeller / qubit (stable loops, rich superposition, low curvature)?


4. Status

 DMT preprocessing pipeline with publication-grade logging.
 Propofol analysis framework.
 Meditation dataset integration.
 Unified TDA/metrics library shared across datasets.
 Phase 1 triangulation analysis.
 Phase 2 prediction validation on propofol awakenings.
 Necker cube validation study (separate).
 Formal manuscript on "Phenomenological Superposition and the SelfPlex".


5. Installation
bashconda env create -f environment.yml
conda activate selfplex

6. License
TBD

7. Citation
If you use this code or find this research useful, please cite:

[Citation to be added upon publication]
