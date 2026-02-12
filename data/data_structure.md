## Data directory to use (verified)

This document describes the current pipeline under:

- `/home/ajzhou/spt_tod_structured_data/caliberated_data/scripts/`

Primary data roots used by the current scripts:

- Raw downsampled TOD root:
  - `/sptgrid/data/bolodata/downsampled/`
- Per-observation calibration frame root:
  - `/sptgrid/analysis/calarchive/v5/calframe/`
- Structured output root:
  - `/scratch/ajzhou/caliberated_data/data/`

Last verified: 2026-02-11 (against code in this repository)

## Directory layout (verified)

Input trees:

- Raw downsampled observation:
  - `/sptgrid/data/bolodata/downsampled/<field>/<obsid>/`
  - contains:
    - `nominal_online_cal.g3`
    - `0000.g3`, `0001.g3`, ...
- Calibration archive frame for the same observation:
  - `/sptgrid/analysis/calarchive/v5/calframe/<field>/<obsid>.g3`

Structured output tree:

- `/scratch/ajzhou/caliberated_data/data/<field>/<obsid>/`
  - intermediate calibrated chunk(s):
    - `0000_calibrated.g3`, `0001_calibrated.g3`, ...
  - extracted products:
    - `binned_tod_10arcmin/<chunk>_calibrated_scanXYZ.npz`

Field-level run log from compact pipeline:

- `/scratch/ajzhou/caliberated_data/data/<field>/_compact_progress.csv`

## Calibration pipeline (verified)

Calibration is implemented by `calibrate_downsampled_obs.py` and orchestrated by
`calibrate_then_extract_compact.py`.

Per chunk, the calibrated pipeline order is:

1. `core.G3Reader` over:
   - `calarchive/<field>/<obsid>.g3`
   - `<obs_dir>/nominal_online_cal.g3`
   - `<obs_dir>/<chunk>.g3`
2. `ensure_bolometer_properties`:
   - if a Calibration frame has `NominalBolometerProperties` but not
     `BolometerProperties`, copy it to `BolometerProperties`
3. Optional non-turn scan preselection:
   - only used when `--nonturn-scan-indices` is provided
4. `std_processing.DropWasteFrames` (includes turnaround removal behavior)
5. `transients.balloons.BalloonAvoider`
6. `timestreamflagging.PruneRawTimestreams`:
   - input key: `RawTimestreams_I`
   - output key: `DeflaggedRawTimestreams_I`
7. `calibration.CalibrateRawTimestreams`:
   - input key: `DeflaggedRawTimestreams_I`
   - output key: `CalTimestreams`
   - `opacity=True`, `kcmb=True`
8. `pointing.UpdateBoresightPointing`:
   - `pointing_key="OnlineRaDecRotation"`
9. Optional `todfilter.TodFiltering` to `CalTimestreamsFiltered`:
   - default is disabled (`WRITE_MINIMAL_FILTERED_TOD=False`)
   - current filter defaults if enabled:
     - `poly_order=1`
     - `hpf_filter_frequency=-1.0` (disabled)
     - `lpf_filter_frequency=-1.0` (disabled)
10. `core.G3Writer` writes frame streams:
    - `Observation`, `PipelineInfo`, `Wiring`, `Calibration`, `Scan`

Compact-run defaults currently set in `calibrate_then_extract_compact.py`:

- calibrate all non-turn scans: `CALIBRATE_NONTURN_SCAN_INDICES=None`
- extract all non-turn scans: `EXTRACT_SCAN_INDICES=None`
- delete calibrated intermediate `.g3`: `DELETE_CALIBRATED_G3=True`
- keep skip flag enabled: `SKIP_IF_EXPECTED_OUTPUT_EXISTS=True`
  - note: skip check is strict only when `EXTRACT_SCAN_INDICES` is an explicit list

## Verified `.g3` structure (example observation)

Example calibrated file pattern:

- `/scratch/ajzhou/caliberated_data/data/<field>/<obsid>/<chunk>_calibrated.g3`

### `<chunk>_calibrated.g3` (calibrated)

Frame streams written by the calibrator are fixed:

- `Observation`
- `PipelineInfo`
- `Wiring`
- `Calibration`
- `Scan`

Actual frame counts vary by observation/chunk and are not hardcoded.

#### Calibration frame

Key items required by extractor:

- `BolometerProperties`: `spt3g.calibration.BolometerPropertiesMap`
  - keys are detector ids (`str`, e.g. `"2019.000"`)
  - extractor uses only:
    - `x_offset`, `y_offset`
  - conversion used by extractor:
    - `offset_arcmin = offset / core.G3Units.arcmin`

#### Scan frames

Required keys for extraction:

- `CalTimestreams` (`TOD_KEY` default):
  - type: `spt3g.core.G3TimestreamMap`
  - extractor converts TOD to mK via division by `core.G3Units.mK`
- `OnlineBoresightRa`, `OnlineBoresightDec`:
  - boresight angle timestreams
- `OnlineRaDecRotation`:
  - quaternion timestream (shape `(n_time, 4)` when converted to ndarray)
- `Turnaround` (optional flag):
  - extractor drops any frame where `bool(fr.get("Turnaround", False))` is `True`

## Extracted binned `.npz` schema (`extract_binned_tod.py`)

Per non-turnaround Scan frame, the extractor writes:

- `/scratch/ajzhou/caliberated_data/data/<field>/<obsid>/binned_tod_10arcmin/<chunk>_calibrated_scanXYZ.npz`

with:

- **Unit conventions (SPT3G-independent downstream)**
  - angles: degrees (`*_deg`) or arcmin (`*_arcmin`)
  - TOD: mK (`eff_tod_mk`)

- **Raw detector metadata**
  - `raw_bolos`: detector ids, shape `(n_raw,)`
  - `raw_offsets_arcmin`: shape `(n_raw, 2)` with columns `(x, y)`
  - `raw_to_eff`: raw-to-effective detector map, shape `(n_raw,)`

- **Effective detector metadata**
  - `eff_offsets_arcmin`: effective detector centroid offsets, shape `(n_eff, 2)`
  - `eff_counts`: raw detector counts per effective detector, shape `(n_eff,)`
  - `eff_box_index`: integer focal-plane box index `(ix, iy)`, shape `(n_eff, 2)`
  - grouping hyperparameters:
    - `EFFECTIVE_BOX_ARCMIN = 5.0`
    - `FOCAL_X_MIN_ARCMIN = -80.0`, `FOCAL_X_MAX_ARCMIN = 80.0`
    - `FOCAL_Y_MIN_ARCMIN = -60.0`, `FOCAL_Y_MAX_ARCMIN = 60.0`

- **Time axis**
  - `t_bin_center_s`: shape `(n_t,)`
  - `BIN_SEC = 0.1`
  - `n_samp_per_bin = round(BIN_SEC * sample_rate_hz)`, with minimum 1

- **Binned TOD**
  - `eff_tod_mk`: shape `(n_t, n_eff)`
  - preprocessing before binning:
    - detector-wise polynomial subtraction with
      `REMOVE_TIME_POLYNOMIAL_ORDER = 1`
    - NaNs are ignored in averaging operations

- **Binned pointing**
  - `boresight_pos_deg`: shape `(n_t, 2)`
  - `eff_pos_deg`: shape `(n_t, n_eff, 2)`
  - `USE_RAW_DETECTOR_POINTING = False`:
    - effective pointing is computed from `eff_offsets_arcmin` and mid-bin quaternion
    - call path uses `maps.get_detector_pointing(...)` per effective detector, per time bin

- **Pixel indices**
  - `pix_index`: shape `(n_t, n_eff, 2)`
  - `pixel_size_deg = PIXEL_SIZE_ARCMIN / 60.0`
  - `pix_index[...,0] = floor(eff_pos_deg[...,0] / pixel_size_deg)`
  - `pix_index[...,1] = floor(eff_pos_deg[...,1] / pixel_size_deg)`

- **Provenance fields written in each npz**
  - `bin_sec`, `sample_rate_hz`, `pixel_size_deg`
  - `effective_box_arcmin`
  - `focal_x_min_arcmin`, `focal_x_max_arcmin`
  - `focal_y_min_arcmin`, `focal_y_max_arcmin`
  - `tod_units` (string `"mK"`)
  - `tod_key` (string, default `"CalTimestreams"`)
  - `obs_dir`, `g3_file`

## RA convention (continuous branch)

RA is periodic modulo $360^\circ$. The extractor stores RA on a continuous branch
$[180, 540)$:

- `ra = ra % 360`
- if `ra < 180`, set `ra = ra + 360`

This mapping is applied before RA averaging and before pixel-index construction.
