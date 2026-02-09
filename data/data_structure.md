## Data directory to use (verified)

- Primary example (per Tom Crawford):
  - `/sptlocal/user/tcrawfor/for_reijo/spt3g_data/ra0hdec-59.75/79891185/`

Last verified: 2026-02-05

## Directory layout (verified)

`/sptlocal/user/tcrawfor/for_reijo/spt3g_data/<field>/<obsid>/`

- Calibrated TOD chunk (used by `extract_binned_tod.py`):
  - `0001_calibrated.g3`

## Verified `.g3` structure (example observation)

Example directory:

- `/sptlocal/user/tcrawfor/for_reijo/spt3g_data/ra0hdec-59.75/79891185/`

### `0001_calibrated.g3` (calibrated)

Frame counts (verified):

- `Observation`: 1
- `PipelineInfo`: 1
- `Wiring`: 1
- `Calibration`: 1
- `Scan`: 8

#### Calibration frame

Key items (verified):

- `BolometerProperties`: `spt3g.calibration.BolometerPropertiesMap` (n_det = 14266)
  - keys are detector ids (`str`, e.g. `"2019.000"`)
  - each value is `spt3g.calibration.BolometerProperties`; verified public fields include:
    - `x_offset`, `y_offset` (angle-like values; focal-plane offsets in SPT3G angle units)
    - `band`, `band_string`, `band_vstring`, `bandwidth`, `center_frequency`, `coupling`
    - `physical_name`, `pixel_id`, `pixel_type`, `pol_angle`, `pol_efficiency`, `wafer_id`
    - `Description`, `Summary`, `hash`
  - `extract_binned_tod.py` uses **only** `x_offset`, `y_offset` and converts to arcmin via
    `offset_arcmin = offset / core.G3Units.arcmin`

Offset validity (verified):

- In this file, not all `x_offset/y_offset` are finite:
  - finite offsets: 11210 / 14266
  - non-finite offsets: 3056 / 14266

#### Scan frames

Key items used by `extract_binned_tod.py` (verified from this file):

- `CalTimestreams`: `spt3g.core.G3TimestreamMap`
  - units: `Tcmb` (per-detector `G3Timestream.units`)
  - in this file (all non-turnaround scans): `n_det = 11018`, `sample_rate = 152.587890625 Hz`
  - `n_time` varies slightly across scans here: `15716` or `15717` samples, and always matches boresight/quaternion length
- `OnlineBoresightRa`, `OnlineBoresightDec`: `spt3g.core.G3Timestream`
  - stored in SPT3G angle units; the extractor converts to degrees via division by `core.G3Units.deg`
- `OnlineRaDecRotation`: `spt3g.core.G3VectorQuat`, shape `(n_time, 4)`
- `Turnaround` (optional): present and `True` on turnaround Scan frames, and absent on non-turnaround Scan frames
  - the extractor drops any Scan with `bool(fr.get("Turnaround", False)) == True`

## Extracted binned `.npz` schema (`extract_binned_tod.py`)

Per non-turnaround Scan frame, `extract_binned_tod.py` writes `outputs/extract_binned_tod_scanXYZ.npz` with:

- **Unit conventions (no SPT3G required downstream)**:
  - angles: **degrees** (`*_deg`) or **arcmin** (`*_arcmin`)
  - TOD: **mK** (`eff_tod_mk`)
- **Raw detector metadata** (SPT3G-independent):
  - `raw_bolos`: detector ids (`str`, e.g. `"2019.000"`), shape `(n_raw,)`
  - `raw_offsets_arcmin`: focal-plane offsets in arcmin, shape `(n_raw, 2)` with columns `(x, y)`
    - definition: `raw_offsets_arcmin[d] = (x_offset[d], y_offset[d]) / core.G3Units.arcmin`
  - `raw_to_eff`: raw→effective mapping, shape `(n_raw,)` (`int64`)
    - `raw_to_eff[d] = e` means raw detector `d` contributes to effective detector `e`
  - `n_raw` meaning (verified for this file’s first non-turnaround scan):
    - start from all detectors in `CalTimestreams` (`11018`)
    - drop detectors with non-finite `(x_offset, y_offset)` (`10940` left)
    - drop detectors with all-NaN TOD in the scan (`10938` left)
- **Effective detector metadata** (focal-plane boxes of size `effective_box_arcmin × effective_box_arcmin`):
  - `eff_counts`: number of raw detectors per effective detector, shape `(n_eff,)` (`int64`)
    - definition: `eff_counts[e] = #{ d : raw_to_eff[d] = e }`
  - `eff_box_index`: effective-detector focal-plane box indices, shape `(n_eff, 2)` (`int64`) with columns `(ix, iy)`
    - let `Δ = effective_box_arcmin`, `x_min = focal_x_min_arcmin`, `y_min = focal_y_min_arcmin`
    - let `nx = ceil((focal_x_max_arcmin - focal_x_min_arcmin) / Δ)`
    - for each raw detector offset `(x_d, y_d)` (arcmin), define
      - `ix_d = floor((x_d - x_min) / Δ)`
      - `iy_d = floor((y_d - y_min) / Δ)`
      - `cell_d = ix_d + nx * iy_d`
    - each effective detector corresponds to one unique `(ix, iy)` pair; `eff_box_index[e] = (ix, iy)`
  - `eff_offsets_arcmin`: effective-detector centroid offsets in arcmin, shape `(n_eff, 2)` with columns `(x, y)`
    - definition (uniform average over detectors in the group):
      - `eff_offsets_arcmin[e] = (1/eff_counts[e]) * Σ_{d: raw_to_eff[d]=e} raw_offsets_arcmin[d]`
- **Binned time axis**:
  - `t_bin_center_s`: seconds from scan start (bin centers), shape `(n_t,)` (`float64`)
    - the code uses `n_samp_per_bin = round(bin_sec * sample_rate_hz)` and then
      `t_bin_center_s[t] = (t + 0.5) * n_samp_per_bin / sample_rate_hz`
- **Binned TOD (mK)**:
  - `eff_tod_mk`: shape `(n_t, n_eff)` (`float32`)
  - definition (uniform averages; NaNs ignored):
    - let `tod_Tcmb[d, s]` be the calibrated TOD from `CalTimestreams` for raw detector `d` and sample `s`
    - convert to mK: `tod_mk[d, s] = tod_Tcmb[d, s] / core.G3Units.mK`
    - subtract per-detector scan mean: `tod0_mk[d, s] = tod_mk[d, s] - mean_s(tod_mk[d, s])` (NaNs ignored)
    - time-bin per raw detector:
      - `raw_bin_mk[t, d] = mean_{s in bin t}(tod0_mk[d, s])` (NaNs ignored)
    - effective-detector bin (for each time bin `t`), averaging only the raw detectors with finite `raw_bin_mk[t, d]`:
      - `eff_tod_mk[t, e] = mean_{d: raw_to_eff[d]=e}(raw_bin_mk[t, d])` (NaNs ignored)
- **Binned pointing (degrees)**:
  - `boresight_pos_deg`: shape `(n_t, 2)` (`float32`) with columns `(RA, Dec)`
    - per time bin, it stores the arithmetic mean of `OnlineBoresightRa/Dec` over all samples in the bin
    - before averaging, boresight RA is mapped to the continuous branch $[180, 540)$
  - `eff_pos_deg`: shape `(n_t, n_eff, 2)` (`float32`) with columns `(RA, Dec)`
    - per time bin, pointing is computed using the **mid-sample** quaternion in the bin (TOD uses all samples)
    - per raw detector, the code calls `maps.get_detector_pointing(x_offset, y_offset, qbin, Equatorial)` and converts to degrees
    - before averaging, detector RA is mapped to the continuous branch $[180, 540)$ (per detector, per time bin)
    - verified: with `(x_offset, y_offset) = (0, 0)`, `maps.get_detector_pointing` matches `OnlineBoresightRa/Dec` at the same sample
      (so detector pointing includes boresight and the full rotation in `OnlineRaDecRotation`)
    - effective-detector centroid is a uniform average of the raw-detector sky positions in the group:
      - `eff_pos_deg[t, e] = mean_{d: raw_to_eff[d]=e}(raw_pos_deg[t, d])` (component-wise; NaNs ignored)
- **Pointing matrix** (global pixelization anchored at (0,0)):
  - `pixel_size_deg`: scalar (`float`)
  - `pix_index`: integer pixel indices, shape `(n_t, n_eff, 2)` (`int64`) with columns `(ix, iy)`
    - definition: `pix_index[...,0] = floor(eff_pos_deg[...,0] / pixel_size_deg)`,
      `pix_index[...,1] = floor(eff_pos_deg[...,1] / pixel_size_deg)`
    - indices may be negative
- **Parameters / provenance**:
  - `bin_sec`, `sample_rate_hz`, `pixel_size_deg`
  - `effective_box_arcmin`, `focal_x_min_arcmin`, `focal_x_max_arcmin`, `focal_y_min_arcmin`, `focal_y_max_arcmin`
  - `tod_key` (always `"CalTimestreams"`), `tod_units` (always `"mK"`)
  - `obs_dir`, `g3_file`

## RA convention (continuous branch)

RA is an angle defined modulo $360^\circ$. For SPT, we want a *continuous* RA coordinate that does not jump at $0/360$.
Since the relevant SPT footprint spans roughly RA $300^\circ \to 105^\circ$ (crossing 0), we adopt a fixed convention:

- **Store all RA in degrees on the branch $[180, 540)$** by mapping:
  - `ra = ra % 360`
  - `ra = ra + 360` if `ra < 180`

This turns values like `359.9, 0.1` into `359.9, 360.1`, preserving continuity for averaging and for pixelization.
