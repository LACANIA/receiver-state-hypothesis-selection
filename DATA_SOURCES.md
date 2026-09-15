# Original data sources

Official repository/documentation links below were checked on 2026-09-15. No dataset is redistributed. Public access does not automatically grant redistribution rights; use each owner's current terms.

| Original source | Link | Files and intended use |
|---|---|---|
| Measured Iridium Doppler observations | [Author repository](https://github.com/Baoshan-Song/Certifiable-Doppler-positioning), [registered CSV path](https://github.com/Baoshan-Song/Certifiable-Doppler-positioning/blob/main/matlab/data/iridium/Iridium_Doppler_measurements.csv) | `Iridium_Doppler_measurements.csv`; measured-static case and source geometry for controlled observations. The downloaded revision must match the study's frozen input identity. |
| UrbanNav Tokyo | [Official UrbanNav repository](https://github.com/IPNL-POLYU/UrbanNavDataset#dataset-1-urbannav-tk-20181219) | Select `UrbanNav-TK-20181219`, Odaiba and Shinjuku. The preparation code reads `reference.csv`, `imu.csv` and GNSS timing files such as `rover_ublox.obs`. LiDAR is outside this workflow. |
| GVINS sports_field | [Official dataset repository](https://github.com/HKUST-Aerial-Robotics/GVINS-Dataset) | Follow the current `sports_field` download link in its dataset table; ROS1 bag with raw GNSS, PVT and broadcast ephemerides. This is an initialization-assisted GPS-L1 mechanism test. |
| CelesTrak orbital elements | [Official GP/OMM documentation](https://celestrak.org/NORAD/documentation/gp-data-formats.php) | OMM XML and NORAD identifiers. The current feed is mutable and cannot substitute for the exact historical D2 input. |
| Additional physical-layer error-source record | [Registered Mendeley record, version 2](https://doi.org/10.17632/xcxspv8c2r.2) | The original project associates an Iridium corpus with error-model work. Exact membership of `1208-1009_20_subset.txt` requires author confirmation; do not treat this link as a verified substitute for that exact subset. Existing frozen noise-model parameters are retained in the configurations. |

GVINS software and data are distinct resources: the software is [HKUST GVINS](https://github.com/HKUST-Aerial-Robotics/GVINS); message definitions come from [gnss_comm](https://github.com/HKUST-Aerial-Robotics/gnss_comm). Orbit dependencies include [SGP4](https://pypi.org/project/sgp4/) and [Skyfield](https://rhodesmill.org/skyfield/). These are dependencies, not additional observation datasets.

## Where to place downloads

For the preserved manual source tree, use the following locations relative to `manual_workflows/tree/`, or set the corresponding script's input variable to your own data directory before running:

- Iridium: `leo-c/_new_solver_lab/final_release/MA_BGTR_v7_2_freeze_r2/data/real_iridium/Iridium_Doppler_measurements.csv`.
- UrbanNav: `datasets/UrbanNav/UrbanNav-TK-20181219/raw/`, retaining the two route directories and original filenames. Check `DATA_ROOT` in the preparation script.
- GVINS: keep `sports_field.bag` in your own raw-data directory and pass its path to the extractor. Subsequent stages use the generated pilot/cache and alignment directories beneath `leo-c_new_solver_lab/GVINS_adapter/`.
- D2: use the original and normalized OMM paths defined by `d2_common.py` under `flagship/34_d2_snapshot_capture/`. Obtain the frozen historical input from the authors if the original host no longer provides it.

## Exact D2 identity

Capture: **2026-08-27 10:14:35 UTC**. Original snapshot SHA256: `d1b58796a12dff8d3a3020d58e93516d529850508fbb910e529f0af109059827`.

The study uses 80/80 target NORAD objects, 92 eligible evaluation blocks and 552 outputs. Observations cover 0–120 s; the reporting epoch is 110 s. Dynamic candidates use `p_hat(110) = p_hat(120) - 10*v_hat`. This is retrospective full-window estimation, not real-time navigation.

Development and controlled motion-clock observations are generated using the retained code and frozen configurations. Intermediate observations, fitted records, Jacobian arrays, input inventories and result tables are intentionally absent. Exact historical snapshots, initialization records and provenance inventories may need to be obtained from the corresponding author; none is represented here as a newly invented public download.
