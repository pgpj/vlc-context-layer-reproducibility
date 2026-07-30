# Visible-Light Localization as a Context Layer for Hybrid RF/VLC Indoor IoT

## Reproducibility package for COMMAG-26-00412.R1

This archive supports the controlled experiments introduced in the final major revision of **Visible-Light Localization as a Context Layer for Hybrid RF/VLC Indoor IoT**.

## Scope and provenance

The previously submitted R1 archive did not contain the original simulation code, datasets, seeds, weights, or logs. Therefore, the material in this archive implements the **new, fixed R2 controlled protocol** used for the revised ablation, repeated-run SNR study, missingness analysis, overhead calculation, and aggregated CDF.

## Contents

- `config/fixed_protocol.json`: fixed geometry, dataset, missingness, training, and federated settings.
- `src/`: optical simulator, training workers, aggregation, SNR/CDF evaluation, and report-asset generation.
- `checkpoints/`: trained weights for the six compared methods and five seeds.
- `results/`: per-seed and aggregate metrics, CDF error samples, SNR sweep, overhead, and environment information.
- `figures/`: vector figures generated from the archived results.
- `requirements.txt`: Python dependencies.

## Main protocol

- Seeds: 11, 23, 37, 53, and 71.
- Trajectories per seed: 160 training, 40 validation, and 40 test.
- Samples per trajectory: 80 at 10 Hz.
- Window length: 12 samples.
- Missingness: 7% independent per-anchor loss plus 3% complete-frame losses in bursts of 2-4 samples.
- Central training: 30 epochs.
- Federated training: eight clients, 40 rounds, one local epoch, 75% participation, sample-count-weighted FedAvg.

## Reproduction

From the repository root, install the dependencies and run the worker and aggregation scripts. The execution environment should be recorded because exact wall-clock latency depends on the CPU and software stack.

```bash
python -m pip install -r requirements.txt
python src/run_workers.py
python src/aggregate_results.py
python src/evaluate_snr_and_cdf.py
python src/make_report_assets.py
```

## Availability record

Persistent repository URL/DOI: https://github.com/pgpj/vlc-context-layer-reproducibility



## Citation and license

Use `CITATION.cff` for citation metadata. A software/data license has not yet been selected; see `LICENSE_PENDING.md` before making the repository public.
