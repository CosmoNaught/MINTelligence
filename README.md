# MINTelligence (MINTe) - MINTverse RNN (Core Baseline)

This repo contains the legacy GRU/LSTM baseline I use for forecasting malaria dynamics from MINTverse simulations.

It’s deliberately self-contained in a single script (`main.py`) so that you can:
- point it at a DuckDB file of simulation outputs,
- pick a target (prevalence or clinical cases),
- and train + evaluate GRU/LSTM models end-to-end with one command.

Newer work (Mamba/SSM models, JAX stacks, etc.) lives elsewhere. This script exists as a **strong, reproducible baseline** that I can always compare against.

---

## What this actually does

High-level pipeline:

1. **Connect to DuckDB**  
   - Reads a table of agent-based simulation outputs (one row per day per simulation).  
   - Can subsample by:
     - `parameter_index` (`--param-limit`)
     - number of simulations per parameter (`--sim-limit`).

2. **Aggregate and define the prediction target**  
   Two modes:

   - `--predictor prevalence`  
     - Uses raw counts `n_detect_lm_0_1825` / `n_age_0_1825` to compute prevalence.  
     - Aggregates into windows of `--window-size` days over the last 6 years of the simulation.  
     - Stores ratio-of-sums (not mean of ratios) to get a proper prevalence over the window.

   - `--predictor cases`  
     - Uses `n_inc_clinical_0_36500` and `n_age_0_36500`.  
     - Computes *cases per 1000 person-days*, aggregated over the window.  
     - Carries exposure along (total person-days) so we can weight the loss appropriately later.

3. **Filter out uninteresting simulations**  
   - Drop parameter–simulation combinations with negligible signal:
     - `--min-prevalence` (e.g. prevalence < 1%)
     - or `--min-cases` (e.g. very low incidence)
   - This prevents the model from spending all its capacity learning "zero everywhere" dynamics.

4. **Create leakage-free train/val/test splits**  
   - Splitting happens at the **parameter level**, not at the time-step level:
     - 70% of parameters → train
     - 15% → validation
     - 15% → test
   - All simulations under the same parameter go into the same split.
   - A CSV copy of the split (parameter, simulation, global IDs, split label) is saved so that runs are reproducible (`--use-existing-split` to re-use it).
   - There is an explicit integrity check to ensure there is no overlap between splits.

5. **Feature construction & scaling**

   For each parameter–simulation:

   - **Static covariates** (per parameter):
     - e.g. `eir`, `dn0_use`, `dn0_future`, `Q0`, `phi_bednets`, `seasonal`, `routine`, `itn_use`, `irs_use`, `itn_future`, `irs_future`, `lsm`.
   - Some covariates are only allowed to be “visible” after the intervention time (`9 * 365` days).  
     Before that, the corresponding columns are zeroed out:
     - `dn0_future`, `itn_future`, `irs_future`, `lsm`, `routine`.
   - Static covariates are scaled using a `StandardScaler` **fit only on the training parameters** (one row per param–sim to avoid overweighting long series).

   Time features:

   - Optionally encode seasonal structure as:
     - `sin(2π * day_of_year / 365)`, `cos(2π * day_of_year / 365)` (`--use-cyclical-time`).
   - Otherwise, use a normalised time index in `[0, 1]`.
   - Add event-related dynamics:
     - binary flag: `post9` (has intervention happened yet)
     - `time_since_post9` (in years)

   The resulting feature vector per time step is:

   - With cyclical time:  
     `sin, cos, scaled_static..., post9, time_since_post9`
   - Without cyclical time:  
     `t_norm, scaled_static..., post9, time_since_post9`

6. **Target transforms**

   Targets are transformed at train time and inverted for metrics/plots:

   - Prevalence (bounded in [0, 1]):
     - Clip to `[eps, 1-eps]`, then apply a logit.
   - Counts/rates (cases per 1000 person-days):
     - Apply `log1p` after clamping to ≥ 0.

   This makes the regression problem better-behaved and gives the model a nicer loss landscape.

7. **Windowed time-series dataset**

   - Sequences are built as rolling windows of length `--lookback` over each time series.
   - The `TimeSeriesDataset` stores the full time series per param–sim and only creates windows lazily to avoid blowing up RAM.
   - For training:
     - Windows can stride by `--train-stride` (to subsample when there’s a lot of data).
   - For validation/test:
     - Non-overlapping windows (stride = lookback).

8. **Models: GRU and LSTM**

   Both models are very standard, on purpose:

   - `GRUModel` / `LSTMModel`:
     - Backbone: `nn.GRU` or `nn.LSTM`
     - Optional dropout between recurrent layers
     - LayerNorm on the hidden state
     - Final linear layer to a scalar output
     - Identity activation (because we train in the transformed target space)

   There’s a `ModelFactory` so the main training loop can switch between GRU and LSTM cleanly.

9. **Loss: value + shape**

   The `Trainer` optimises a weighted objective:

   - Base term:
     - Weighted MSE between predictions and targets.
     - For prevalence, weights are 1.
     - For cases, weights are person-days exposure in the window (to reflect confidence).
   - Optional **first derivative loss** (`--diff-loss-alpha`):
     - Penalises mismatch in the first temporal derivative of the signal.
     - Intuition: match not just the level, but also the *shape* / slope.
   - Optional **second derivative loss** (`--diff2-loss-beta`):
     - Penalises mismatch in curvature.
     - Encourages smoother and more realistic trajectories.

   Loss can be computed either in:
   - transformed space (`--loss-space transformed`), or
   - natural space (`--loss-space natural`, via inverse transforms).

10. **Training loop**

    - Mixed-precision training using `torch.amp.autocast` and `GradScaler`.
    - Early stopping with patience (`--patience`).
    - LR scheduling via `ReduceLROnPlateau` on validation loss.
    - Best model (by validation loss) is saved, plus a final checkpoint.
    - Training history (train/val loss per epoch) is written to JSON.

11. **Evaluation & metrics**

    On the test set, the trainer computes:

    - MSE, RMSE, MAE, R²
    - SMAPE
    - Bias (mean prediction error)
    - For prevalence, an approximate log-likelihood under a Bernoulli model.

    All metrics are computed **in natural space** after inverse transformation.

    Predictions and targets are flattened across windows/time for the metrics; raw arrays are also returned for downstream analysis.

12. **Hyperparameter optimisation (Optuna)**

    If `--run-tuning` is enabled:

    - Runs Optuna for both GRU and LSTM with:
      - TPE sampler
      - median pruner
    - Searches over:
      - learning rate
      - weight decay
      - hidden size
      - number of layers
      - dropout
      - lookback window length
    - Each trial:
      - builds datasets & loaders,
      - trains a model with early stopping,
      - returns the best validation loss.
    - Best configs for each model type are written out as JSON and can be reused via `--use-tuned-parameters`.

    There is some extra cleanup logic around DataLoader workers and GPU memory to keep long Optuna runs sane.

13. **Visualisation**

    The `Visualizer` can:

    - Plot training curves for GRU and LSTM (train vs val loss, best epoch).
    - Plot a bar chart comparing test metrics across GRU and LSTM.
    - Plot actual vs predicted trajectories for a random subset of parameter sets.

---

## One File To Rule Them All

This script is deliberately “one-file modular”:

- All the logical pieces are there:
  - `Config`, `DataModule`, `TimeSeriesDataset`, `ModelFactory`, `Trainer`, `HyperparameterOptimizer`, `Visualizer`, `main`.
- But they live in a single `main.py` so that:
  - a collaborator can reproduce the baseline by running one script against one database,
  - there’s no ambiguity about which version of which module was used,
  - it’s easy to archive the whole baseline as a single artefact.

In newer work I split things into proper Python packages and use more sophisticated config management (Hydra etc.). This script is intentionally left as a **self-contained baseline**.

---

## Running it

Typical usage:

```bash
python main.py \
  --db-path /path/to/mintverse.duckdb \
  --table-name simulation_results \
  --predictor prevalence \
  --window-size 7 \
  --min-prevalence 0.01 \
  --use-cyclical-time \
  --hidden-size 128 \
  --num-layers 2 \
  --dropout 0.1 \
  --lookback 52 \
  --batch-size 1024 \
  --epochs 64 \
  --device cuda \
  --output-dir results_prevalence \
  --run-tuning \
  --tuning-output-dir results_tuned_prevalence
````

For cases instead of prevalence:

```bash
python main.py \
  --db-path /path/to/mintverse.duckdb \
  --table-name simulation_results \
  --predictor cases \
  --window-size 30 \
  --min-cases 0.1 \
  --use-cyclical-time \
  ... (same idea)
```

---

## Dependencies

Core stack:

* Python ≥ 3.10
* PyTorch
* DuckDB
* NumPy, Pandas, scikit-learn
* Optuna
* Matplotlib
* tqdm

Install via your favourite environment manager; this script doesn’t enforce a specific tooling stack. Tried and tested with `uv` but that's my flavour of choice. Pretty sure `hatch` work ok with the script layout too.

---

## What this is for (and what it isn’t)

This code is **not** trying to be a generic time-series library or a polished production system.

It’s specifically:

* a **baseline GRU/LSTM forecaster** for ABM-generated malaria time series,
* with:

  * careful handling of data leakage,
  * proper weighting for heterogeneous exposure,
  * shape-aware losses,
  * and a simple Optuna wrapper.
 
# Coming Soon!

- PyPI implementation for training and inference with pre-trained weights for back-end API support
- ONNX compatibility layer
```
