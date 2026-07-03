# World Cup Prediction Model

This is an MVP football match prediction project focused on parameter experiments. It trains on historical international tournament samples, uses 2026 FIFA World Cup group-stage matches as the validation set, and searches feature weights for the highest validation accuracy.

The output is probabilistic, not a deterministic score forecast. Football scores are noisy, so this project is a modelling/resume project, not a betting system.

## Project Overview

The pipeline combines:

- time-aware Elo ratings
- recent form features that only use matches before the target match
- coach proxy features
- injury/squad-availability proxy features
- tournament context features
- automated weight search over Elo, form, coach, injury, decay, and Elo K-factor parameters

## Data Sources

- `data/sample/historical_matches.csv`: compact historical tournament training sample.
- `data/raw/worldcup_2026_group_stage.csv`: fetched from 2026 FIFA World Cup group pages on Wikipedia by `scripts/fetch_2026_group_stage.py`.
- `data/sample/coach_data.csv` and `data/sample/injury_data.csv`: proxy data structures for incomplete coach and injury data.

## Run

```powershell
cd D:\Lottery\worldcup-prediction-model
python scripts/fetch_2026_group_stage.py
python experiments/run_weight_search.py --max-combos 300
python experiments/train_final_model.py
python experiments/predict_round_of_32.py
python experiments/optimize_r32_knockout_weights.py
```

Outputs are written to:

- `outputs/weight_search_results.csv`
- `outputs/best_params.json`
- `outputs/model_report.md`
- `outputs/round_of_32_validation.csv`
- `outputs/round_of_32_predictions.csv`
- `outputs/round_of_32_report.md`
- `outputs/round_of_32_threshold_selection.csv`
- `outputs/淘汰赛权重实验结果.csv`
- `outputs/淘汰赛单一最佳预测.csv`
- `outputs/淘汰赛前10策略平均预测.csv`
- `outputs/淘汰赛前10策略明细.csv`
- `outputs/r32_knockout_weight_experiment_report.md`
- `models/final_model.joblib`

## Example Prediction

```powershell
python -m src.predict_demo Spain Austria
```

The demo returns win/draw/loss probabilities, expected goals, and top Poisson scorelines.

## Round of 32 Prediction

To validate on completed Round of 32 matches and predict the six upcoming fixtures:

```powershell
python experiments/predict_round_of_32.py
```

This writes validation metrics and test predictions under `outputs/round_of_32_*`. The prediction output includes Poisson-derived WDL probabilities plus model consistency warnings when the classifier and goal model disagree.

## Knockout Weight Experiment

To use all completed Round of 32 matches as a test set and search whether higher 2026 group-stage weight plus lower historical weight improves accuracy/conflict:

```powershell
python experiments/optimize_r32_knockout_weights.py
```

This writes the searched configurations, the single best prediction table, and a top-10 strategy ensemble average under `outputs/r32_knockout_weight_*`.

## Evaluation

The current experiment sorts by validation `accuracy`, matching the requested objective. It also records `log_loss`, `brier_score`, `macro_f1`, and goal MAE/RMSE so probability quality remains visible.

## Limitations

- The included historical training data is deliberately small for an MVP.
- Coach and injury data are proxy/sample features where real structured feeds are unavailable.
- Validation results can shift as live 2026 group-stage pages are updated.
- No future match information is used when building Elo or form features.

## Future Improvements

- Replace sample historical data with a complete international-match dataset.
- Add real coach tenure and injury availability feeds.
- Expand model comparison across XGBoost, LightGBM, and calibrated sklearn models.
- Add FastAPI and a lightweight UI after the modelling pipeline is stable.
