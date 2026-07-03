# R32 Knockout Weight Experiment

## Goal

Use all completed Round of 32 matches as the test set, then search higher 2026 group-stage weights and lower historical weights to improve accuracy and reduce model conflicts.

## Baseline

```json
{
  "alpha_classifier": 1.0,
  "conflict_gate": false,
  "accuracy": 0.46153846153846156,
  "advancer_accuracy": 0.6923076923076923,
  "conflict_count": 1,
  "warning_count": 9,
  "mean_wdl_probability_diff": 0.33820862397845936,
  "group_weight": 1.1,
  "historical_scale": 1.0,
  "lambda_decay": 0.002,
  "test_matches": 13
}
```

## Best Strategy

```json
{
  "alpha_classifier": 0.4,
  "conflict_gate": false,
  "accuracy": 0.7692307692307693,
  "advancer_accuracy": 0.7692307692307693,
  "conflict_count": 0,
  "warning_count": 2,
  "mean_wdl_probability_diff": 0.10389887558479509,
  "group_weight": 4.0,
  "historical_scale": 0.2,
  "lambda_decay": 0.01,
  "test_matches": 13
}
```

## Best Predictions

| date | home_team | away_team | actual_result | predicted_result | actual_advancer | predicted_advancer | final_home_win | final_draw | final_away_win | expected_home_goals | expected_away_goals | poisson_home_win | poisson_draw | poisson_away_win | model_conflict | model_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-28 | South Africa | Canada | away_win | away_win | Canada | Canada | 0.2904 | 0.2558 | 0.4538 | 1.3119 | 2.0610 | 0.2437 | 0.2166 | 0.5397 | False |  |
| 2026-06-29 | Germany | Paraguay | draw | home_win | Paraguay | Germany | 0.5946 | 0.2375 | 0.1679 | 2.2962 | 1.6408 | 0.5151 | 0.2066 | 0.2783 | False |  |
| 2026-06-29 | Brazil | Japan | home_win | home_win | Brazil | Brazil | 0.5074 | 0.3405 | 0.1521 | 1.6650 | 1.0949 | 0.5036 | 0.2453 | 0.2511 | False |  |
| 2026-06-29 | Netherlands | Morocco | draw | draw | Morocco | Morocco | 0.1766 | 0.4514 | 0.3720 | 0.4434 | 1.3703 | 0.1195 | 0.2790 | 0.6014 | False | classification model and goal model disagree. |
| 2026-06-30 | France | Sweden | home_win | home_win | France | France | 0.8139 | 0.1496 | 0.0366 | 2.8723 | 0.6445 | 0.8141 | 0.1249 | 0.0609 | False |  |
| 2026-06-30 | Ivory Coast | Norway | away_win | draw | Norway | Ivory Coast | 0.3119 | 0.4603 | 0.2279 | 0.7114 | 0.8884 | 0.2739 | 0.3514 | 0.3748 | False |  |
| 2026-06-30 | Mexico | Ecuador | home_win | home_win | Mexico | Mexico | 0.4896 | 0.3322 | 0.1782 | 1.1559 | 0.9431 | 0.4053 | 0.2978 | 0.2969 | False |  |
| 2026-07-01 | England | DR Congo | home_win | home_win | England | England | 0.7237 | 0.1808 | 0.0954 | 2.8100 | 1.2866 | 0.6708 | 0.1701 | 0.1591 | False |  |
| 2026-07-01 | United States | Bosnia and Herzegovina | home_win | home_win | United States | United States | 0.5790 | 0.2954 | 0.1257 | 2.2822 | 1.2739 | 0.5898 | 0.2016 | 0.2086 | False |  |
| 2026-07-01 | Belgium | Senegal | home_win | away_win | Belgium | Senegal | 0.2243 | 0.2278 | 0.5479 | 1.2712 | 2.0865 | 0.2318 | 0.2144 | 0.5537 | False |  |
| 2026-07-02 | Portugal | Croatia | home_win | home_win | Portugal | Portugal | 0.5552 | 0.3058 | 0.1390 | 1.0788 | 0.6860 | 0.4452 | 0.3237 | 0.2311 | False |  |
| 2026-07-02 | Spain | Austria | home_win | home_win | Spain | Spain | 0.7839 | 0.1573 | 0.0588 | 2.4579 | 0.7523 | 0.7382 | 0.1639 | 0.0979 | False |  |
| 2026-07-02 | Switzerland | Algeria | home_win | home_win | Switzerland | Switzerland | 0.3517 | 0.3330 | 0.3154 | 0.8580 | 1.5130 | 0.2141 | 0.2614 | 0.5245 | False | classification model and goal model disagree. |

## Top 10 Strategy Ensemble For Upcoming Matches

| date | home_team | away_team | predicted_advancer | final_home_win | final_draw | final_away_win | expected_home_goals | expected_away_goals | top_scorelines | strategy_conflict_count | strategy_warning_count | prediction_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07-03 | Argentina | Cape Verde | Argentina | 0.8345 | 0.1162 | 0.0493 | 2.7980 | 1.0562 | [{"score": "2-1", "probability": 0.094}, {"score": "2-0", "probability": 0.089}, {"score": "3-1", "probability": 0.087}, {"score": "3-0", "probability": 0.083}, {"score": "1-1", "probability": 0.067}] | 0 | 0 | 0.7183 |
| 2026-07-03 | Colombia | Ghana | Colombia | 0.4716 | 0.2589 | 0.2696 | 1.4004 | 1.4491 | [{"score": "1-1", "probability": 0.118}, {"score": "1-2", "probability": 0.086}, {"score": "0-1", "probability": 0.084}, {"score": "2-1", "probability": 0.083}, {"score": "1-0", "probability": 0.082}] | 2 | 3 | 0.2020 |
| 2026-07-03 | Australia | Egypt | Australia | 0.4925 | 0.2143 | 0.2932 | 1.0708 | 1.2872 | [{"score": "1-1", "probability": 0.131}, {"score": "0-1", "probability": 0.122}, {"score": "1-0", "probability": 0.102}, {"score": "0-0", "probability": 0.095}, {"score": "1-2", "probability": 0.084}] | 4 | 5 | 0.1992 |