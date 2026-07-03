# R32 Knockout Weight Experiment

## Goal

Use all completed Round of 32 matches as the test set, then search higher 2026 group-stage weights and lower historical weights to improve accuracy and reduce model conflicts.

## Baseline

```json
{
  "alpha_classifier": 1.0,
  "conflict_gate": false,
  "accuracy": 0.5,
  "advancer_accuracy": 0.6,
  "conflict_count": 1,
  "warning_count": 7,
  "mean_wdl_probability_diff": 0.3459091388604525,
  "group_weight": 1.1,
  "historical_scale": 1.0,
  "lambda_decay": 0.002,
  "test_matches": 10
}
```

## Best Strategy

```json
{
  "alpha_classifier": 0.0,
  "conflict_gate": false,
  "accuracy": 0.7,
  "advancer_accuracy": 0.8,
  "conflict_count": 0,
  "warning_count": 0,
  "mean_wdl_probability_diff": 3.3306690738754695e-17,
  "group_weight": 4.0,
  "historical_scale": 0.2,
  "lambda_decay": 0.01,
  "test_matches": 10
}
```

## Best Predictions

| date | home_team | away_team | actual_result | predicted_result | actual_advancer | predicted_advancer | final_home_win | final_draw | final_away_win | expected_home_goals | expected_away_goals | poisson_home_win | poisson_draw | poisson_away_win | model_conflict | model_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-28 | South Africa | Canada | away_win | away_win | Canada | Canada | 0.2437 | 0.2166 | 0.5397 | 1.3119 | 2.0610 | 0.2437 | 0.2166 | 0.5397 | False |  |
| 2026-06-29 | Germany | Paraguay | draw | home_win | Paraguay | Germany | 0.5151 | 0.2066 | 0.2783 | 2.2962 | 1.6408 | 0.5151 | 0.2066 | 0.2783 | False |  |
| 2026-06-29 | Brazil | Japan | home_win | home_win | Brazil | Brazil | 0.5036 | 0.2453 | 0.2511 | 1.6650 | 1.0949 | 0.5036 | 0.2453 | 0.2511 | False |  |
| 2026-06-29 | Netherlands | Morocco | draw | away_win | Morocco | Morocco | 0.1195 | 0.2790 | 0.6014 | 0.4434 | 1.3703 | 0.1195 | 0.2790 | 0.6014 | False |  |
| 2026-06-30 | France | Sweden | home_win | home_win | France | France | 0.8141 | 0.1249 | 0.0609 | 2.8723 | 0.6445 | 0.8141 | 0.1249 | 0.0609 | False |  |
| 2026-06-30 | Ivory Coast | Norway | away_win | away_win | Norway | Norway | 0.2739 | 0.3514 | 0.3748 | 0.7114 | 0.8884 | 0.2739 | 0.3514 | 0.3748 | False |  |
| 2026-06-30 | Mexico | Ecuador | home_win | home_win | Mexico | Mexico | 0.4053 | 0.2978 | 0.2969 | 1.1559 | 0.9431 | 0.4053 | 0.2978 | 0.2969 | False |  |
| 2026-07-01 | England | DR Congo | home_win | home_win | England | England | 0.6708 | 0.1701 | 0.1591 | 2.8100 | 1.2866 | 0.6708 | 0.1701 | 0.1591 | False |  |
| 2026-07-01 | United States | Bosnia and Herzegovina | home_win | home_win | United States | United States | 0.5898 | 0.2016 | 0.2086 | 2.2822 | 1.2739 | 0.5898 | 0.2016 | 0.2086 | False |  |
| 2026-07-01 | Belgium | Senegal | home_win | away_win | Belgium | Senegal | 0.2318 | 0.2144 | 0.5537 | 1.2712 | 2.0865 | 0.2318 | 0.2144 | 0.5537 | False |  |

## Top 10 Strategy Ensemble For Upcoming Matches

| date | home_team | away_team | predicted_advancer | final_home_win | final_draw | final_away_win | expected_home_goals | expected_away_goals | top_scorelines | strategy_conflict_count | strategy_warning_count | prediction_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07-02 | Portugal | Croatia | Portugal | 0.4341 | 0.4223 | 0.1436 | 0.8832 | 0.4270 | [{"score": "0-0", "probability": 0.27}, {"score": "1-0", "probability": 0.238}, {"score": "0-1", "probability": 0.115}, {"score": "2-0", "probability": 0.105}, {"score": "1-1", "probability": 0.102}] | 0 | 0 | 0.0118 |
| 2026-07-02 | Spain | Austria | Spain | 0.7670 | 0.1685 | 0.0645 | 2.2640 | 0.5212 | [{"score": "2-0", "probability": 0.163}, {"score": "1-0", "probability": 0.144}, {"score": "3-0", "probability": 0.123}, {"score": "2-1", "probability": 0.085}, {"score": "1-1", "probability": 0.075}] | 0 | 0 | 0.5985 |
| 2026-07-02 | Switzerland | Algeria | Algeria | 0.2130 | 0.3368 | 0.4502 | 0.6449 | 1.2580 | [{"score": "0-1", "probability": 0.188}, {"score": "0-0", "probability": 0.149}, {"score": "1-1", "probability": 0.121}, {"score": "0-2", "probability": 0.118}, {"score": "1-0", "probability": 0.096}] | 0 | 0 | 0.1134 |
| 2026-07-03 | Argentina | Cape Verde | Argentina | 0.7555 | 0.1557 | 0.0888 | 2.5624 | 0.7971 | [{"score": "2-0", "probability": 0.12}, {"score": "3-0", "probability": 0.102}, {"score": "2-1", "probability": 0.095}, {"score": "1-0", "probability": 0.093}, {"score": "3-1", "probability": 0.081}] | 0 | 0 | 0.5998 |
| 2026-07-03 | Colombia | Ghana | Colombia | 0.3782 | 0.2836 | 0.3382 | 1.2100 | 1.2096 | [{"score": "1-1", "probability": 0.131}, {"score": "1-0", "probability": 0.108}, {"score": "0-1", "probability": 0.108}, {"score": "0-0", "probability": 0.089}, {"score": "2-1", "probability": 0.079}] | 0 | 0 | 0.0401 |
| 2026-07-03 | Australia | Egypt | Egypt | 0.3412 | 0.3147 | 0.3441 | 0.9149 | 1.0151 | [{"score": "0-1", "probability": 0.147}, {"score": "0-0", "probability": 0.145}, {"score": "1-1", "probability": 0.135}, {"score": "1-0", "probability": 0.133}, {"score": "0-2", "probability": 0.075}] | 0 | 0 | 0.0029 |