# Quarter-Final Prediction

## Method

Training uses only 2026 group-stage matches plus completed Round of 32 matches. Completed Round of 16 matches are used as the holdout test set. Quarter-finals are prediction targets only.

Team attacking and defensive rates are adjusted by opponent strength, then shrunk toward the tournament average. Round-of-32 advancement adds a small bonus to team points so penalty or extra-time advancement is visible without letting Round of 16 data leak into training.

## Summary

```json
{
  "model": "group_plus_r32_opponent_adjusted",
  "training_matches": 88,
  "training_scope": "2026 group stage + completed round of 32",
  "test_matches": 8,
  "test_scope": "completed round of 16",
  "prediction_scope": "quarter finals",
  "result_accuracy": 0.375,
  "advancer_accuracy": 0.5,
  "field_goal_rate": 1.4602272727272727,
  "shrinkage_to_field_average": 0.3,
  "knockout_advancer_point_bonus": 0.35
}
```

## Quarter-Final Predictions

| date | match_no | home_team | away_team | predicted_advancer | home_win | draw | away_win | home_advancer_probability | away_advancer_probability | expected_home_goals | expected_away_goals | top_scorelines |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07-09 | 97 | France | Morocco | France | 0.365 | 0.291 | 0.344 | 0.561 | 0.439 | 1.124 | 1.081 | [{"score": "1-1", "probability": 0.134}, {"score": "1-0", "probability": 0.124}, {"score": "0-1", "probability": 0.119}, {"score": "0-0", "probability": 0.11}, {"score": "2-1", "probability": 0.075}] |
| 2026-07-10 | 98 | Spain | Belgium | Spain | 0.503 | 0.303 | 0.194 | 0.725 | 0.275 | 1.221 | 0.637 | [{"score": "1-0", "probability": 0.19}, {"score": "0-0", "probability": 0.156}, {"score": "1-1", "probability": 0.121}, {"score": "2-0", "probability": 0.116}, {"score": "0-1", "probability": 0.099}] |
| 2026-07-11 | 99 | England | Norway | England | 0.407 | 0.218 | 0.375 | 0.547 | 0.453 | 1.855 | 1.774 | [{"score": "1-1", "probability": 0.087}, {"score": "2-1", "probability": 0.081}, {"score": "1-2", "probability": 0.078}, {"score": "2-2", "probability": 0.072}, {"score": "3-1", "probability": 0.05}] |
| 2026-07-11 | 100 | Argentina | Switzerland | Argentina | 0.345 | 0.273 | 0.382 | 0.504 | 0.496 | 1.189 | 1.267 | [{"score": "1-1", "probability": 0.129}, {"score": "0-1", "probability": 0.109}, {"score": "1-0", "probability": 0.102}, {"score": "0-0", "probability": 0.086}, {"score": "1-2", "probability": 0.082}] |

## Round of 16 Holdout Test

| date | match_no | home_team | away_team | actual_result | predicted_result | actual_advancer | predicted_advancer | result_correct | advancer_correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07-05 | 89 | Canada | Morocco | away_win | away_win | Morocco | Morocco | True | True |
| 2026-07-05 | 90 | Paraguay | France | away_win | away_win | France | France | True | True |
| 2026-07-06 | 91 | Brazil | Norway | away_win | home_win | Norway | Brazil | False | False |
| 2026-07-06 | 92 | Mexico | England | away_win | home_win | England | Mexico | False | False |
| 2026-07-06 | 93 | Spain | Portugal | home_win | home_win | Spain | Spain | True | True |
| 2026-07-06 | 94 | United States | Belgium | away_win | home_win | Belgium | United States | False | False |
| 2026-07-07 | 95 | Argentina | Egypt | home_win | away_win | Argentina | Argentina | False | True |
| 2026-07-07 | 96 | Switzerland | Colombia | draw | away_win | Switzerland | Colombia | False | False |