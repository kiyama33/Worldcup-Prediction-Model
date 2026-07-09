# World Cup Knockout Prediction

当前项目只保留一个主流程：用已经完成的 2026 世界杯比赛构建球队强度评分，然后预测下一轮淘汰赛。

模型输出是概率估计，不是确定比分。足球比分噪声很大，结果只适合做概率参考。

## Current Method

- 训练集：2026 小组赛 + 32 进 16。
- 测试集：16 进 8 已结束比赛。
- 预测集：8 进 4比赛。
- 方法：对每队进攻、防守、小组/淘汰赛表现做对手强度校正，再用 Poisson 分布生成胜平负、晋级概率和最可能比分。
- 防过拟合：16 进 8只作为测试集，不参与训练；8 进 4只作为预测目标。

## Data

- `data/raw/worldcup_2026_group_stage.csv`: 2026 小组赛。
- `data/raw/worldcup_2026_round_of_32.csv`: 32 进 16。
- `data/raw/worldcup_2026_round_of_16.csv`: 16 进 8。
- `data/raw/worldcup_2026_quarter_finals.csv`: 8 进 4预测赛程。

## Run

```powershell
cd D:\Lottery\worldcup-prediction-model
python experiments\predict_quarter_finals.py
```

## Outputs

- `outputs/8进4预测.csv`: 8 进 4预测结果。
- `outputs/16进8测试集结果.csv`: 16 进 8测试集表现。
- `outputs/球队强度评分.csv`: 当前训练集下的球队评分。
- `outputs/quarter_final_prediction_report.md`: 方法、指标和预测摘要。

CSV 里的关键字段：

- `home_win`, `draw`, `away_win`: 90 分钟胜平负概率。
- `home_advancer_probability`, `away_advancer_probability`: 晋级概率，包含平局后加时/点球的抽象估计。
- `expected_home_goals`, `expected_away_goals`: 90 分钟期望进球。
- `top_scorelines`: 最可能的 90 分钟比分。
- `team_strength`: 对手强度校正后的球队综合评分。
- `schedule_strength`: 已赛对手强度，越高代表赛程越难。

## Notes

- 32 进 16中的晋级结果会给球队一个小幅 bonus，用来表示点球/加时晋级信息，但 16 进 8不会进入训练。
- 如果 CSV 正被 Excel 占用，脚本会自动写入 `_updated` 或带时间戳的文件。
