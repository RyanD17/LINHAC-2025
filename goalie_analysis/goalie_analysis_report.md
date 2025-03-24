# Goalie Index Analysis Report

## Overview

This analysis evaluated 28 goalies, with 28 having at least 5 shots.

The Goalie Index is calculated using the formula:
```
Goalie Index = (Goals Saved Above Expected * 2) - (Dangerous Rebound % * 5) + (Save % * 3) + (Defensive Adjustment * 1)
```

This formula rewards:
- Goalies who outperform their expected goals against
- Goalies who control rebounds effectively
- Goalies who maintain a high save percentage
- Goalies who perform well under defensive pressure

## Top Performing Goalies

| Goalie ID | Total Shots | Save % | Goals Saved Above Expected | Dangerous Rebound % | Goalie Index |
|-----------|-------------|--------|---------------------------|---------------------|-------------|
| 628001 | 1316 | 0.426 | 701.14 | 0.311 | 1402.50 |
| 869001 | 1325 | 0.429 | 693.55 | 0.303 | 1387.37 |
| 814001 | 1308 | 0.435 | 679.24 | 0.289 | 1358.84 |
| 524001 | 1284 | 0.424 | 678.92 | 0.313 | 1358.05 |
| 877001 | 1257 | 0.422 | 672.60 | 0.329 | 1345.32 |
| 503001 | 1304 | 0.434 | 672.42 | 0.311 | 1345.09 |
| 885001 | 1249 | 0.424 | 661.44 | 0.308 | 1323.11 |
| 726001 | 1371 | 0.484 | 644.80 | 0.313 | 1290.00 |
| 855001 | 1168 | 0.407 | 639.76 | 0.324 | 1279.63 |
| 795001 | 1205 | 0.440 | 626.36 | 0.304 | 1253.03 |

## Key Findings

- Average save percentage: 0.499
- Average goals saved above expected: 84.64
- Average dangerous rebound percentage: 0.308

### Correlation Analysis

- Correlation between save percentage and goals saved above expected: -0.953
- Correlation between save percentage and dangerous rebound percentage: 0.055

## Recommendations

1. For more accurate analysis, collect more shot data with complete location information.
2. Consider tracking rebounds that result in high-danger chances even if no shot is recorded.
3. Develop a goalie scouting system that weights these metrics based on team defensive structure.
4. Analyze goalie performance against shot quality to identify specialists.
5. Track goalie fatigue by analyzing performance changes over time and with shot volume.
