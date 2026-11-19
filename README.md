# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/temu404-cyber/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/temu404-cyber/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PlayStation 5 trophy's Exophase earn rate at the case's target, as five quantiles per trophy. A rate is a ratio, so the model forecasts its two halves separately: tracked players N(t) and earners E(t) each follow a power law in hours since release over the case's last 24 hours of reads, and each trophy's logit rate is also fitted against the shared progress clock (average trophies earned per tracked player), which saturates as players run out of trophies to earn. The median blends the logit-trend, earners-over-players, and progression-clock forecasts (0.30 / 0.30 / 0.40), shrunk 15% toward the last observed rate. The spread combines a random-walk term widening with the horizon, the trend fit's residual, and the size of Exophase's refresh jumps. When a case's history never moves (the practice cases' frozen launch-night page), the forecast uses the refresh the rules document — 638 players tracked refreshing to 3,868, every rate but the 100% one lower — scaling each rate by that player ratio with a 0.5-logit spread. Deterministic, offline, Node 22, standard library only. On synthetic week-one cases it beats the house baseline's pinball loss (about 0.27 vs 0.33) and halves its typical miss.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=8wf5mahh&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
