// GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).
//
// Forecasts, for every GTA VI PlayStation 5 trophy, its earn rate on Exophase at
// the case's target, as five quantiles (levels 0.05, 0.25, 0.5, 0.75, 0.95).
// Runs offline, Node 22, standard library only, fully deterministic: the same
// case file always produces the same bytes.
//
// The model — why it beats "the rate stays where it is":
// Exophase's tracked-player count explodes after launch while players also make
// progress, so a trophy's earn rate is pulled in two directions at once: new
// players dilute it (the denominator grows) and existing players earning the
// trophy push it up. A rate is a ratio, so the forecast models the ratio's two
// halves separately and then recombines them:
//
//   * players N(t): log N grows about linearly in log(1 + hours since release)
//     (a power law) over the case's last 24 hours of reads;
//   * earners E_i(t) = rate_i * N / 100: log E_i grows about linearly in the
//     same clock, again over the last 24 hours;
//   * progress p(t) = trophies_earned / N, the average trophies per tracked
//     player: a shared clock for how far the player base has got. Each
//     trophy's logit rate is also fitted against p, which saturates as players
//     run out of trophies to earn, so this leg of the forecast saturates too.
//
// Three medians come out of that — a logit-trend extrapolation, an
// earners-over-players decomposition, and a progression-clock extrapolation —
// and the forecast median is their weighted mean on the logit scale (0.30 /
// 0.30 / 0.40), shrunk 15% toward the last observed rate as a hedge against
// runaway extrapolation over a 4.5-day horizon, and capped so its odds move at
// most 50x from the odds the case has ever shown for that trophy.
//
// The spread is honest about a launch week: a random-walk term that widens
// with the square root of the horizon (the house baseline's 0.5 logit units
// per root day), plus the trend fit's residual spread, plus half the spread of
// the jumps between consecutive distinct reads (Exophase refreshes in jumps:
// the house saw the page unchanged for hours, then 638 -> 3,868 players
// tracked in one refresh, every rate but the 100% one lower).
//
// When a case's history never changes (the practice cases: every read before
// the checkpoint is the same launch-night page), no trend can be fitted. The
// forecast then uses the refresh the rules document — the page sat at 638
// players tracked, then refreshed to 3,868 with every rate but the 100% one
// lower — scaling the last rate by that player ratio (638/3868), with a
// spread for earners having grown during the stale window.
//
// Only the case file is read: no network, no clock, no randomness. Reads are
// compressed to one observation per distinct Exophase refresh (consecutive
// identical reads carry no information), and every fit uses the last 24 hours
// of reads, or all of them when the case spans less than that.
//
// Usage (from the kit's directory, as the starter documents):
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/**
 * Players tracked after the launch night's first refresh, as the rules
 * document it: the page sat at 638 players tracked, then refreshed to 3,868
 * with every rate but the 100% one lower. Used only as the prior for a case
 * whose history never moved (see the frozen-page branch in forecastCase).
 */
const REFRESH_PLAYERS = 3868;

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: Trophy[];
  history: Read[];
  notes?: string;
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const clamp01 = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => {
  if (l >= 0) return (100 / (1 + Math.exp(-l)));
  const e = Math.exp(l);
  return (100 * e) / (1 + e);
};
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Ordinary least squares of y on x. Returns null when the fit is degenerate. */
function ols(xs: number[], ys: number[]): { a: number; b: number; residSd: number } | null {
  const n = xs.length;
  if (n < 2) return null;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx <= 0) return null;
  const b = sxy / sxx;
  const a = my - b * mx;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (a + b * xs[i]);
    ss += r * r;
  }
  const residSd = n > 2 ? Math.sqrt(ss / (n - 2)) : 0.2;
  return { a, b, residSd };
}

function sd(values: number[]): number {
  const n = values.length;
  if (n < 2) return 0;
  let m = 0;
  for (const v of values) m += v;
  m /= n;
  let ss = 0;
  for (const v of values) ss += (v - m) * (v - m);
  return Math.sqrt(ss / (n - 1));
}

function forecastCase(c: Case): { trophies: Array<{ id: number; q: number[] }> } {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${c.id} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${c.id} has no history`);
  const releaseMs = Date.parse(`${c.game.release_date}T00:00:00Z`);
  const checkpointMs = Date.parse(c.at);
  const targetMs = Date.parse(c.target);
  const days = c.horizon_minutes / 1440;
  const hours = (ms: number): number => (ms - releaseMs) / 3_600_000;
  const xOf = (ms: number): number => Math.log(1 + hours(ms));
  const xTarget = xOf(targetMs);

  // One observation per distinct Exophase refresh: drop consecutive identical
  // reads (same players tracked, same trophies earned, same rates).
  const distinct: Read[] = [];
  for (const r of c.history) {
    const prev = distinct[distinct.length - 1];
    if (
      prev !== undefined &&
      prev.players_tracked === r.players_tracked &&
      prev.trophies_earned === r.trophies_earned &&
      prev.rates.length === r.rates.length &&
      prev.rates.every((v, i) => v === r.rates[i])
    ) {
      continue;
    }
    distinct.push(r);
  }

  // Fit on the last 24 hours of reads; the whole case when it spans less.
  const windowStart = checkpointMs - 24 * 3_600_000;
  let fitReads = distinct.filter((r) => Date.parse(r.at) >= windowStart);
  if (fitReads.length < 3) fitReads = distinct;
  const lastRead = distinct[distinct.length - 1];
  const lastN = lastRead.players_tracked;
  const lastP = lastRead.players_tracked > 0 ? lastRead.trophies_earned / lastRead.players_tracked : 0;

  const xs = fitReads.map((r) => xOf(Date.parse(r.at)));
  // Players tracked: log N against log(1 + hours since release).
  const lnN = fitReads.map((r) => Math.log(Math.max(1, r.players_tracked)));
  const fitN = ols(xs, lnN);
  let nTarget = lastN;
  if (fitN !== null) {
    const bN = clamp01(fitN.b, 0, 1.2);
    nTarget = Math.max(lastN, Math.exp(fitN.a + bN * xTarget));
  }
  // Progress clock: average trophies earned per tracked player.
  const ps = fitReads.map((r) => (r.players_tracked > 0 ? r.trophies_earned / r.players_tracked : 0));
  const fitP = ols(xs, ps);
  let pTarget = lastP;
  if (fitP !== null) {
    pTarget = clamp01(fitP.a + fitP.b * xTarget, 0.3 * lastP, 55);
  }

  const rows: Array<{ id: number; q: number[] }> = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const trophy = c.trophies[i];
    const toPoints = (reads: Read[], xsAll: number[], psAll: number[]) =>
      reads
        .map((r, j) => ({ x: xsAll[j], n: r.players_tracked, p: psAll[j], rate: r.rates[i] ?? null }))
        .filter((w) => w.rate !== null) as Array<{ x: number; n: number; p: number; rate: number }>;
    // Fit on the last 24 hours; but when a trophy's rate stopped moving in
    // that window (it saturated, or the page went stale), fall back to the
    // whole history so the fit still sees how the trophy got there. Only a
    // trophy whose rate never moved at all takes the frozen-page branch.
    let pts = toPoints(fitReads, xs, ps);
    if (new Set(pts.map((w) => w.rate)).size < 2) {
      const xsAll = distinct.map((r) => xOf(Date.parse(r.at)));
      const psAll = distinct.map((r) => (r.players_tracked > 0 ? r.trophies_earned / r.players_tracked : 0));
      pts = toPoints(distinct, xsAll, psAll);
    }
    const rLastRaw = lastRead.rates[i] ?? null;
    const rLast = rLastRaw !== null ? rLastRaw : pts.length > 0 ? pts[pts.length - 1].rate : null;

    const quantiles = (median: number, sigma: number): number[] => {
      const l = logit(median);
      let prev = -Infinity;
      return Z.map((z) => {
        const v = clampPct(invLogit(l + z * sigma));
        const r = round4(Math.max(prev, v));
        prev = r;
        return r;
      });
    };

    // No rate ever observed for this trophy.
    if (rLast === null || pts.length === 0) {
      rows.push({ id: trophy.id, q: quantiles(0.05, 1.6) });
      continue;
    }
    const distinctRates = new Set(pts.map((w) => w.rate)).size;
    const flat = distinctRates <= 1 || pts.length < 3;

    if (flat) {
      // The page never moved before the checkpoint: the next Exophase refresh
      // is expected to dilute the rate. The rules document how the launch
      // night's first refresh behaved: the page sat unchanged at 638 players
      // tracked, then refreshed to 3,868 players with every rate but the 100%
      // one lower — the earners counted were essentially those of the last
      // snapshot while the tracked base jumped. So the median scales the last
      // rate by the player ratio of that documented refresh (first read's
      // players over 3,868), and a trophy everybody already has stays near
      // everybody, as the rules say it did. The spread (0.5 logit units)
      // covers earners having grown during the stale window and the next
      // jump differing in size. This branch only fires when a case's history
      // never moved at all; a case with any movement (the final's 2.5 days of
      // reads) takes the trend fits below instead.
      if (rLast >= 99.99) {
        rows.push({ id: trophy.id, q: quantiles(99.99, 0.5) });
      } else if (rLast <= 0.01) {
        rows.push({ id: trophy.id, q: quantiles(0.05, 1.6) });
      } else {
        const firstN = distinct[0].players_tracked;
        const scale = firstN > 0 && firstN < REFRESH_PLAYERS ? firstN / REFRESH_PLAYERS : 0.5;
        rows.push({ id: trophy.id, q: quantiles(rLast * scale, 0.5) });
      }
      continue;
    }

    const ys = pts.map((w) => logit(w.rate));
    const es = pts.map((w) => Math.log((w.rate * w.n) / 100 + 0.5));
    const px = pts.map((w) => w.x);
    const pp = pts.map((w) => w.p);
    const fitTrend = ols(px, ys);
    const fitEarn = ols(px, es);
    const fitProg = ols(pp, ys);

    // Odds may move at most 50x from the odds this trophy has ever shown.
    const oddsSeen = pts.map((w) => {
      const p = clampPct(w.rate) / 100;
      return p / (1 - p);
    });
    const oddsMax = Math.max(...oddsSeen) * 50;
    const oddsMin = Math.min(...oddsSeen) / 50;
    const capOdds = (medianPct: number): number => {
      const p = clampPct(medianPct) / 100;
      const o = clamp01(p / (1 - p), oddsMin, oddsMax);
      return 100 * (o / (1 + o));
    };

    let logitMedian: number | null = null;
    let resid = 0.2;
    if (fitTrend !== null && fitEarn !== null && fitProg !== null) {
      const mTrend = capOdds(invLogit(fitTrend.a + fitTrend.b * xTarget));
      const eTarget = Math.max(0, Math.exp(fitEarn.a + fitEarn.b * xTarget) - 0.5);
      const mEarn = capOdds((100 * eTarget) / nTarget);
      const mProg = capOdds(invLogit(fitProg.a + fitProg.b * pTarget));
      logitMedian =
        0.3 * logit(mTrend) + 0.3 * logit(mEarn) + 0.4 * logit(mProg);
      // Hedge 15% toward the last observed rate.
      logitMedian = 0.85 * logitMedian + 0.15 * logit(rLast);
      resid = clamp01(fitTrend.residSd, 0.15, 0.9);
    }
    if (logitMedian === null) {
      rows.push({ id: trophy.id, q: quantiles(rLast * 0.5, 1.6) });
      continue;
    }

    // Spread: random-walk widening with horizon, fit residual, refresh jumps.
    const diffs: number[] = [];
    for (let j = 1; j < ys.length; j++) diffs.push(ys[j] - ys[j - 1]);
    const jump = clamp01(sd(diffs), 0, 0.8);
    const rw = 0.5 * Math.sqrt(days);
    let sigma = Math.sqrt(rw * rw + resid * resid + (0.5 * jump) * (0.5 * jump));
    sigma = clamp01(sigma, 0.4, 2.2);

    let median = invLogit(logitMedian);
    if (rLast >= 99.99) {
      median = 98;
      sigma = Math.max(sigma, 0.8);
    } else if (rLast <= 0.01) {
      median = Math.max(median, 0.05);
      sigma = Math.max(sigma, 1.5);
    }
    rows.push({ id: trophy.id, q: quantiles(median, sigma) });
  }
  rows.sort((a, b) => a.id - b.id);
  return { trophies: rows };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const out = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
