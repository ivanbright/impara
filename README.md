# impara

**Don't generate ideas. Discover problems.**

Impara is a developer tool for finding **real problems worth solving**. It
surfaces recurring problems from public signals, keeps the evidence attached to
each problem, and tells you honestly when there is not enough of it.

It is deliberately *not* a startup-idea generator. Impara is comfortable saying
"This is probably not worth building."

## Install

    pip install impara

Requires Python 3.8+.

## Quick start

```bash
impara discover                    # corroborated problems (2+ signals)
impara discover --all              # include single-signal leads
impara discover --market education
impara discover --country rwanda
impara discover --deep             # also scan the rest of each matched thread
impara discover --json             # machine-readable output

impara investigate #001            # full problem profile + evidence
impara validate   #001             # what to check before building
impara define     #001             # narrow MVP, features deliberately excluded

impara sources                     # live source health
impara corpus                      # accumulated signal evidence
impara corpus --clear              # reset the local corpus
impara list                        # re-show the last run
```

### Useful flags

| Flag | Effect |
|---|---|
| `--all` | also report single-signal leads (default hides them) |
| `--min-signals N` | corroboration floor, default `2` |
| `--min-evidence weak\|moderate\|strong` | filter by evidence strength |
| `--deep` | expand matched threads for more comments (slower) |
| `--fresh` | ignore the saved corpus, use only this run |
| `--json` | structured output on every command |
| `--source github\|hackernews\|stackexchange` | restrict sources (repeatable) |

### Evidence accumulates

Every verified signal is kept in a local corpus (`impara corpus`), so repeated
runs build corroboration over time instead of re-deriving a scattered snapshot.
That corpus is local to your machine and never uploaded.

## The workflow

```text
DISCOVER  →  UNDERSTAND  →  VALIDATE  →  DEFINE  →  BUILD
```

## Evidence first

Nothing counts as evidence unless a complaint pattern is **literally present**
in text fetched from a real source. Search rankings do not promote a candidate.

```text
REAL SIGNALS → EXTRACT → CLUSTER → ASSESS EVIDENCE → ANALYSE → PROPOSE
```

Every scoring dimension is labelled `observed` or `inferred`. Inference is
never presented as fact, and weak evidence is reported as **insufficient
evidence** rather than hidden.

## Signal sources

| Source | Status |
|---|---|
| GitHub issues | public, no auth |
| Hacker News (Algolia) | public, no auth |
| Stack Overflow (Stack Exchange) | public, no auth |
| Reddit | **not queried** — anonymous API access is blocked |

`impara sources` reports live health. If a source is unreachable it is marked
unavailable and contributes zero signals.

### What to expect

Complaint patterns are genuinely rare in public discussion data — a scan of
~3,700 real comments yielded ~19 that matched. That is the point: Impara
reports **insufficient evidence** rather than padding a list. Most runs will
surface a small number of corroborated problems, and many will still be marked
unverified. Investigate them before you build.

## Output principles

- Analytical, evidence-driven, honest, developer-oriented.
- Never optimises for the number of ideas. It optimises for the number of
  opportunities that make you say *"I actually want to investigate this."*
- Structured, machine-readable output available everywhere via `--json`.

## Backward compatibility

The arithmetic helpers from 0.2.0 still work:

```python
from impara import add, multiply, average
add(10, 25)                 # 35
average([10, 20, 30, 40])   # 25.0
```

```bash
impara add 10 25            # 35
```

## License

MIT
