# Datasets

The raw data is **not committed**: the meters table alone is 626 MB, well
past what belongs in a git repository. This directory is where the pipeline
expects to find it.

## Expected files

| File | Size | Rows | Used by |
|---|---|---|---|
| `meters.csv` | ~626 MB | ~19 M | batch training |
| `weather.csv` | ~7.6 MB | ~140 K | batch training, producer replay |
| `building_information.csv` | ~48 KB | ~1.4 K | batch training |
| `new_meters.csv` | ~296 MB | ~9 M | streaming, actual vs forecast |
| `new_building_information.csv` | ~22 KB | ~1.4 K | streaming, static join |

Override the location with `GRIDSMART_DATA_DIR` if you keep them elsewhere.

## Schema

Full column-level documentation, including which fields carry missing values
and how each is treated, is in [`../docs/data-dictionary.md`](../docs/data-dictionary.md).

## Provenance

The dataset combines real-world building energy measurements with synthetic
augmentation, covering 16 geographical sites that span several countries and
both hemispheres. The three `latent_*` columns in the building table are
deliberately unlabelled: their meaning was withheld for privacy reasons, so
they are treated purely as anonymous numeric signals.

That hemisphere spread is why the peak/off-peak feature is derived from each
site's own observed temperatures rather than from the calendar, see
[`../docs/adr/0002-six-hour-aggregation-granularity.md`](../docs/adr/0002-six-hour-aggregation-granularity.md)
and `gridsmart.features.add_season_flag`.
