# Data dictionary

Three source tables, joined on `building_id` and `site_id` / `timestamp`.
Types below are the ones **declared** in `gridsmart.schemas`, schema
inference is never used, for the reasons in that module's docstring.

---

## `meters.csv`, energy readings

The time series. ~19 M rows, 626 MB.

| Column | Declared type | Notes |
|---|---|---|
| `building_id` | `IntegerType` (not null) | FK to buildings. Non-nullable in the schema because a NULL join key silently drops rows. |
| `meter_type` | `StringType` | One of `e`, `c`, `s`, `h`: electricity, chilled water, steam, hot water. |
| `ts` | `TimestampType` (not null) | Reading time, hourly. |
| `value` | `DecimalType(18,3)` | Consumption. **The metadata says decimal**; inference typed this as an integer whenever the sample missed the fractional rows. |
| `row_id` | `IntegerType` | Surrogate key, unused downstream. |

**Missing values:** present, caused by sensor and network failures. Handled
by the `Imputer` stage inside the pipeline, not dropped, dropping would bias
the training set toward buildings with reliable telemetry.

**Target construction:** the four meter types are **summed** per building per
timestamp, then aggregated into 6-hour blocks to form `agg_value`. Predicting
a single carrier would understate demand by roughly half at sites with
chilled-water cooling. → [ADR 0002](adr/0002-six-hour-aggregation-granularity.md)

---

## `building_information.csv`, static metadata

~1,400 rows.

| Column | Declared type | Notes |
|---|---|---|
| `site_id` | `IntegerType` (not null) | FK to weather. 16 distinct sites. |
| `building_id` | `IntegerType` (not null) | PK. |
| `primary_use` | `StringType` | `Education`, `Office`, `Retail`, … Indexed and one-hot encoded with `handleInvalid="keep"`. |
| `square_feet` | `IntegerType` | Gross floor area. Heavily right-skewed → used as `log1p(square_feet)`. |
| `floor_count` | `IntegerType` | Frequently null. |
| `year_built` | `IntegerType` | Frequently null. |
| `latent_y` / `latent_s` / `latent_r` | `DecimalType(18,6)` | Meaning withheld for privacy. Treated as anonymous numeric signals. |
| `row_id` | `IntegerType` | Surrogate key. |

**A note on the latent columns.** Their meaning is unknown by design, which
makes them a genuine ethics question rather than just a modelling one: a
feature whose semantics you cannot inspect is a feature whose bias you cannot
audit. They are retained only because they are already anonymised and carry
no identifiable content.

---

## `weather.csv`, site observations

Hourly, per site. ~140 K rows.

| Column | Declared type | Missing? | Notes |
|---|---|---|---|
| `site_id` | `IntegerType` (not null) |, | Join key. |
| `timestamp` | `TimestampType` (not null) |, | Observation time. |
| `air_temperature` | `DecimalType(10,2)` | rare | °C. Drives the peak/off-peak flag. |
| `cloud_coverage` | `IntegerType` | **frequent** | Oktas, 0–8. |
| `dew_temperature` | `DecimalType(10,2)` | rare | °C. Proxy for humidity → cooling load. |
| `sea_level_pressure` | `DecimalType(10,2)` | **frequent** | Millibars, ~1013 typical. |
| `wind_direction` | `IntegerType` | **frequent** | Degrees, 0–360. |
| `wind_speed` | `DecimalType(10,2)` | rare | m/s. |

### The three flavours of "missing"

The raw file encodes absent readings three ways, and only one of them is a
real NULL:

| Encoding | Treatment |
|---|---|
| empty field | → NULL |
| literal string `"null"` | → NULL |
| `0` | → NULL **for most columns** |

The zero rule needs care. A sea-level pressure of 0 mb is physically
impossible, so leaving it as data drags the imputed mean far below the true
value. But **0° wind direction is due north**: a perfectly valid bearing.
Blanket-nulling every zero would delete every genuine northerly reading, so
`wind_direction` is exempt.

`normalise_missing()` handles this, and both cases are covered by tests.

---

## Derived features

Produced by `features.engineer()`, which contains no `fit` and therefore
behaves identically on a bounded and an unbounded DataFrame.

| Feature | Derivation | Why |
|---|---|---|
| `agg_value` → `label` | Sum of 4 meters, bucketed to 6-hour blocks | The prediction target. |
| `interval` | Hour → one of four fixed daily blocks | The operator's dispatch unit. |
| `season_flag` | Per-site: 3 hottest + 3 coldest months = `peak` | Sites span both hemispheres; the calendar means different things at each. |
| `log_sqft` | `log1p(square_feet)` | Area spans orders of magnitude; `log1p` keeps the tail from dominating split selection and handles 0 safely. |
| `hour` | `hour(ts)` | Demand is strongly diurnal: the single most predictive feature. |
| `day_of_week` | `dayofweek(ts)` | Weekday/weekend occupancy shift. |
| `month` | `month(ts)` | Seasonal trend beyond the binary peak flag. |

### Deliberately excluded

`building_id` and `site_id` are carried through for joins and debugging but
are **never fed to the model**. Leaking an arbitrary identifier into a tree
ensemble lets it memorise individual buildings instead of learning the
drivers of consumption. It inflates test scores on known buildings and
collapses on any new one.
