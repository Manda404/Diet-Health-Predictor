# Test Fixtures

## `mock_diet_data.csv`

A small (17-row), hand-crafted dataset with the same schema as
`data/healthy_diet_calorie_intake.csv`, used by every unit and integration test
that needs data. It is deliberately not a random sample of the real dataset —
each row exists to exercise a specific edge case, and the values are legible
enough to reason about by hand in test assertions:

| Row(s)                | Purpose                                                                 |
|------------------------|--------------------------------------------------------------------------|
| `P0001` (appears twice) | Exact duplicate `Person_ID` -> `DataCleaner` deduplication              |
| `P0012`                | Missing `Weight_kg` -> numeric imputation (median)                      |
| `P0009`                | Missing `Activity_Level` -> categorical imputation (mode)               |
| `P0013`                | Missing `Fat_Intake_g` -> numeric imputation (median)                   |
| `P0014`                | `Fat_Intake_g = 900` (real range ~45-90) -> IQR outlier clipping        |
| `P0007` / `P0011`      | Age 17 / 18 -> Child / Young Adult boundary                             |
| `P0004` / `P0002`      | Age 22 / 30 -> Young Adult / Adult boundary (also `P0009` at 29)        |
| `P0010` / `P0003`      | Age 59 / 60 -> Adult / Senior boundary                                  |
| `P0005`                | `Gender = Other` -> exercises the averaged BMR / ideal-weight branch    |

Do not "clean up" the duplicate, the missing values, or the outlier — removing
them silently breaks the tests that depend on them. If you need a fixture
*without* these issues (e.g. to test a happy path in isolation), build a small
DataFrame inline in that test instead of editing this file.

Class balance after deduplication (16 unique persons): `Healthy=8`,
`Overweight=4`, `Obese=2`, `Underweight=2` — every class has at least 2 members,
which is the minimum `sklearn.model_selection.train_test_split(stratify=...)`
needs to place at least one member of each class in both the train and test
split.
