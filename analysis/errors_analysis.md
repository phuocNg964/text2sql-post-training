# Error Analysis — Qwen2.5-Coder-3B-Instruct (Spider dev, n=100)

**Baseline EX**: 59.0% · **Failed**: 41/100 · **Invalid SQL**: 21 · **Wrong Answer**: 20

---

## Error Taxonomy Summary

| Category | Count | % of Errors | Subtypes |
|---|:---:|:---:|---|
| Unnecessary JOIN | 21 | 51% | Single-table query solved with spurious joins |
| Negation / Exclusion Logic | 5 | 12% | NOT IN vs != on same row, EXCEPT vs filter |
| Wrong Join Path / Schema Linking | 4 | 10% | Wrong FK, wrong table for column |
| Aggregation / GROUP BY | 4 | 10% | Missing aggregation, wrong aggregation target |
| Quantifier / Comparison Reasoning | 3 | 7% | ANY vs ALL, wrong comparison target |
| Wrong Output Columns | 2 | 5% | Returns descriptions instead of codes, missing DISTINCT |
| Filter Value / Casing | 2 | 5% | Case mismatch in string literals |

---

## Category 1 — Unnecessary JOIN (21 errors)

**The dominant failure mode.** The model adds JOINs to unrelated tables when the answer requires only a single table (or fewer tables than predicted). This causes `invalid_sql` when the joined table doesn't have the expected columns, or `wrong_answer` when the JOIN changes cardinality.

**SFT implication**: The model needs to learn that when all requested columns exist in one table, no JOIN is needed.

| # | Type | DB | Question (abbreviated) | Issue |
|---|---|---|---|---|
| 1 | invalid_sql | flight_2 | Airport with least flights | Drops the AIRPORTS JOIN, uses bare `ORDER BY COUNT(*)` on airports without GROUP BY |
| 3 | wrong_answer | battle_death | Battles between Kaloyan and Baldwin I | Adds unnecessary JOIN to `ship` table; only `battle` table needed |
| 5 | wrong_answer | network_1 | Names and grades of each high schooler | JOINs `Friend` table; only `Highschooler` needed → changes result set |
| 9 | invalid_sql | world_1 | Avg life expectancy in Central Africa | JOINs `countrylanguage`; only `country` needed → column mismatch |
| 10 | invalid_sql | student_transcripts_tracking | Earliest school graduate name | JOINs `Student_Enrolment` + `Transcripts` instead of simply ordering `Students` |
| 12 | wrong_answer | tvshow | Count of English TV channels | JOINs `Cartoon`; only `TV_Channel` needed → inflates count |
| 13 | invalid_sql | orchestra | Record companies by founding year | JOINs `performance`; only `orchestra` needed → wrong columns |
| 15 | invalid_sql | concert_singer | Song by youngest singer | JOINs `concert`; only `singer` needed → wrong columns |
| 17 | invalid_sql | world_1 | Population and life expectancy of Brazil | JOINs `city`; only `country` needed → returns city-level data |
| 22 | wrong_answer | course_teach | Teachers aged 32 or 33 | JOINs `course_arrange`; only `teacher` needed → misses teachers without courses |
| 24 | invalid_sql | network_1 | Grade of each high schooler | JOINs `Friend`; only `Highschooler` needed → wrong columns |
| 25 | invalid_sql | employee_hire_evaluation | Manager name/district of top shop | JOINs `employee`; only `shop` needed → wrong columns |
| 28 | wrong_answer | course_teach | Teachers not from Little Lever | JOINs `course_arrange`; only `teacher` needed → misses teachers without courses |
| 32 | invalid_sql | tvshow | Cartoon titles/directors by air date | JOINs `TV_Channel`; `Cartoon.Original_air_date` suffices |
| 33 | invalid_sql | world_1 | Country in Asia with lowest life expectancy | JOINs `countrylanguage`; only `country` needed |
| 34 | invalid_sql | pets_1 | Avg and max age per pet type | JOINs `Students` + `Has_Pet`; only `Pets` needed → computes student age instead of pet age |
| 35 | invalid_sql | world_1 | Name/population/head of state for largest country | JOINs `city`; only `country` needed → returns city-level data |
| 36 | invalid_sql | car_1 | Distinct car models produced after 1980 | Wrong join path through `car_names.MakeId` instead of `model` column |
| 39 | invalid_sql | car_1 | 4-cylinder car with most horsepower | Wrong join path; adds unnecessary GROUP BY + SUM |
| 41 | invalid_sql | voter_1 | Vote details for contestant Tabatha Gehling | Adds unnecessary JOIN to `AREA_CODE_STATE`; mixes up column sources |
| 21 | invalid_sql | car_1 | Different models for cars after 1980 | Same as #36 — wrong join path through MakeId |

---

## Category 2 — Negation / Exclusion Logic (5 errors)

**The model confuses row-level filtering (`!=`) with set-level exclusion (`NOT IN` / `EXCEPT`).** This is a reasoning failure: "students who don't have a cat" ≠ "rows where pet type is not cat".

**SFT implication**: Training examples should emphasize NOT IN subqueries and EXCEPT patterns for exclusion semantics.

| # | Type | DB | Question (abbreviated) | Issue |
|---|---|---|---|---|
| 6 | wrong_answer | pets_1 | Students with dog but not cat | Uses `PetType = 'Dog' AND PetType != 'Cat'` (always true when Dog) instead of NOT IN subquery |
| 7 | wrong_answer | pets_1 | Students without a cat pet | Uses `PetType != 'cat'` filter instead of NOT IN subquery; returns students with other pets rather than students without cats |
| 29 | wrong_answer | tvshow | TV channels that don't play Ben Jones cartoons | Uses `Directed_by != 'Ben Jones'` instead of EXCEPT; returns channels that play *any* non-Ben-Jones cartoon |
| 14 | wrong_answer | network_1 | Kyle's friends | Filters `WHERE T2.name = 'Kyle'` on the friend side instead of the student side; returns Kyle instead of Kyle's friends |
| 26 | wrong_answer | student_transcripts_tracking | Students in NC not in degree program | Uses `NOT IN` (semantically close) but different string value `'North Carolina'` vs `'NorthCarolina'` |

---

## Category 3 — Wrong Join Path / Schema Linking (4 errors)

**The model picks the wrong foreign key or joins through the wrong intermediate table.** Often the `car_1` database schema (with its `car_names.MakeId` vs `model_list.Model` distinction) trips it up.

**SFT implication**: Schema representation in the prompt must make foreign key relationships unambiguous.

| # | Type | DB | Question (abbreviated) | Issue |
|---|---|---|---|---|
| 4 | invalid_sql | car_1 | Countries with >3 makers or Fiat | Uses `WHERE COUNT()` in a non-aggregated query instead of HAVING; collapses UNION into single query |
| 8 | invalid_sql | car_1 | Car model with min horsepower | Joins on `MakeId = MakeId` instead of correct FK path |
| 23 | invalid_sql | car_1 | Makers that produced cars in 1970 | Joins `car_names` directly to `cars_data` on wrong FK; misses `car_makers` → `model_list` path |
| 27 | invalid_sql | dog_kennels | Professionals with below-avg treatment cost | Hallucinates a `Breeds` join and `'Golden Retriever'` filter that doesn't exist in the question |

---

## Category 4 — Aggregation / GROUP BY (4 errors)

**The model misapplies aggregation**: counting rows instead of reading a column value, grouping at the wrong level, or missing GROUP BY entirely.

**SFT implication**: Demonstrate correct aggregation patterns (COUNT vs column value, GROUP BY + HAVING, subquery aggregation).

| # | Type | DB | Question (abbreviated) | Issue |
|---|---|---|---|---|
| 2 | wrong_answer | student_transcripts_tracking | Average transcript print date | Adds GROUP BY per transcript instead of a simple AVG over all rows |
| 11 | wrong_answer | wta_1 | Player with most tours | Uses COUNT(*) of ranking rows instead of reading the `tours` column value |
| 20 | wrong_answer | cre_Doc_Template_Mgt | Most common template type in documents | Groups within `Templates` only, ignoring the JOIN to `Documents` which determines actual usage |
| 31 | invalid_sql | network_1 | Student IDs and friend counts | Adds unnecessary JOIN to `Highschooler`; simple `GROUP BY student_id` on `Friend` suffices |

---

## Category 5 — Quantifier / Comparison Reasoning (3 errors)

**The model confuses "any" (min) vs "all" (max), or compares the wrong column in a subquery.**

**SFT implication**: Include examples with explicit ANY/ALL quantifier patterns and cross-column comparisons.

| # | Type | DB | Question (abbreviated) | Issue |
|---|---|---|---|---|
| 16 | wrong_answer | world_1 | Asian countries with population > any African country | Uses `MAX(Population)` instead of `MIN(Population)` — "any" means greater than at least one |
| 18 | wrong_answer | car_1 | Cars with accelerate > car with largest HP | Compares Accelerate against `MAX(Horsepower)` instead of the Accelerate *of* the highest-HP car |
| 38 | wrong_answer | wta_1 | Winner with most matches + rank points | Builds overly complex subquery structure; fails to use direct `winner_name` / `winner_rank_points` columns |

---

## Category 6 — Wrong Output Columns (2 errors)

**The model returns descriptive names instead of codes, or misses DISTINCT.**

| # | Type | DB | Question (abbreviated) | Issue |
|---|---|---|---|---|
| 37 | wrong_answer | dog_kennels | All breed type and size type combinations | Returns `breed_name` + `size_description` instead of `breed_code` + `size_code` |
| 19 | wrong_answer | world_1 | Country codes for non-English speakers | Returns `country.Code` via a JOIN instead of `countrylanguage.CountryCode` directly; missing DISTINCT |

---

## Category 7 — Filter Value / Casing (2 errors)

**The query logic is correct but the string literal has wrong casing, causing an empty result.**

| # | Type | DB | Question (abbreviated) | Issue |
|---|---|---|---|---|
| 40 | wrong_answer | flight_2 | Number of JetBlue Airways flights | Uses `'Jetblue Airways'` (lowercase 'b') instead of `'JetBlue Airways'` |
| 30 | wrong_answer | network_1 | High schooler with most likes | Groups by `T2.name` instead of `T1.student_id`; could return wrong result if names collide |

---

## SFT Priority Matrix

Based on frequency and fix difficulty, ordered by training priority:

| Priority | Category | Count | SFT Strategy |
|:---:|---|:---:|---|
| **P0** | Unnecessary JOIN | 21 | Teach single-table query patterns. Include many examples where all columns are in one table and the gold SQL has no JOIN. |
| **P1** | Negation / Exclusion | 5 | Teach NOT IN subquery and EXCEPT patterns. Contrast with row-level `!=` filtering. |
| **P2** | Schema Linking / Join Path | 4 | Improve schema representation in prompts (explicit FK annotations). Include `car_1`-style multi-hop join examples. |
| **P3** | Aggregation / GROUP BY | 4 | Teach when to aggregate vs. when to read a column directly. GROUP BY + HAVING examples. |
| **P4** | Quantifier Reasoning | 3 | Include ANY/ALL subquery examples. Cross-column comparison patterns. |
| **P5** | Output Columns | 2 | Teach code vs. name column selection based on question wording. |
| **P6** | Filter Value / Casing | 2 | Hard to fix via SFT alone; may need schema value hints in prompt or case-insensitive execution. |