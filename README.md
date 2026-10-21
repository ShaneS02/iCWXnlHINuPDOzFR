# Semantic Candidate Matching and Ranking System

## Overview

This project implements a semantic candidate-matching pipeline to help recruiters identify and rank candidate profiles based on a job-title query. It compares job-title meaning, and produce a ranked shortlist.

The pipeline also includes a feedback-based re-ranking stage. A recruiter can mark one or more candidates as preferred, and the system uses those examples to adjust the ranking based on similarities in job title, job sector, seniority, city, and region.

## Features

- Cleans and standardizes raw job-title text.
- Uses semantic embeddings to map job titles to a predefined set of standardized titles.
- Assigns candidates to broad job sectors.
- Parses location strings into city and region fields.
- Estimates seniority from job-title semantics and connection counts.
- Ranks candidates against a recruiter-provided job-title query.
- Applies a score threshold and returns up to a configurable number of candidates.
- Re-ranks candidates using recruiter-starred examples and multiple candidate attributes.

## Workflow

The notebook follows this high-level sequence:

1. Load and clean candidate data.
2. Generate semantic features and structured candidate attributes.
3. Create embeddings for candidate job titles.
4. Retrieve and rank candidates for a job-title query.
5. Use recruiter feedback to re-rank the shortlist.

## How the Process Was Implemented

### 1. Load and clean the data

The notebook reads candidate data from `../data/data.csv`. It cleans the `job_title` field by removing punctuation and special characters, converting text to lowercase, and normalizing repeated whitespace.

This reduces superficial differences between titles before semantic processing. The pipeline then creates a structured dataframe containing the candidate `id` and cleaned `job_title`, which is enriched in subsequent stages.

### 2. Load the pretrained embedding model

The notebook loads `SentenceTransformer("all-MiniLM-L6-v2")`. This pretrained model converts text into numerical vectors, or embeddings, that represent semantic similarity.

Embeddings are generated with `normalize_embeddings=True`. With normalized vectors, a dot product can be used as a cosine-similarity measure. This lets the system compare job titles even when their wording is not identical.

### 3. Standardize job titles and identify job sectors

A dictionary that maps a set of known job titles to broad sectors was implemented. The categories include:

- Technology
- Finance
- Human resources
- Customer service
- Engineering
- Education
- Sales and marketing
- Operations
- Healthcare
- Other

The model embeds both the standardized title list and each cleaned candidate job title. It calculates similarities between candidate titles and the standardized title embeddings, selects the closest title, and assigns that title when the best similarity score is at least `0.40`. Otherwise, the candidate receives the standardized title `Other`.

The sector is then assigned from the selected standardized title using the mapping dictionary. This creates two structured features: `concise_job_title` and `job_sector`.

### 4. Parse location information

The notebook cleans location strings by replacing a trailing `Kanada` with `Canada` and trimming whitespace. For locations containing a comma, it splits the string into `city` and `region`. When no comma is present, the full location string is assigned to `region`, while `city` remains missing.

### 5. Estimate candidate seniority

The notebook creates lists of seniority-related terms for senior and junior roles, then embeds those terms with the same pretrained model. Each candidate's original job title is embedded and compared with the senior and junior term embeddings.

For each candidate, the maximum similarity to the senior terms and the maximum similarity to the junior terms are calculated. The scores are then adjusted using the candidate's connection count:

- A seniority bonus of `0.08` is applied to the senior score for at least 500 connections, or `0.02` for at least 200 connections.
- A juniority bonus of `0.08` is applied to the junior score for at most 100 connections, or `0.02` for fewer than 500 connections.

Using a threshold of `0.50`, a candidate is labelled `senior` if the adjusted senior score passes the threshold and exceeds the adjusted junior score. The equivalent condition assigns `junior`. All other candidates default to `mid`.

### 6. Search and rank candidates

The `semantic_search` function accepts a query, candidate dataframe, precomputed embeddings, and ranking parameters. Its default settings are a threshold of `0.50` and `top_k=100`.

For each query, the function:

1. Generates an embedding for the recruiter query.
2. Calculates similarity against each candidate's standardized title embedding.
3. Calculates similarity against each candidate's original title embedding.
4. Finds the closest standardized title to the query and uses it to infer the query sector.
5. Assigns a sector-match value of `1` when a candidate's sector matches the inferred sector, otherwise `0`.
6. Combines the scores using the following formula:

   `fit = 0.7 × concise_similarity + 0.2 × original_similarity + 0.1 × sector_match`

7. Filters out candidates below the score threshold.

The returned results include the individual similarity components and the combined `fit` score, making the ranking easier to inspect.

### 7 Incorporate recruiter feedback and re-rank candidates

The `candidate_feedback_similarity` function introduces a second ranking stage. It accepts the current result dataframe, embeddings for the candidates in that dataframe, and the positions of candidates selected by the recruiter.

The function calculates how similar each candidate is to each starred example using five signals:

| Signal                         | Weight |
| ------------------------------ | -----: |
| Job-title embedding similarity |   0.45 |
| Job-sector match               |   0.20 |
| Seniority similarity           |   0.15 |
| City match                     |   0.10 |
| Region match                   |   0.10 |

For seniority, the labels are converted to an ordered scale: junior = 1, mid = 2, senior = 3. Similarity decreases as the difference between seniority levels increases.

For sector, city, and region, a match scores `1` and a mismatch scores `0`. Missing categorical or seniority values are excluded from that pair's weighted calculation rather than automatically counted as mismatches. The available weights are re-normalized through the score denominator.

When multiple candidates are starred, the function calculates scores against each starred candidate and keeps the highest pairwise feedback score for each candidate. It then combines the original ranking score and feedback score:

`re_ranked_score = 0.7 × fit + 0.3 × feedback_similarity`

Finally, starred candidates are placed first, with the remaining ordering determined by the re-ranked score. This makes the feedback stage a form of example-based re-ranking rather than model training.

**Position handling:** `starred_positions` uses zero-based row positions within the dataframe passed to the function, not candidate IDs or the displayed one-based row numbers. The notebook's example uses `[1]`, which selects the second row in `results`.

## Conclusion

This project builds a candidate-matching workflow around pretrained semantic embeddings. It transforms raw job titles into standardized titles and broad sectors, derives location and seniority features, and combines title similarity with sector agreement to rank candidates for a recruiter query. It then supports an interactive feedback loop in which starred candidates influence the ordering of the shortlist through title, sector, seniority, and location similarities.

The main outcome is a two-stage retrieval and re-ranking pipeline: semantic search produces an initial shortlist, and recruiter feedback adapts that shortlist to the recruiter's preferences.

One suggestion for improvement in the furture, instead of allowing individual starred candidates to disproportionately influence the ranking, exploring learning-to-rank methods that learn from multiple recruiter decisions and validated job-related outcomes should be taken into cosideration. This could make the ranking more consistent across searches and less prone to bias, provided the initial feedback is itself audited for bias
