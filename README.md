[![Published](https://img.shields.io/badge/Published-SciData-beige)](https://doi.org/10.1038/s41597-025-06502-7)
[![Code DOI](https://img.shields.io/badge/Code%20Archive-Zenodo-blue)](https://doi.org/10.5281/zenodo.17694675)
[![Data Archive](https://img.shields.io/badge/Data%20Archive-Harvard%20Dataverse-green)](https://doi.org/10.7910/DVN/O4CRXK)
[![Data Mirror](https://img.shields.io/badge/Data%20Mirror-HuggingFace%20🤗-green)](https://doi.org/10.57967/hf/7164)
[![Cite This](https://img.shields.io/badge/Cite%20This-BibTeX-lightgrey)](#Citation)

# 🇲🇾 Malaysian Election Corpus (MECo): Federal & State-level Election Results since 1955

This repository contains the full processing and validation pipeline underlying the research paper, which has been accepted for publication in *Scientific Data*.

**Latest data update**: 15th Sabah state election (2025) + all by-elections before 2008 

**Summary**: Empirical research and public knowledge on Malaysia's elections have long been constrained by a lack of high-quality open data, particularly in the absence of a Freedom of Information framework. This paper introduces the first component of the Malaysian Election Corpus (MECo), an open-access panel database covering all federal, state, and by-elections since 1955. MECo includes candidate- and constituency-level data for all 9,998 electoral contests from 1955 to 2025, standardised with unique identifiers for candidates, parties, and coalitions. The database also provides summary statistics (electorate size, voter turnout, majority size, rejected ballots, unreturned ballots) for each contest, and key demographic data (age, gender, ethnicity) for candidates. This is the most well-curated publicly available data on Malaysian elections, and will unlock new opportunities for research, data journalism, and civic engagement.

## Repository Structure

| Directory/File                  | Description                                                              |
|---------------------------------|--------------------------------------------------------------------------|
| `data/`                         | Final publication-grade datasets (tabular CSV + Parquet)                 |
| `dashboards/`                   | Processed data supporting the interactive website                        |
| `logs/`                         | Record of corrections made to official source                            |
| `tex/`                          | LaTeX files for manuscript generation                                    |
| `compile.py`                    | Compile and validate the clean + standardised data                       |
| `dataviz.py`                    | Generate summary visualisations                                          |
| `dashboards.py`                 | Generate harmonised panels for visualisation/dashboarding                |
| `gen_candidate_uid.py`          | Generate unique base-32 Crockford strings from running numbers           |
| `helper.py`                     | Helper functions used across scripts                                     |

## Features

- Compilation and validation of Malaysian election results (1955–present)
- Standardisation of candidate, parties, and coalitions with unique identifiers
- Lookup tables for extensibility
- Visualisation scripts for quick exploration
- LaTeX manuscript source files

## Installation and Usage

1. Clone the repository:
```bash
git clone git@github.com:thevesh/paper-meco-results.git
cd paper-meco-results
```

2. This project uses `uv` to manage Python dependencies.
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

3. Compile data and dashboards:
```bash
python3 gen_candidate_uid.py
python3 compile.py
python3 dashboards.py
```

## Citation

If you use this work, please cite it as:

> Thevananthan, T. The Malaysian Election Corpus (MECo): Federal and State-Level Election Results from 1955 to 2025. Sci Data (2025). https://doi.org/10.1038/s41597-025-06502-7

``` BibTeX

@article{thevesh2025meco1,
	author = {Thevananthan, Thevesh},
	year = {2025},
	doi = {10.1038/s41597-025-06502-7},
	isbn = {2052-4463},
	journal = {Scientific Data},
	title = {The Malaysian Election Corpus (MECo): Federal and State-Level Election Results from 1955 to 2025},
	url = {https://doi.org/10.1038/s41597-025-06502-7}
}
```

## Questions / Suggestions

If you want to improve the quality of the underlying data, please fork this repo, then make a pull request for review. However, do consider opening an issue to discuss your desired changes first!
