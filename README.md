[![Preprint](https://img.shields.io/badge/Preprint-arXiv-orange)](https://doi.org/10.48550/arxiv.2505.06564)
[![Code DOI](https://img.shields.io/badge/Code%20Archive-Zenodo-blue)](https://doi.org/10.5281/zenodo.17694675)
[![Cite This Work](https://img.shields.io/badge/Cite%20This%20Work-BibTeX-green)](#Citation)
[![Python](https://img.shields.io/badge/Python-3.12+-yellow)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-CC0_1.0-lightgrey)](LICENSE)

# 🇲🇾 Malaysian Election Corpus (MECo): Federal & State-level Election Results since 1955

Empirical research and public knowledge on Malaysia's elections have long been constrained by a lack of high-quality open data, particularly in the absence of a Freedom of Information framework. This paper introduces the Malaysian Election Corpus (MECo), an open-access panel database covering all federal and state general elections since 1955, as well as by-elections since 2008. MECo includes candidate- and constituency-level data for 9,704 electoral contests across seven decades, standardised with unique identifiers for candidates, parties, and coalitions. The database also provides summary statistics (electorate size, voter turnout, majority size, rejected votes, unreturned ballots) for each contest, and key demographic data (age, gender, ethnicity) for candidates. This is the most well-curated publicly available data on Malaysian elections, and will unlock new opportunities for research, data journalism, and civic engagement.

This repository contains the full data processing and analysis pipeline underlying the research paper.

## Repository Structure

| Directory/File                  | Description                                                              |
|---------------------------------|--------------------------------------------------------------------------|
| `src-data/`                     | Raw source data (tabular CSV + Parquet)                                  |
| `logs/`                         | Logs and correction files                                               |
| `tex/`                          | LaTeX files for manuscript generation                                    |
| `compile.py`                    | Main script to compile the cleaned and standardised dataset              |
| `dataviz.py`                    | Script to generate summary visualisations                               |
| `dashboards.py`                 | Script for generating harmonised panels for visualisation/dashboarding   |
| `fuzzymatch.py`                 | Utility script for fuzzy matching candidate names                        |
| `helper.py`                     | Helper functions used across scripts                                     |
| `requirements.txt`              | Python dependencies                                                     |
| `README.md`                     | This file                                                                |
| `LICENSE`                       | License file (CC0)                                                       |

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

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the main analysis script:
```bash
python compile.py
python dashboards.py
```

5. Generate visualizations:
```bash
python dataviz.py
```

## Citation

If you use this work, please cite it as:

> Thevesh Thevananthan, "The Malaysian Election Corpus (MECo): Federal and State-Level Election Results from 1955 to 2025", 2025.

``` BibTeX
@misc{thevesh2025mecoresults,
      title={The Malaysian Election Corpus (MECo): Federal and State-Level Election Results from 1955 to 2025}, 
      author={Thevesh Thevananthan},
      year={2025},
      eprint={2505.06564},
      archivePrefix={arXiv},
      primaryClass={cs.CY},
      url={https://arxiv.org/abs/2505.06564}, 
}
```

## Questions / Suggestions

If you want to improve the quality of the underlying data, please fork this repo, then make a pull request for Thevesh's review. However, do consider opening an issue to discuss your desired changes first!
