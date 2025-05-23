[![Preprint](https://img.shields.io/badge/project-paper-orange)](https://doi.org/10.48550/arxiv.2505.06564)
[![Cite This Work](https://img.shields.io/badge/citation-ready-green)](#Citation)
[![Python](https://img.shields.io/badge/python-3.11+-pink.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-CC0_1.0-blue.svg)](LICENSE)

# Malaysian Election Corpus (MECo): Federal & State-level Election Results since 1955

Empirical research and public knowledge on Malaysia’s elections have long been constrained by a lack of high-quality open data, particularly in the absence of a Freedom of Information framework. We introduce the Malaysian Election Corpus (MECo; ElectionData.MY), an open-access panel database covering all federal and state general elections from 1955 to the present, as well as by-elections from 2008 onward. MECo includes candidate- and constituency-level results for nearly 10,000 contests across seven decades, standardised with unique identifiers for candidates, parties, and constituencies. The database also provides summary statistics on electorate size, voter turnout, rejected votes, and unreturned ballots. This is the most well-curated publicly available data on Malaysian elections, and will unlock new opportunities for research, data journalism, and civic engagement.

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
- Standardisation of candidate, party, and constituency names with unique identifiers
- Lookup tables for reproducible research
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
@misc{thevananthan2025malaysianelectioncorpusmeco,
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

Contributions are not welcome, in order to maintain appropriate provenance for academic credit. However, you are free to open an issue!
