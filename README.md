# Genome-Wide Association and Fine-Mapping Identify Major Genetic Contributors and Pathways in Alzheimer’s Disease

## Overview

This project delivers a comprehensive, reproducible pipeline for integrative genomic analysis of Alzheimer’s disease (AD). Leveraging large-scale, curated GWAS summary statistics, advanced statistical methodologies, and functional annotation, this resource enables the systematic identification and prioritization of genetic variants, genes, and pathways implicated in AD and related neurodegenerative traits.

---

## Table of Contents

- [Background](#background)
- [Objectives](#objectives)
- [Methods](#methods)
- [Key Results](#key-results)
- [Features](#features)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Data Availability](#data-availability)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)
- [Contributors](#contributors)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Background

Alzheimer’s disease is a progressive neurodegenerative disorder with a complex, polygenic genetic architecture. While genome-wide association studies (GWAS) have identified numerous risk loci, much of the heritability remains unexplained, and the functional mechanisms connecting genetic variation to disease remain poorly understood. This project addresses these gaps by integrating GWAS summary statistics with fine-mapping, gene-based analyses, and pathway enrichment to illuminate both established and novel contributors to AD risk.

---

## Objectives

- **Comprehensive Variant Discovery:** Systematically identify genetic variants and genes associated with AD and related phenotypes using robust, reproducible pipelines.
- **Fine-Mapping:** Prioritize likely causal variants via Bayesian fine-mapping approaches.
- **Functional Annotation:** Characterize the regulatory and coding landscape of significant variants.
- **Pathway Analysis:** Uncover biological processes and molecular pathways enriched for AD-associated variation.
- **Network Analysis:** Visualize and interpret the molecular interaction networks underlying AD genetics.
- **Resource Sharing:** Provide open, well-documented code and workflows for the broader research community.

---

## Methods

- **Data Source:**  
  - Alzheimer’s Disease Variant Portal (ADVP) curated variant table (GRCh38/hg38) ([Kuksa et al., 2022](https://doi.org/10.3233/JAD-215055))
- **Pipeline Components:**  
  - **Data Processing:** Automated cleaning, filtering, and integration of GWAS summary statistics.
  - **Statistical Analysis:**  
    - Association testing using [SciPy](https://scipy.org/) and [statsmodels](https://www.statsmodels.org/)
    - Multiple testing correction (Benjamini-Hochberg FDR)
    - P-value distribution and genomic inflation factor (λGC) calculation
    - Permutation-based significance assessment
  - **Fine-Mapping:** Bayesian fine-mapping to estimate posterior probabilities for causality ([Benner et al., 2016](https://doi.org/10.1093/bioinformatics/btw018))
  - **Gene and Pathway Enrichment:**  
    - Gene Ontology (GO) and pathway enrichment via [gseapy](https://github.com/zqfang/GSEApy)
    - Functional annotation (intronic, intergenic, missense, UTR, etc.)
  - **Network Analysis:** Construction and visualization of gene interaction networks with [NetworkX](https://networkx.org/) ([Hasan et al., 2023](https://doi.org/10.1007/978-1-0716-3327-4_35))
  - **Visualization:** Publication-quality plots with matplotlib, seaborn, and Plotly
  - **Reproducibility:** All code and parameters are version controlled and deposited in this repository

---

## Key Results

- **Variant Discovery:**  
  - 2,006 statistically significant associations (p < 1×10⁻⁵) identified from 6,346 variants across 946 genes and 844 phenotypes.
- **Fine-Mapping:**  
  - ~2,500 variants with high posterior probability of causality, with pronounced enrichment in established AD loci such as APOE and TOMM40, and novel regions (e.g., MIDEAS, ACP3, MMP3, ZCWPW1).
- **Chromosomal and Gene-Level Insights:**  
  - Chromosome 19 and 11 identified as hotspots for causal variants; pronounced heterogeneity in variant burden across the genome.
  - Most genes harbor few variants; a small subset (e.g., CLU, CR1, SORL1) show disproportionately high burdens.
- **Functional Annotation:**  
  - Majority of significant variants are intronic or regulatory, highlighting the importance of non-coding elements.
- **Pathway and Network Analysis:**  
  - Enrichment in immune signaling, amyloid processing, ERK/MAPK cascades, and cholesterol metabolism.
  - Network analysis reveals key hub genes and interconnected molecular modules.
- **Statistical Robustness:**  
  - Permutation testing confirms that observed associations are unlikely due to chance.
  - Genomic inflation factor indicates substantial polygenic signal but also warrants caution for potential confounding.
- **Phenotype Focus:**  
  - AD, late-onset AD, and CSF biomarkers dominate the genetic association landscape, demonstrating strong genotype-phenotype relationships.

---

## Features

- **Fully Automated Python Pipeline:**  
  Minimal user intervention required; all steps from data loading to visualization are scripted.
- **Modular Design:**  
  Each analysis component can be run independently or as part of an end-to-end workflow.
- **Comprehensive Visualization Suite:**  
  Includes Manhattan plots, QQ plots, heatmaps, bar charts, and network diagrams.
- **Extensive Documentation:**  
  Detailed comments, usage instructions, and supplementary materials.
- **Reproducibility:**  
  All code, environments, and parameters are version controlled.

---

## Usage

### Prerequisites

- Python 3.8+
- Recommended: Create a virtual environment

### Installation

```bash
git clone https://github.com/ejtettevi/ad_gwas_analysis.git
cd ad_gwas_analysis
pip install -r requirements.txt
```

### Running the Pipeline

```bash
python 01_main_analysis_script.py
```

### Output

- Results (tables and figures) saved in the results directory.
- Supplementary data are in supplementary file.

---

## Data Availability

- **Primary Dataset:**  
  Alzheimer’s Disease Variant Portal (ADVP) curated variant table (GRCh38/hg38)  
  [Kuksa et al., 2022](https://doi.org/10.3233/JAD-215055)

- **Code:**  
  All analysis code is included in this repository.
---

## Limitations & Future Work

- **Summary-Level Data:**  
  The current pipeline operates on summary-level statistics. Individual-level genotype and covariate data are not included, limiting advanced corrections for population stratification, batch effects, and cryptic relatedness.
- **Genomic Inflation:**  
  While genomic control is applied, high λGC suggests residual confounding may persist. Users are encouraged to interpret results with caution.
- **Extensibility:**  
  Future releases will support integration with individual-level data and more sophisticated modeling (e.g., mixed linear models, PCA-based corrections).
- **Community Contributions:**  
  Pull requests and issues for new features, bug fixes, or additional analyses are welcome.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for terms.

---

## Acknowledgements

We gratefully acknowledge the Alzheimer’s Disease Variant Portal (ADVP) team, all data contributors, and the open-source software community whose tools enabled this research.  
Special thanks to the funding agencies and institutional partners supporting this work.

---

## Contact

**Corresponding Author:**  
Edward J. Tettevi  
ejtettevi@gmail.com

For questions, collaboration, please contact the corresponding author.
