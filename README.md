# Analysis of the Psychological Impact and Relation of Organizational Networks on Employee Collaborative Processes
—Exploring Group Dynamics through LLM Multi-Agent Simulation—

This repository contains the simulation source code and analysis scripts used to generate the results presented in our ECML 2026 paper.

## Getting Started

### Environment Setup
Ensure you have a recent Python environment set up. Since the project uses API keys to power the LLM Multi-Agent Simulation, you must create a `.env` file in the root directory and define your keys:
```env
OPENAI_API_KEY="your-api-key-here"
GEMINI_API_KEY="your-api-key-here"
```

### Execution Workflow

To reproduce the simulation and subsequent analyses, execute the scripts in the following order:

#### 1. Running the Simulation (`simulations/`)
1. **`run_full_simulation.py`**: Execute this script first to run and complete the LLM agent conversation simulations.
2. **Data Generation**: 
   - Run `create_amos_dataset.py`, followed by `analyze.py` and `pre_data_shape.py`. 
   - This process aggregates the outputs and generates formatted data inside the `results/05_...` folder.
3. **Network Matrix Generation**: 
   - Run `analyze_network_silo.py`.
   - This step creates conversation-based adjacency matrices, saved in the `results/07_...` folder.

#### 2. Running the Analysis (`analysis/`)
Once the simulation steps up to `07_` are complete, proceed to evaluate the results using the "LLM-as-a-judge" methodology:
4. **`run_full_analysis.py`**: Execution of this script is required to run the primary judgments.
5. **Data Organization**: 
   - Run the data organization script located at `analysis/雑務/organize_08_data.py` (雑務: Miscellaneous). 
   - This formats and organizes the `results/08_...` folder.
6. **Main Analysis Execution**: 
   - Navigate to `analysis/本分析/scripts/` (本分析: Main Analysis).
   - Execute the scripts inside this folder in ascending numerical order (`01_...`, `02_...`, etc.) to generate the final analyzed folders and visual results.

---

## Important Notes & Disclaimers

### Missing `results/06_...` and Unused Data
You may notice that the `results/06_` folder is missing from the directory sequence, and the `04_` and `05_` datasets are barely used outside of general conversation aggregation. During the simulation design, the self-disclosure phase — in which agents were prompted to express their internal concerns during a questionnaire phase based on their current thought state — did not function effectively enough to yield meaningful results. Consequently, these specific evaluations were not utilized in the final paper. To keep the repository clean and straightforward, the generated `06_...` formatted dataset and its accompanying generation script have been intentionally deleted.

### Customer Persona Data Source
Although not actively utilized in the final research paper's analysis, the simulation workflow internally employs a customer persona model to evaluate generated product proposals. The demographic data (`demographics.xlsx`) used for this evaluation was formatted from `h20026.csv` using the `mk_customer_persona.py` script. The original public statistical data (`h20026.csv`) was obtained from the [e-Stat portal (Japanese Government Statistics)](https://www.e-stat.go.jp/stat-search/files?page=1&layout=datalist&toukei=00450061&tstat=000001206248&cycle=7&tclass1=000001206253&stat_infid=000040076425&tclass2val=0).
(https://www.e-stat.go.jp/stat-search/files?page=1&layout=datalist&toukei=00450061&tstat=000001206248&cycle=7&tclass1=000001206253&stat_infid=000040076425&tclass2val=0)


## Acknowledgements & Third-Party Licenses

This project includes and builds upon the local `semantic_similarity_rating` package for calculating response probability distributions based on LLM embeddings.

- **Original Author(s):** Maier, B. F., Aslak, U., Fiaschi, L., Pappas, K., Wiecki, T. (2025). Measuring Synthetic Consumer Purchase Intent Using Embeddings-Similarity Ratings.
- **Original Source:** [pymc-labs/semantic-similarity-rating](https://github.com/pymc-labs/semantic-similarity-rating)
(https://github.com/pymc-labs/semantic-similarity-rating)

The original package is distributed under the **Apache License 2.0**. A copy of this license can be found in `pymc-labs_LICENSE`, and the original documentation is available in `pymc-labs_README.md`. Any modifications made to the `semantic_similarity_rating` files within this repository are the work of the current repository owner.

## License

With the exception of the third-party `semantic_similarity_rating` package (which remains under its Apache License 2.0 terms), the code in this repository is provided under the **MIT License**. See the `LICENSE` file for details.
