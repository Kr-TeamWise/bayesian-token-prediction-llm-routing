# LLM Routing with Bayesian Framework

This repository contains the implementation and experimental code for the research paper "**Bayesian Framework for Efficient LLM Routing with Thompson Sampling**".

## Overview

This project implements a novel Bayesian framework for routing queries to Large Language Models (LLMs) using Thompson Sampling. The framework addresses the cold start problem in LLM routing by leveraging model family relationships and uncertainty quantification.

## Key Features

- **Bayesian Token Prediction**: Predict token usage with uncertainty quantification
- **Thompson Sampling Router**: Balance exploration and exploitation for optimal model selection
- **Cold Start Handling**: Leverage model family priors for new models
- **Comprehensive Evaluation**: Benchmarking against multiple baseline methods

## Project Structure

```
├── main_experiment.py          # Main experimental pipeline
├── main.py                     # Simple entry point
├── experiments/
│   ├── bayesian_predictor.py   # Bayesian token prediction model
│   ├── data_loader.py          # LM Arena data loader and preprocessing
│   └── thompson_router.py      # Thompson Sampling routing implementation
├── data/                       # Data directory (LM Arena dataset)
├── results/                    # Experimental results and visualizations
├── pyproject.toml             # Project dependencies
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.12+
- UV package manager (recommended) or pip

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd research
```

2. Install dependencies:

```bash
# Using UV (recommended)
uv install

# Or using pip
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the main experiment:

```bash
python main_experiment.py
```

This will execute the complete experimental pipeline including:

1. Data loading and preprocessing
2. Model family correlation analysis
3. Bayesian token prediction model training
4. Thompson Sampling router evaluation
5. Cold start experiments
6. Baseline comparisons
7. Results visualization and reporting

### Individual Components

#### 1. Data Loading

```python
from experiments.data_loader import LMArenaDataLoader

loader = LMArenaDataLoader(cache_dir="./data")
raw_data = loader.download_lmarena_data()
processed_data = loader.preprocess_data()
```

#### 2. Bayesian Token Prediction

```python
from experiments.bayesian_predictor import BayesianTokenPredictor

model_families = {
    'openai': ['gpt-4-1106-preview', 'gpt-4-0613'],
    'anthropic': ['claude-2.1', 'claude-2.0'],
    # ... more families
}

predictor = BayesianTokenPredictor(model_families)
training_results = predictor.fit(train_data)
```

#### 3. Thompson Sampling Router

```python
from experiments.thompson_router import ThompsonSamplingRouter

router = ThompsonSamplingRouter(
    models=available_models,
    token_predictor=predictor,
    cost_weight=0.3
)

selected_model = router.select_model(
    query_features,
    predicted_tokens,
    model_uncertainties
)
```

## Experimental Results

The framework demonstrates significant improvements over baseline methods:

- **Performance**: +15-25% improvement over random routing
- **Cost Efficiency**: Up to 30% cost reduction while maintaining quality
- **Cold Start**: Rapid convergence for new models (50-100 queries)
- **Uncertainty Calibration**: Well-calibrated uncertainty estimates

Key metrics tracked:

- Routing accuracy
- Cost efficiency
- Convergence time
- Model utilization distribution
- Uncertainty calibration

## Model Families Supported

The framework supports the following LLM families:

- **OpenAI**: GPT-4, GPT-3.5 variants
- **Anthropic**: Claude-2, Claude Instant
- **Google**: Gemini Pro, PaLM-2
- **Meta**: LLaMA-2 (various sizes)
- **Mistral**: Mixtral, Mistral variants

## Baseline Methods

The framework is compared against:

1. **Random Routing**: Uniform random selection
2. **Always Premium**: Always select highest-quality model
3. **Simple Threshold**: Rule-based complexity thresholding
4. **Cost-Only**: Always select cheapest model
5. **Simple Utility**: Basic utility-based selection

## Data

The experiments use the [LM Arena Human Preference Dataset](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k) containing 55k human preference comparisons between LLMs.

If the dataset download fails, the system automatically generates realistic simulation data for testing.

## Reproducibility

All experiments use fixed random seeds for reproducibility:

- Data splits: `random_state=42`
- Model training: `random_state=42`
- Sampling procedures: `np.random.seed(42)`

## Dependencies

Key dependencies include:

- `numpy>=2.2.6`: Numerical computations
- `pandas>=2.3.0`: Data manipulation
- `scikit-learn>=1.7.0`: Machine learning models
- `scipy>=1.15.3`: Statistical functions
- `datasets>=3.6.0`: Hugging Face dataset loading
- `matplotlib>=3.10.3`: Visualization
- `seaborn>=0.13.2`: Statistical plotting

See `pyproject.toml` for complete dependency list.

## Contributing

This is research code accompanying an academic paper. For questions or issues, please open a GitHub issue.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{your_paper_2024,
  title={Bayesian Framework for Efficient LLM Routing with Thompson Sampling},
  author={Your Name and Collaborators},
  journal={Conference/Journal Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Results Directory

The `results/` directory contains:

- Experimental plots and visualizations
- Performance comparison tables
- Detailed analysis reports
- Model confidence metrics
- Economic impact analysis

All results are automatically generated and saved in both PNG and markdown formats for easy reporting and analysis.
