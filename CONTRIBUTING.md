# Contributing to Bayesian LLM Routing Research

Thank you for your interest in contributing to our research on Bayesian Framework for Efficient LLM Routing! This document provides guidelines for contributing to this academic research project.

## 🎯 Types of Contributions

### 1. **Research Extensions**

- Novel algorithmic improvements
- Additional baseline implementations
- New evaluation metrics or datasets
- Theoretical analysis and proofs

### 2. **Experimental Enhancements**

- Bug fixes in experimental code
- Performance optimizations
- Additional visualizations
- Reproducibility improvements

### 3. **Documentation**

- Code documentation improvements
- Tutorial additions
- Clarification of mathematical formulations
- Translation to other languages

## 🔬 Research Contribution Guidelines

### Before Contributing

1. **Read the Paper**: Understand the theoretical foundation and methodology
2. **Run Experiments**: Reproduce the baseline results on your system
3. **Check Issues**: Review existing issues and discussions
4. **Theoretical Discussions**: Join our research Discord community at https://discord.gg/yuzNhYBe

### Research Standards

- **Theoretical Rigor**: New algorithms should include theoretical justification
- **Experimental Validation**: Empirical claims must be supported by statistically significant evidence
- **Baseline Comparisons**: New methods should be compared against existing baselines
- **Reproducibility**: All experiments must be reproducible with provided code

## 🚀 Getting Started

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/Kr-TeamWise bayesian-token-prediction-llm-routing
cd bayesian-llm-routing

# Install dependencies in development mode
uv install --dev
# or: pip install -e ".[dev]"

# Run tests to verify setup
python -m pytest tests/
```

### Running Experiments

```bash
# Full experimental pipeline (may take 30+ minutes)
python main_experiment.py

# Individual components for faster iteration
python -c "from experiments.bayesian_predictor import BayesianTokenPredictor; ..."
```

## 📝 Pull Request Process

### 1. **Create Feature Branch**

```bash
git checkout -b feature/your-research-contribution
```

### 2. **Development Checklist**

- [ ] Code follows PEP 8 style guidelines
- [ ] All functions have comprehensive docstrings
- [ ] Mathematical formulations are clearly documented
- [ ] Experiments include statistical significance testing
- [ ] Results are reproducible with fixed random seeds

### 3. **Commit Message Format**

```
type(scope): Brief description

Detailed explanation of changes, including:
- Theoretical motivation
- Experimental validation
- Performance implications

Fixes #issue-number
```

Types: `feat`, `fix`, `docs`, `experiment`, `theory`, `refactor`

### 4. **Testing Requirements**

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific experimental validation
python tests/test_reproducibility.py

# Check code style
flake8 experiments/ main_experiment.py
```

## 🧪 Experimental Standards

### New Algorithm Implementations

1. **Theoretical Foundation**: Include mathematical derivation in docstrings
2. **Complexity Analysis**: Document time/space complexity
3. **Hyperparameter Sensitivity**: Test robustness to parameter choices
4. **Statistical Validation**: Use appropriate significance tests

### Baseline Comparisons

1. **Fair Implementation**: Use published hyperparameters when available
2. **Multiple Metrics**: Evaluate across cost, quality, and speed dimensions
3. **Error Analysis**: Include confidence intervals and statistical tests
4. **Ablation Studies**: Isolate individual component contributions

### Data and Evaluation

1. **Temporal Splits**: Maintain chronological order to avoid leakage
2. **Cross-Validation**: Use appropriate CV strategy for time series data
3. **Multiple Datasets**: Test generalization beyond LM Arena when possible
4. **Negative Results**: Document failed approaches for scientific value

## 📊 Code Quality Standards

### Documentation Requirements

```python
def new_routing_algorithm(query_features, models, **kwargs):
    """
    Novel routing algorithm with theoretical justification.

    This implements the XYZ algorithm from [Reference], which optimizes
    the utility function U(x) = E[Quality] - λ*E[Cost] + exploration_bonus.

    Mathematical Foundation:
    =======================
    The algorithm solves the optimization problem:
        argmax_i E[U_i(x)] subject to budget constraints

    Where U_i(x) is defined as... [detailed mathematical explanation]

    Parameters:
    -----------
    query_features : np.ndarray, shape (n_features,)
        Feature vector for the input query
    models : List[str]
        Available model identifiers
    **kwargs : dict
        Additional algorithm-specific parameters

    Returns:
    --------
    selected_model : str
        Optimal model for the given query
    confidence : float
        Selection confidence score [0, 1]

    References:
    -----------
    [1] Author et al. "Paper Title", Conference YYYY

    Examples:
    ---------
    >>> features = np.array([0.5, 0.3, 1.0, 0.0])
    >>> models = ['gpt-4', 'claude-2']
    >>> selected, conf = new_routing_algorithm(features, models)
    """
```

### Performance Requirements

- **Efficiency**: Routing decisions should complete in <100ms
- **Memory**: Maintain reasonable memory footprint for production use
- **Scalability**: Test with 10+ models and various query loads

## 🏆 Recognition

Contributors to significant research extensions will be acknowledged in:

- GitHub contributors list
- Paper acknowledgments (for substantial contributions)
- Conference presentation credits
- Follow-up research collaborations

## 📞 Getting Help

- **Research Questions**: Open a GitHub Discussion
- **Implementation Issues**: Create a GitHub Issue with reproducible example
- **Theoretical Discussions**: Join our research Slack/Discord [https://discord.gg/yuzNhYBe]
- **Collaboration Opportunities**: Email [yush7881@korea.ac.kr]

## 📜 Code of Conduct

This project follows standard academic research ethics:

- **Honest Reporting**: No cherry-picking results or misleading claims
- **Proper Attribution**: Cite all related work and acknowledge contributions
- **Open Science**: Share negative results and failed approaches
- **Respectful Collaboration**: Maintain professional and inclusive environment

## 🎓 Academic Guidelines

### For Student Contributors

- Discuss contributions with your advisor
- Ensure contributions align with your research goals
- Consider co-authorship opportunities for significant contributions

### For Postdocs/Faculty

- Propose collaborative extensions via GitHub Issues
- Share complementary datasets or evaluation frameworks
- Consider joint paper submissions for major extensions

---

**Thank you for helping advance the science of efficient LLM routing! 🚀**
