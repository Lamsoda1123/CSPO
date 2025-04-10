# CSPO: Cross-Market Synergistic Stock Price Movement Forecasting with Pseudo-volatility Optimization

## Abstract
This repository contains the implementation of CSPO, a novel framework for stock price movement forecasting that leverages cross-market synergy and pseudo-volatility optimization. CSPO demonstrates superior performance over existing methods by effectively addressing two key market characteristics: stock exogeneity and volatility heterogeneity.

## Key Features
- **Cross-market Synergy**: Leverages external futures knowledge to enrich stock embeddings with cross-market insights
- **Pseudo-volatility Optimization**: Models stock-specific forecasting confidence through pseudo-volatility, enabling dynamic adaptation of the optimization process
- **Transformer-based Architecture**: Implements an effective deep neural architecture for capturing intricate market patterns

## Technical Approach
CSPO combines:
1. A transformer-based model for processing stock and future features
2. Multi-head attention mechanisms for temporal pattern extraction
3. LoRA (Low-Rank Adaptation) layers for efficient fine-tuning
4. Pseudo-volatility modeling for adaptive optimization

## Results
Extensive experiments including industrial evaluation and public benchmarking demonstrate CSPO's effectiveness in:
- Improving prediction accuracy
- Enhancing model robustness
- Capturing complex market dynamics

This repository contains the complete codebase for the CSPO framework developed using transformer architectures. The model is designed to predict stock prices and volatility while addressing market complexity through innovative cross-market and volatility-aware approaches.


## Project Structure

### Core Model Files
- `model.py`: Contains the main neural network architectures:
  - `Model`: Base transformer model for stock prediction
  - `loraModel`: Variant using LoRA (Low-Rank Adaptation) for efficient fine-tuning
  - `Model2`: Ensemble model combining multiple base models

### Training Scripts
- `main100.py`, `main300.py`, `main500.py`: Training scripts with different configurations
- `main300_20.py`, `main300_21.py`, `main300_22.py`: Variants of main300 with different parameters
- `main300_inf.py`, `main500_inf.py`: Inference scripts
- `main300_mse.py`: Training with MSE loss

### Supporting Modules
- `decoderGT.py`: Transformer decoder implementation
- `loraModel.py`: LoRA transformer implementation
- `pytorch_transformer_ts.py`: Time series transformer utilities
- `utils.py`: Utility functions
- `dataset.py`: Data loading and preprocessing

### Experimental Directories
- `final_exp/`: Final experiment configurations
- `final_exp_ablation/`: Ablation study configurations

## Model Details

The models use transformer architectures to:
1. Process stock and future features
2. Extract temporal patterns
3. Predict stock prices and volatility

Key components include:
- Multi-head attention
- Layer normalization
- Positional encoding
- LoRA adaptation layers (in loraModel)