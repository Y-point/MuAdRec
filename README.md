# MuadRec

MuadRec: Multi-modal Fusion with Adaptive Null Space Dimension for Sequential Recommendation

## Environment Requirements

### Python Version
Python 3.7+

### Dependencies
- torch>=1.8.0
- numpy>=1.19.0
- pandas>=1.2.0
- pyyaml>=5.4.0

### Install Dependencies

```bash
pip install torch torchvision
pip install numpy pandas pyyaml
```

## File Description

| File | Purpose |
|------|---------|
| `train.py` | Training script, the main entry point for model training |
| `muadrec.py` | Model definition file, contains MuAdRec architecture and core algorithms |
| `utils.py` | Utility functions, includes the evaluate function for model evaluation |
| `prepare_dataset_generic.py` | Data preprocessing script, converts raw data into training format |
| `config.yaml` | Configuration file, defines model hyperparameters and training parameters |
| `ASO/` | Dataset directory (needs to be created or provided) |
| `saved/` | Model saving directory (automatically created during training) |
| `log/` | Training log directory (automatically created during training) |

## Quick Start

### 1. Prepare Dataset

Put your preprocessed dataset into the `ASO/` directory. The dataset should include:
- `train_data.df` - Training data
- `val_data.df` - Validation data  
- `test_data.df` - Test data
- `data_statis.df` - Data statistics
- `items_pop.npy` - Item popularity
- `text_emb.pickle` - Text embeddings
- `image_emb.pickle` - Image embeddings

### 2. Configure Parameters

Modify the `config.yaml` file to adjust model hyperparameters.

### 3. Run Training

```bash
python train.py --config_files=config.yaml
```

