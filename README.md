# ImageSegmentation (Mask2Former Traffic Image Segmentation)

**Status: Under Construction 🚧**

This repository contains an in-progress implementation of Mask2Former for traffic image segmentation. The main source of truth and the most complete workflow is currently in the Jupyter notebook:

- `notebooks/Project.ipynb`

## Project Structure

- `notebooks/Project.ipynb`  
  The primary, working implementation. Contains data loading, preprocessing, model training, and evaluation in a single notebook.
- `src/`  
  Intended for modular Python code (dataset, model, training, evaluation, utils, etc.). **These scripts are not yet fully functional or stable.**
- `prepare_data.py`, `train_model.py`, `evaluate_model.py`  
  Standalone scripts for data preparation, training, and evaluation. **These are still being developed and may not work as expected.**
- `data/`  
  Data directory. Place your raw data in `data/raw/data.pkl`.
- `models/`, `logs/`  
  Output directories for model checkpoints and logs.

## How to Use

1. **Start with the notebook:**
   - Open `notebooks/Project.ipynb` for the most up-to-date and working code.
2. **Python scripts:**
   - The scripts and modules outside the notebook are being refactored for modularity and maintainability. They are not guaranteed to work yet.

## Contributing

Contributions are **very welcome**! If you would like to help modularize the Mask2Former implementation, improve the codebase, or add features, please open an issue or submit a pull request.

- Suggestions for code structure, error handling, and best practices are appreciated.
- Please see the notebook for the current logic and data flow.


**Note:** This codebase is a work in progress. For the latest working pipeline, refer to the notebook. The modular Python scripts are under active development.
