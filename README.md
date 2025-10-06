# ML Ops Assignment 1 - House Price Prediction

Repository structure:
- misc.py            : utility functions (data loading, preprocessing, training/evaluation helpers)
- train.py           : trains DecisionTreeRegressor (for branch `dtree`)
- train2.py          : trains KernelRidge (for branch `kernelridge`)
- requirements.txt
- .github/workflows/ci.yml

## Setup (recommended conda)
1. Create conda env:
   ```
   conda create -n mlops1 python=3.9 -y
   conda activate mlops1
   pip install -r requirements.txt
   ```

## Run locally
- Decision Tree:
  ```
  python train.py --n_runs 5 --test_size 0.2
  ```
- Kernel Ridge:
  ```
  python train2.py --n_runs 5 --test_size 0.2
  ```
