[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Buddha-subedi/Microwave_Precipitation_Retrievals_from_B-RAINS/blob/main/main_notebook_BRAINS.ipynb)
# Boosted Transfer Learning for Passive Microwave Precipitation Retrievals

This repository represents the development of an algorithm, called the Boosted tRansfer-leArning for PMW precipitatioN Retrievals(B-RAINS). The algorithm relies on integrating the information content from Earth System Models (ESMs)
into the retrieval process and allows the fusion of multi-satellite observations across varying spatial and temporal resolutions through meta-model learning. The algorithm first detects the precipitation phase and then estimates its rate,
while conditioning the results to some atmospheric and surface-type variables.

<div style="display: flex; justify-content: center;">
  <img src="images/Fig_01.png" width="600" />
</div>


<p align="center"><em>Boosted tRansfer-leArning for precIpitatioN RetrievalS (B-RAINS) presents an ensemble learning architecture that stacks parallel XGBoost base learners and combines their inference through a meta-model. Step 1 detects the precipitation occurrence and phase, and Step 2 estimates the rain and snow rate, with the subscripts ``L'' and ``R'' denoting labels and rates of the data sets. The retrievals transfer the learning from ERA5 to satellite through incremental training of the base learners in both steps. After learning ERA5, the number of parallel trees, tree booster numbers, depths, and splitting nodes are frozen (FZ) for the top part of the decision trees.</em></p>

<a name="4"></a> <br>
## Code

<a name="41"></a> <br>
###   Setup

```python
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
import scipy.io
import pmw_utils
import importlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from scipy.interpolate import interp1d
importlib.reload(pmw_utils)
from pmw_utils import plot_confusion_matrix, BRAINS_model
```
<a name="42"></a> <br>
 ### Load the Data
```python
# Load the training data

paths = {
    'cpr': Path.cwd() / 'data' / 'df_cpr_phase.npz',
    'dpr': Path.cwd() / 'data' / 'df_dpr_phase.npz',
    'era5': Path.cwd() / 'data' / 'df_era5_phase.npz'
}

data = {k: np.load(p) for k, p in paths.items()}
df_cpr_phase, df_dpr_phase, df_era5_phase = (pd.DataFrame({k: v[k] for k in v.files}) for v in data.values())

for name, df, sep in zip(
    ['ERA5 Samples for Classification', 'CPR Samples for Classification', 'DPR Samples for CLassification'],
    [df_era5_phase, df_cpr_phase, df_dpr_phase],
    ['##############################', '##############################', '']
):
    counts = df['Prcp flag'].value_counts().sort_index()
    print(f"**{name}**\nTotal samples for classification: {len(df)}")
    print(f"Clear: {counts.get(0,0)}, Rain: {counts.get(1,0)}, Snow: {counts.get(2,0)}")
    if sep: print(sep)
```


To load the B-RAINS model, use the following code
```python
model_dir = r'G:\Shared drives\SAFL Ebtehaj Group\Buddha Research\Research 1\model'

era5_dpr_base_learner = xgb.XGBClassifier()
era5_dpr_base_learner.load_model(os.path.join(model_dir, 'classifier_incremental_dpr_optuna_maximize_min_f1.json'))

era5_cpr_base_learner = xgb.XGBClassifier()
era5_cpr_base_learner.load_model(os.path.join(model_dir, 'classifier_incremental_cpr_optuna_maximize_min_f1.json'))

meta_model = xgb.XGBClassifier()
meta_model.load_model(os.path.join(model_dir, 'incremental_meta_both_point_zero_one.json'))

snow_rate_booster_tl = xgb.Booster()
snow_rate_booster_tl.load_model(os.path.join(model_dir, 'xgb_tl_snow_rate.json'))

rain_rate_booster_tl = xgb.Booster()
rain_rate_booster_tl.load_model(os.path.join(model_dir, 'xgb_tl_rain_rate.json'))
```

## Dataset
The dataset for training the networks and retrieving sample orbits is available [here](https://drive.google.com/drive/u/0/folders/1Njpyd_nWbNwxumzqJXwW5GhjkMftDVzW).
