[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Buddha-subedi/Microwave_Precipitation_Retrievals_from_B-RAINS/blob/main/B-RAINS_demo.ipynb)
# Boosted Transfer Learning for Passive Microwave Precipitation Retrievals

This repository represents the development of an algorithm, called the Boosted tRansfer-leArning for PMW precipitatioN Retrievals(B-RAINS). The algorithm relies on integrating the information content from Earth System Models (ESMs)
into the retrieval process and allows the fusion of multi-satellite observations across varying spatial and temporal resolutions through meta-model learning. The algorithm first detects the precipitation phase and then estimates its rate,
while conditioning the results to some atmospheric and surface-type variables.

<p align="center">
  <img src="images/Fig_01.png" width="700" />
</p>

<p align="center"><em>Boosted tRansfer-leArning for precIpitatioN RetrievalS (B-RAINS) presents an ensemble learning architecture that stacks parallel XGBoost base learners and combines their inference through a meta-model. Step 1 detects the precipitation occurrence and phase, and Step 2 estimates the rain and snow rate, with the subscripts ``L'' and ``R'' denoting labels and rates of the data sets. The retrievals transfer the learning from ERA5 to satellite through incremental training of the base learners in both steps. After learning ERA5, the number of parallel trees, tree booster numbers, depths, and splitting nodes are frozen (FZ) for the top part of the decision trees.</em></p>

<a name="4"></a> <br>
## Code

<a name="41"></a> <br>
###   Setup
To run this notebook on Google Colab, clone this repository
```python
!git clone https://github.com/Buddha-subedi/Microwave_Precipitation_Retrievals_from_B-RAINS.git
os.chdir("Microwave_Precipitation_Retrievals_from_B-RAINS")
```


```python
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from pathlib import Path
import xgboost as xgb
import os
import scipy.io
import pmw_utils
import importlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error
import scipy.stats as stats
from scipy.interpolate import interp1d
importlib.reload(pmw_utils)
from pmw_utils import plot_confusion_matrix, BRAINS_model
```
<a name="42"></a> <br>
 ### Load the Data
 
```python
paths_phase = {
    'cpr_train': 'data/df_cpr_phase_train.npz',
    'cpr_test':  'data/df_cpr_phase_test.npz',
    'dpr_train': 'data/df_dpr_phase_train.npz',
    'dpr_test':  'data/df_dpr_phase_test.npz',
    'era5_train': 'data/df_era5_phase_train.npz',
    'era5_test':  'data/df_era5_phase_test.npz'
}
data = {k: np.load(p) for k, p in paths_phase.items()}
dfs = {k: pd.DataFrame(dict(v)) for k, v in data.items()}
df_cpr_phase_train = dfs['cpr_train']
df_cpr_phase_test  = dfs['cpr_test']
df_dpr_phase_train = dfs['dpr_train']
df_dpr_phase_test  = dfs['dpr_test']
df_era5_phase_train = dfs['era5_train']
df_era5_phase_test  = dfs['era5_test']
```



<a name="43"></a> <br>
 ### Train the B-RAINS Model
B-RAINS Model has 4 base learners. The hyperparameters and snippet of code adopted for stage 1 and stage 2 for the ERA5-CPR phase detection is provided below

```python
#stage 1
params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'eval_metric': 'merror',
    'reg_alpha': 0.095,
    'reg_lambda': 7.843,
    'max_depth': 18,
    'num_parallel_tree': 3,
    'learning_rate': 0.330808,
    'gamma': 0.661776,
    'verbosity': 0
}

booster_era5 = xgb.train(
    params=params,
    dtrain=dtrain,
    evals=evals,
    num_boost_round=30,
    verbose_eval=True
)


#stage 2
classes = np.unique(df_70_train_cpr['Prcp flag'])                         
class_weights = {0: 1, 1: 1.167, 2: 1.766}
sample_weights_70 = df_70_train_cpr['Prcp flag'].map(lambda x: class_weights[classes.tolist().index(x)])
# Set parameters
params_1 = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'eval_metric': 'merror',
    'subsample': 0.5,
    'reg_alpha': 6.948,
    'reg_lambda': 5.0278,
    'max_depth': 18,
    'num_parallel_tree': 6,
    'learning_rate': 0.018,
    'gamma': 0.32,
    'verbosity': 0
}

booster_era5 = xgb.train(
    params=params_1,
    dtrain=dtrain_era5,
    evals=evals,
    num_boost_round=30,
    verbose_eval=True
)

# Train with the new data (booster here is the final model that is first trained on coarse
# resolution information from ERA5 and then fine-tuned on fine resolution satellite information)
params_2 = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'merror',
    'reg_alpha': 6.948,
    'reg_lambda': 5.0278,
    'max_depth': 15,
    'num_parallel_tree': 6,
    'learning_rate': 0.018,
    'gamma': 0.32,
    'verbosity': 0
}

booster_cpr = xgb.train(
    params_2,
    dtrain_cpr,
    num_boost_round=80,
    evals=evals,
    xgb_model=booster_era5,
    verbose_eval=True,
    feval=f1_eval_all_classes
)
```


<a name="44"></a> <br>
 ### Orbital Retrievals
```python
[phase, rain, snow, latitude, longitude] = BRAINS_model(path_orbit_004780, booster_cpr, booster_dpr, meta_model,
                     snow_rate_booster, rain_rate_booster,
                     df_cdf_rain, df_cdf_snow);
```
<p align="center">
  <img src="images/Fig_08.png" alt="Training for ERA5-CPR classifier base learner" width="700" />
</p>
<p align="center">
  <em>Brightness temperatures from GMI (a–d) and precipitation retrievals from B-RAINS (e), GPROF (f), DPR (g), and ERA5 (h) for orbit 044780 on January 15, 2022, capturing an extratropical cyclone over the North Atlantic Ocean and the Canadian provinces of Nova Scotia and New Brunswick.</em>
</p>



## Dataset
The dataset for training the networks and retrieving sample orbits is available here: (https://drive.google.com/drive/folders/1vtrSuf4CKG24wHklNTMDc1fL-drr9oRN?dmr=1&ec=wgc-drive-hero-goto).
