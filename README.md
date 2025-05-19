# Boosted Transfer Learning for Passive Microwave Precipitation Retrievals

This repository represents the development of an algorithm, called the Boosted tRansfer-leArning for PMW precipitatioN Retrievals(B-RAINS). The algorithm relies on integrating the information content from Earth System Models (ESMs)
into the retrieval process and allows the fusion of multi-satellite observations across varying spatial and temporal resolutions through meta-model learning. The algorithm first detects the precipitation phase and then estimates its rate,
while conditioning the results to some atmospheric and surface-type variables.

<img src="images/Fig_01.png"  width="900" />

<p align="center"><em>Boosted tRansfer-leArning for precIpitatioN RetrievalS (B-RAINS) presents an ensemble learning architecture that stacks parallel XGBoost base learners and combines their inference through a meta-model. Step 1 detects the precipitation occurrence and phase, and Step 2 estimates the rain and snow rate, with the subscripts ``L'' and ``R'' denoting labels and rates of the data sets. The retrievals transfer the learning from ERA5 to satellite through incremental training of the base learners in both steps. After learning ERA5, the number of parallel trees, tree booster numbers, depths, and splitting nodes are frozen (FZ) for the top part of the decision trees.</em></p>


## Dataset
The dataset for training the networks and retrieving sample orbits is available [here](https://drive.google.com/drive/u/0/folders/1Njpyd_nWbNwxumzqJXwW5GhjkMftDVzW).
