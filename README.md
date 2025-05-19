# Boosted Transfer Learning for Passive Microwave Precipitation Retrievals

This repository represents the development of an algorithm, called the Boosted tRansfer-leArning for PMW precipitatioN Retrievals(B-RAINS). The algorithm relies on integrating the information content from Earth System Models (ESMs)
into the retrieval process and allows the fusion of multi-satellite observations across varying spatial and temporal resolutions through meta-model learning. The algorithm first detects the precipitation phase and then estimates its rate,
while conditioning the results to some atmospheric and surface-type variables.

<img src="images/Fig_01.png"  width="800" />

<p align="center"><em>Brightness temperatures from GMI (a--d) and precipitation retrievals from B-RAINS (e), GPROF (f), DPR (g), and ERA5 (h) for orbit 044780 on January 15, 2022, capturing an extratropical cyclone over the North Atlantic Ocean and the Canadian provinces of Nova Scotia and New Brunswick.</em></p>


## Dataset
The dataset for training the networks and retrieving sample orbits is available [here](https://drive.google.com/drive/u/0/folders/1Njpyd_nWbNwxumzqJXwW5GhjkMftDVzW).
