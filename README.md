# Relation-Extraction
Deep Graph Analytics Improve Biomedical Relation Extraction
 
### Datasets
Acquiring some datasets requires registration in the original data provider site. Therefore the test data provided here is only to enable reproduction of the results as reported in the paper. If interested in training new models, please download the datasets from original resources and use get_data_X in each X dataset folder for preprocessing the data. (X = {AGAC, I2B2, ChemProt, DDI})


### Replication
- Create a virtual environment
```$ conda create -n REenv python=3.6```
- Activate the environment
```$ conda activate REenv```
- Install the libraries as listed in the requirements.txt
```(REenv)$ pip install -r requirements.txt```
- Replicate the results reported for each dataset by running the corresponding X_CNN_GRU.py. (X = {AGAC, I2B2, ChemProt, DDI}) 
```(REenv).../deep_RE/X$ python X_CNN_GRU.py```