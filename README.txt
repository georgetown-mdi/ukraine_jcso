
##
This repository contains code accompanying the article "The Digital Trail of Ukraine’s 2022 Refugee Exodus". 

##
Regretfully, we are not able to share the migration data. But we would still like to share the code implementing our methodology. Consequently, we generate artificial data in the file python/synthetic_data.py which is stored in the file proc_data/syn_y.csv. 

Our preprocessed organic data is avialable as proc_data/ukraine_X.csv. 

## Installation
This analysis was conducted using Python 3.7.6. 
The requirements.txt file contains the modules used for this project, and can be installed via 
    pip install -r requirements.txt
in this directory.

Some scripts assume that the root directory of this repository is the working directory.

Our main analysis occurs in the following files:
 - The file international_eda.py creates Figure 2, which gives the correlation between our prediction data. 
 - The file viz_fits.py creates Figure 5, which shows the fit provided by each data source.
 - The file lag_or_lead.py creates Figure 6, which shows lag and window estimated for each data source.
 - The file international_pred.py creates Figure 7, which performs a predictive evaluation.


