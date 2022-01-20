# Data Sources

PdM1 - https://www.kaggle.com/c/machine-failure-prediction/data?select=train.csv

PdM2 - https://www.kaggle.com/binaicrai/machine-failure-data

# Preprocessing
```
from gym_pm.utils import create_data, update_boundaries

# If the pickle files do not exist, please run the below code.

data = 'PdM2'
create_data(data, split='Train', save=True)
create_data(data, split='Test', save=True)
update_boundaries(data, save=True)
```
