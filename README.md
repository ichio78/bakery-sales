# bakery-sales

**Bakery Sales Prediction (Regression)**

 <p>&nbsp;</p>

**## Overview**

This project builds regression models to predict bakery sales based on weather conditions and event data.  
By combining climate factors (temperature, wind speed, cloudiness, weather code) and calendar events (Kieler Woche festival), the aim is to capture patterns influencing daily sales.

 <p>&nbsp;</p>

**## Dataset**

　- Source: Provided training dataset (not included here)

　- Size: ~9,300 records

　- Target: `Umsatz` (daily sales amount)

 <p>&nbsp;</p>

**## Preprocessing steps**

　- Merged multiple datasets (Weather, Events, Kieler Woche)

　- Treated outliers (removed unrealistic sales > 1000)

　- Handled missing values:

   - Predicted `Bewoelkung` and `Wettercode` using DecisionTreeClassifier
   - Completed imputation for `Temperatur` and `Windgeschwindigkeit`

　- Converted categorical variables (Warengruppe, Wettercode) via one-hot encoding

　- Feature engineering: event flags, date decomposition

 <p>&nbsp;</p>

**## Models & Methods**

- Linear Regression (baseline)

- XGBoost Regressor

- LightGBM Regressor

- CatBoost Regressor

- GridSearchCV for hyperparameter tuning

 <p>&nbsp;</p>

**## Results**

　- Best performance: **CatBoost / XGBoost with R² score evaluation**

　- External data integration and imputation improved stability

　- Highlighted the importance of event features (Kieler Woche) on sales

 <p>&nbsp;</p>

**## Technologies Used**

　- Python, Pandas, NumPy

　- scikit-learn (preprocessing, evaluation, GridSearchCV)

　- XGBoost, LightGBM, CatBoost

　- Matplotlib

　- Jupyter Notebook

 <p>&nbsp;</p>

**## Repository Structure**

```

bakery-sales/

├── bakery_sales.ipynb   # Main notebook

├── README.md            # Project description

└── data/                # Dataset (not included, see below)

```

<p>&nbsp;</p>

**## About Dataset**

The dataset is not included in this repository due to license restrictions. Please download it directly from Kaggle.

https://www.kaggle.com/competitions/bakery-sales-prediction-summer-2025/data

<p>&nbsp;</p>

**## Note**

This notebook was originally developed and executed in a local Jupyter environment. 

Due to the use of custom folder structures (e.g., `data/`, `notebook/`, `model/`), it may not run directly without modifications.  

The main purpose of this repository is to showcase the analysis process and results, rather than to provide a fully reproducible environment.
