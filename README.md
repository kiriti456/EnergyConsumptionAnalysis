# Energy Consumption Analysis and Forecasting

## Description
This project focuses on analyzing and forecasting energy consumption trends through an interactive Streamlit web application. By leveraging statistical and machine learning techniques, it provides valuable insights into energy usage patterns by country and product. The dashboard includes functionalities such as clustering analysis, seasonal decomposition, and forecasting using ARIMA and LSTM.

---

## Steps to Run the Code

### 1. Install Required Libraries
Ensure Python 3.8+ is installed. Use the following command to install the necessary dependencies:

```bash
pip install pandas numpy matplotlib seaborn streamlit scikit-learn statsmodels tensorflow
```

### 2. Execute the main script:
```bash
streamlit run EnergyConsumption.py
```



## Features

### 1. Library Imports

- Libraries for data manipulation (pandas, numpy)
- Visualization libraries (matplotlib, seaborn)
- Machine learning libraries (scikit-learn)
- Time series analysis libraries (statsmodels for ARIMA, STL)
- Deep learning library (tensorflow.keras for LSTM)
- Streamlit for building an interactive web application

### 2. Data Loading and Preprocessing

- Loads energy consumption data from a CSV file (energy_data.csv)
- Converts Time column to datetime
- Converts Value column to numeric, handling errors
- Removes rows with missing values
- Imputes missing values in Value column with the mean of the column

### 3. Data Aggregation

- Groups data by Country, Time, and Product, summing energy consumption (Value)
- Creates a pivot table (time_series_df) for time-series analysis, with Time as the index and unique keys (Country - Product) as columns

### 4. Streamlit Application Setup

Configures the Streamlit app layout and navigation sidebar for different analysis options:
- Country-Wise Consumption
- Product-Wise Consumption
- Country & Product Visualization
- Clustering Analysis
- Seasonal Decomposition
- ARIMA Forecasting
- LSTM Forecasting

### 5. Visualization Functions

- Country-Wise Consumption: Visualizes total energy consumption per country using a bar chart
- Product-Wise Consumption: Similar visualization and analysis as above but for energy products
- Country & Product Visualization: Shows product-specific consumption within a selected country

### 6. Clustering Analysis

- Performs clustering (KMeans, DBSCAN, Agglomerative) on energy products based on consumption
- Uses silhouette score to evaluate and select the best clustering method
- Provides visualizations, numeric product labels, and cluster summaries

### 7. Seasonal Decomposition

- Decomposes time series data into trend, seasonality, and residuals using STL decomposition
- Displays components with plots for better understanding of periodic behavior and anomalies

### 8. ARIMA Forecasting

- Implements ARIMA modeling to predict future energy consumption:
  - Splits data into training and testing sets
  - Fits an ARIMA model and forecasts test data
  - Visualizes forecasts
  - Computes validation metrics (RMSE, MAPE)

### 9. LSTM Forecasting

- Implements deep learning (LSTM) for forecasting:
  - Scales and reshapes time-series data for LSTM input
  - Trains an LSTM model and makes predictions
  - Evaluates forecasts using RMSE and MAPE and visualizes results

### 10. Interactive Navigation

Each analysis/visualization is triggered by the selected option in the sidebar, providing flexibility for user interaction.

This dashboard provides a comprehensive analysis of energy consumption patterns, allowing users to explore trends, perform clustering, and make predictions using advanced statistical and machine learning techniques.
