import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings

warnings.filterwarnings("ignore")

# Load dataset
file_path = 'energy_data.csv'  # Replace with your dataset path
df = pd.read_csv(file_path)

# Preprocessing
def preprocess_data(df):
    df['Time'] = pd.to_datetime(df['Time'], format='%b-%y')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df.dropna(subset=['Value'], inplace=True)
    return df

df = preprocess_data(df)

# Handle missing data (imputation)
def impute_missing_values(df):
    df['Value'] = df['Value'].fillna(df['Value'].mean())
    return df

df = impute_missing_values(df)

# Aggregated Data by Country, Product, and Time
aggregated_df = df.groupby(['Country', 'Time', 'Product']).agg({'Value': 'sum'}).reset_index()
aggregated_df['UniqueKey'] = aggregated_df['Country'] + " - " + aggregated_df['Product']
time_series_df = aggregated_df.pivot(index='Time', columns='UniqueKey', values='Value').fillna(0)

# Streamlit App
st.set_page_config(page_title="Energy Consumption Analysis", layout="wide")

# Sidebar for Navigation
st.sidebar.title("Navigation")
analysis_option = st.sidebar.selectbox("Select Analysis Type", [
    "Country-Wise Consumption",
    "Product-Wise Consumption",
    "Country & Product Visualization",
    "Clustering Analysis",
    "Seasonal Decomposition",
    "ARIMA Forecasting",
    "LSTM Forecasting"
])

# Country Selection for Country-Specific Visualizations
countries = df['Country'].unique()
selected_country = st.sidebar.selectbox("Select Country", countries)

# Filter time series data for the selected country
def filter_country_time_series(df, country):
    country_columns = [col for col in df.columns if col.startswith(country)]
    return df[country_columns]

# Functions for Visualizations and Descriptions
def plot_country_wise_consumption(df):
    st.write("### Country-Wise Energy Consumption")
    st.write("This visualization displays the total energy consumption for each country, enabling a comparison across all countries.")
    country_totals = df.groupby('Country')['Value'].sum().sort_values(ascending=False)
    numeric_labels = range(1, len(country_totals) + 1)
    
    # Plot bar chart with numeric x-axis
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.bar(numeric_labels, country_totals.values, color='skyblue')
    ax.set_title("Overall Energy Consumption by Country")
    ax.set_xlabel("Country (Numeric Labels)")
    ax.set_ylabel("Total Energy Consumption (GWh)")
    ax.set_xticks(numeric_labels)
    ax.set_xticklabels(numeric_labels)
    st.pyplot(fig)
    
    # Display mapping table
    st.write("### Numeric Label to Country Mapping")
    mapping_df = pd.DataFrame({'Label': numeric_labels, 'Country': country_totals.index})
    st.write(mapping_df)
    highest_country = country_totals.idxmax()
    highest_value = country_totals.max()
    st.write(f"### Insights")
    st.write(f"- {highest_country} has the highest energy consumption with **{highest_value} GWh**.")
    st.write(f"- This analysis helps identify countries with the largest energy demands and potential areas for optimization.")

def plot_product_wise_consumption(df):
    st.write("### Product-Wise Energy Consumption")
    st.write("This chart provides a breakdown of energy consumption by product, highlighting which energy sources are used the most.")
    product_totals = df.groupby('Product')['Value'].sum().sort_values(ascending=False)
    numeric_labels = range(1, len(product_totals) + 1)
    
    # Plot bar chart with numeric x-axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(numeric_labels, product_totals.values, color='teal')
    ax.set_title("Overall Energy Consumption by Product")
    ax.set_xlabel("Product (Numeric Labels)")
    ax.set_ylabel("Total Energy Consumption (GWh)")
    ax.set_xticks(numeric_labels)
    ax.set_xticklabels(numeric_labels)
    st.pyplot(fig)
    
    # Display mapping table
    st.write("### Numeric Label to Product Mapping")
    mapping_df = pd.DataFrame({'Label': numeric_labels, 'Product': product_totals.index})
    st.write(mapping_df)
    highest_product = product_totals.idxmax()
    highest_value = product_totals.max()
    st.write(f"### Insights")
    st.write(f"- The most consumed product is **{highest_product}** with **{highest_value} GWh**.")
    st.write(f"- Understanding product-wise consumption can inform energy resource planning and investment decisions.")

def plot_country_product_consumption(df, country):
    st.write(f"### Energy Consumption by Product in {country}")
    st.write(f"This chart breaks down energy consumption in {country} by product, helping to identify energy preferences within the country.")
    
    # Filter data for the selected country
    country_data = df[df['Country'] == country]
    product_totals = country_data.groupby('Product')['Value'].sum().sort_values(ascending=False)
    
    # Generate numeric labels for products
    numeric_labels = range(1, len(product_totals) + 1)
    
    # Plot bar chart with numeric x-axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(numeric_labels, product_totals.values, color='coral')
    ax.set_title(f"Energy Consumption by Product in {country}")
    ax.set_xlabel("Product (Numeric Labels)")
    ax.set_ylabel("Total Energy Consumption (GWh)")
    ax.set_xticks(numeric_labels)
    ax.set_xticklabels(numeric_labels)
    st.pyplot(fig)
    
    # Display mapping table
    st.write("### Numeric Label to Product Mapping")
    mapping_df = pd.DataFrame({'Label': numeric_labels, 'Product': product_totals.index})
    st.write(mapping_df)
    
    # Insights
    highest_product = product_totals.idxmax()
    highest_value = product_totals.max()
    st.write(f"### Insights")
    st.write(f"- In {country}, the most consumed product is **{highest_product}** with **{highest_value} GWh**.")
    st.write(f"- This chart showcases energy usage patterns and preferences specific to {country}.")

def clustering_analysis(df, country=None):
    st.write("### Clustering Analysis")
    st.write(
        """
        Clustering analysis groups energy products based on their consumption levels. 
        This helps identify high-demand, medium-demand, and low-demand products within a country or across all countries.
        """
    )

    def perform_clustering(data, title, is_country_specific=True):
        # Group and preprocess data
        product_totals = data.groupby('Product')['Value'].sum().to_frame().reset_index()
        product_totals.set_index('Product', inplace=True)

        if len(product_totals) < 3:
            st.write("Not enough products for clustering analysis.")
            return None, None, None

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(product_totals)

        # Clustering methods and silhouette scores
        clustering_methods = {
            "KMeans": KMeans(n_clusters=3, random_state=42),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=2),
            "Agglomerative": AgglomerativeClustering(n_clusters=3)
        }
        silhouette_scores = {}
        cluster_results = {}

        for method, model in clustering_methods.items():
            if method == "DBSCAN" and len(scaled_data) < 2:
                continue  # Skip DBSCAN if there are not enough data points

            # Fit and predict clusters
            labels = model.fit_predict(scaled_data)

            # Skip if DBSCAN assigns all points to the same cluster
            if len(set(labels)) < 2:
                silhouette_scores[method] = -1
            else:
                silhouette_scores[method] = silhouette_score(scaled_data, labels)

            cluster_results[method] = labels

        # Select the best method
        best_method = max(silhouette_scores, key=silhouette_scores.get)
        best_labels = cluster_results[best_method]
        product_totals['Cluster'] = best_labels

        # Visualization
        st.write(f"### Best Clustering Method: {best_method} (Silhouette Score: {silhouette_scores[best_method]:.2f})")
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.scatterplot(
            data=product_totals.reset_index(),
            x=range(len(product_totals)),
            y='Value',
            hue='Cluster',
            palette='Set2',
            ax=ax,
            s=100
        )

        # Add numeric labels
        for i, row in product_totals.reset_index().iterrows():
            ax.text(
                i, row['Value'], 
                f"{i + 1}",  # Assign numeric labels
                fontsize=10, ha='center', color='black'
            )
        
        ax.set_title(f"{title} (Best: {best_method})", fontsize=14)
        ax.set_xlabel("Products (Numerically Indexed)", fontsize=12)
        ax.set_ylabel("Energy Consumption (GWh)", fontsize=12)
        ax.legend(title="Cluster")
        st.pyplot(fig)

        # Mapping table
        st.write("### Product Mapping")
        mapping_df = pd.DataFrame({
            'Label': range(1, len(product_totals) + 1),
            'Product': product_totals.index,
            'Cluster': best_labels
        })
        st.dataframe(mapping_df)

        # Cluster Summary
        st.write("### Cluster Breakdown")
        for cluster_id in product_totals['Cluster'].unique():
            cluster_products = product_totals[product_totals['Cluster'] == cluster_id]
            st.write(
                f"- **Cluster {cluster_id}**: {len(cluster_products)} products, including {', '.join(cluster_products.index)}"
            )

        # Insights
        st.write("### Insights")
        st.write(
            f"""
            - The best clustering method was **{best_method}**, determined using the silhouette score.
            - The scatterplot shows clusters with numeric labels for products.
            - Cluster interpretations (e.g., low, medium, high demand) depend on the values in each cluster.
            """
        )

        return best_method, silhouette_scores, product_totals['Cluster']

    # Best method summary
    best_methods = []
    overall_silhouette_scores = []

    # Country-specific clustering
    if country:
        st.write(f"### Clustering Analysis for {country}")
        country_data = df[df['Country'] == country]
        best_method, silhouette_scores, _ = perform_clustering(country_data, f"Clustering of Products in {country}")
        if best_method:
            best_methods.append(best_method)
            overall_silhouette_scores.append(max(silhouette_scores.values()))

    # Overall clustering
    st.write("### Clustering Analysis Across All Countries")
    best_method, silhouette_scores, _ = perform_clustering(df, "Clustering of Products Across All Countries", is_country_specific=False)
    if best_method:
        best_methods.append(best_method)
        overall_silhouette_scores.append(max(silhouette_scores.values()))

    # Summary of the best clustering method
    if best_methods:
        best_method_summary = pd.DataFrame({
            "Clustering Scope": ["Country-Specific" if country else "Global"] * len(best_methods),
            "Best Method": best_methods,
            "Silhouette Score": overall_silhouette_scores
        })
        st.write("### Best Clustering Method Summary")
        st.dataframe(best_method_summary)
    else:
        st.write("No valid clustering results to display.")


def seasonal_decomposition(df, country):
    st.write("### Seasonal Decomposition")
    st.write("""
        Seasonal decomposition splits a time series into three main components:
        - **Trend**: Indicates the long-term direction or movement in the data.
        - **Seasonality**: Captures recurring patterns or cycles in the data.
        - **Residuals**: Represents random noise or unexplained variations.
    """)
    country_ts = filter_country_time_series(df, country)
    for col in country_ts.columns[:3]:
        st.write(f"#### Decomposition for {col}")
        st.write("""
            This visualization breaks the time series into its components to help identify periodic behavior 
            and the underlying trend in the data. 
            - The **Trend** highlights long-term changes in energy usage.
            - **Seasonality** reveals patterns that repeat over time, such as seasonal demand.
            - **Residuals** help assess irregular fluctuations or anomalies in the data.
        """)
        stl = STL(country_ts[col], period=12).fit()
        fig = stl.plot()
        fig.suptitle(f"Seasonal Decomposition of", fontsize=5)
        st.pyplot(fig)

def arima_forecasting(df, country):
    st.write("### ARIMA Forecasting")
    st.write("""
        ARIMA is a statistical modeling technique used for time series forecasting. It stands for:
        - **AR**: Autoregression (using past values to predict future values)
        - **I**: Integrated (differencing to make the data stationary)
        - **MA**: Moving Average (modeling the error term as a combination of past errors).
    """)
    country_ts = filter_country_time_series(df, country)
    for col in country_ts.columns[:1]:
        ts = country_ts[col]
        st.write(f"#### ARIMA Forecasting for {col}")
        
        # Train-test split
        train_size = int(len(ts) * 0.8)
        train, test = ts[:train_size], ts[train_size:]
        
        # Fit ARIMA model
        model = ARIMA(train, order=(1, 1, 1))
        fit = model.fit()
        forecast = fit.forecast(steps=len(test))
        
        # Validation metrics
        rmse = np.sqrt(np.mean((test - forecast) ** 2))
        mape = np.mean(np.abs((test - forecast) / test)) * 100
        
        # Plot results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train, label="Train")
        ax.plot(test.index, test, label="Test", color="orange")
        ax.plot(test.index, forecast, label="Forecast", color="red")
        ax.legend()
        ax.set_title(f"ARIMA Forecasting for {col.split(' - ')[0]}")
        st.pyplot(fig)
        
        st.write(f"**Validation Metrics:**")
        st.write(f"- RMSE: {rmse:.2f}")
        st.write(f"- MAPE: {mape:.2f}%")


def lstm_forecasting(df, country):
    st.write("### LSTM Forecasting")
    st.write("""
        LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network (RNN) designed for sequential data.
        It excels in capturing temporal dependencies and patterns, making it ideal for time series forecasting.
    """)
    country_ts = filter_country_time_series(df, country)
    for col in country_ts.columns[:1]:
        ts = country_ts[col].values.reshape(-1, 1)
        st.write(f"#### LSTM Forecasting for {col}")
        
        # Normalize the data
        scaler = MinMaxScaler()
        scaled_ts = scaler.fit_transform(ts)

        # Prepare data for LSTM
        def create_dataset(data, time_steps=1):
            X, y = [], []
            for i in range(len(data) - time_steps):
                X.append(data[i:i + time_steps, 0])
                y.append(data[i + time_steps, 0])
            return np.array(X), np.array(y)

        time_steps = 12
        X, y = create_dataset(scaled_ts, time_steps)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Train-test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Forecast on test set
        predictions = model.predict(X_test, verbose=0).flatten()
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Validation metrics
        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

        # Plot results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(ts)), ts.flatten(), label="Historical")
        ax.plot(range(len(ts) - len(y_test), len(ts)), y_test, label="Test", color="orange")
        ax.plot(range(len(ts) - len(y_test), len(ts)), predictions, label="Forecast", color="green")
        ax.legend()
        ax.set_title(f"LSTM Forecasting for {col.split(' - ')[0]}")
        st.pyplot(fig)
        
        st.write(f"**Validation Metrics:**")
        st.write(f"- RMSE: {rmse:.2f}")
        st.write(f"- MAPE: {mape:.2f}%")


# Sidebar Navigation Logic
if analysis_option == "Country-Wise Consumption":
    plot_country_wise_consumption(df)
elif analysis_option == "Product-Wise Consumption":
    plot_product_wise_consumption(df)
elif analysis_option == "Country & Product Visualization":
    plot_country_product_consumption(df, selected_country)
elif analysis_option == "Clustering Analysis":
    clustering_analysis(df, selected_country)
elif analysis_option == "Seasonal Decomposition":
    seasonal_decomposition(time_series_df, selected_country)
elif analysis_option == "ARIMA Forecasting":
    arima_forecasting(time_series_df, selected_country)
elif analysis_option == "LSTM Forecasting":
    lstm_forecasting(time_series_df, selected_country)