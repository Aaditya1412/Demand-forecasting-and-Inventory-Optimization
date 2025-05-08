# Demand-forecasting-and-Inventory-Optimization
This project focuses on predicting demand through time series analysis and enhancing inventory management based on the forecasted demand.
Getting Started
Prerequisites
To execute this project, ensure you have the following Python libraries installed:

numpy

pandas

matplotlib

plotly

statsmodels

Install these libraries using pip:

nginx
Copy
Edit
pip install -r requirements.txt
Installation
Clone the repository to your system:

bash
Copy
Edit
git clone https://github.com/AyushMehta1702/Demand-Forecasting-and-Inventory-Optimization-using-Python---Time-Series.git
Move to the project directory:

sql
Copy
Edit
cd Demand-Forecasting-and-Inventory-Optimization-using-Python---Time-Series
Usage
Data Preparation
Ensure that a CSV file named demand_inventory.csv is present, containing the columns 'Date', 'Demand', and 'Inventory'.

Demand Forecasting
Execute the code to perform demand forecasting using SARIMA. Adjust the SARIMA order and seasonal_order parameters as per your dataset.

makefile
Copy
Edit
order = (1, 1, 1,)
seasonal_order = (1, 1, 1, 2)

model = SARIMAX(time_series, order=order, seasonal_order=seasonal_order)
model_fit = model.fit(disp=False)

future_steps = 10

prediction = model_fit.predict(len(time_series), len(time_series) + future_steps - 1)
prediction = prediction.astype(int)
print(prediction)
Inventory Optimization
Compute optimal order quantity, reorder point, safety stock, and total cost using the Newsvendor formula. Adjust parameters such as lead time, service level, holding cost, and stockout cost as per business needs.

ini
Copy
Edit
future_dates = pd.date_range(start=time_series.index[-1] + pd.DateOffset(days=1), periods=future_steps, freq='D')

forecast_demand = pd.Series(prediction, index=future_dates)

initial_inventory = 5500
lead_time = 1
service_level = 0.95

z = np.abs(np.percentile(forecast_demand, 100 * (1 - service_level)))
order_quantity = np.ceil(forecast_demand.mean() + z).astype(int)
reorder_point = forecast_demand.mean() * lead_time + z
safety_stock = reorder_point - forecast_demand.mean() * lead_time

holding_cost = 0.1
stockout_cost = 10

total_h_cost = holding_cost * (initial_inventory + 0.5 * order_quantity)
total_s_cost = stockout_cost * np.maximum(0, forecast_demand.mean() * lead_time - initial_inventory)
total_cost = total_h_cost + total_s_cost
Time Series Analysis
A summary of different time series analysis models:

Model	Explanation
ARMA	Combines autoregressive and moving average components for time series forecasting without accounting for seasonality.
ARIMA	Extends ARMA by including differencing, making it suitable for non-seasonal data.
SARIMA	Builds on ARIMA by incorporating seasonality, useful for seasonal time series data.

Why SARIMA
SARIMA (Seasonal Autoregressive Integrated Moving Average) is chosen for this project because it effectively models data with seasonal patterns, common in demand forecasting. It combines autoregressive, differencing, and moving average components, allowing for accurate short-term and long-term demand predictions. This model is adaptable to various seasonal patterns, making it suitable for diverse business scenarios.

Dataset
The dataset used is the "Inventory Optimization: Case Study" dataset, which provides historical data on demand and inventory levels, making it ideal for analyzing and optimizing inventory management.

Results
The project provides the optimal order quantity, reorder point, safety stock, and total cost as outputs.

bash
Copy
Edit
print("Optimal Order Quantity:", order_quantity)
print("Reorder Point:", reorder_point)
print("Safety Stock:", safety_stock)
print("Total Cost:", total_cost)
Example output:

yaml
Copy
Edit
Optimal Order Quantity: 236  
Reorder Point: 235.25  
Safety Stock: 114.45  
Total Cost: 561.80  
Contributing
To contribute, please open an issue or submit a pull request. Contributions and suggestions are welcome.

License
This project is licensed under the MIT License.
