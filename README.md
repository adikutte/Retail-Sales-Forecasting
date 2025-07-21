Retail Sales Forecasting 

This is an end-to-end Sales Forecasting project using Python and Power BI.

Objective
Predict daily sales for 10 stores and 50 items using historical data, and visualize performance with an interactive dashboard.


Tools & Technologies
-Python (Pandas, Scikit-learn, Matplotlib)
-Random Forest Regressor
-Power BI (for dashboard visualization)
-Git & GitHub


Project Structure
-main.py # Data processing and model
-train.csv # Historical sales data
-test.csv # Test data for prediction
-submission.csv # Final predictions
-dashboard_data.csv # Cleaned export for Power BI
-PowerBI_Dashboard.pbix # Power BI dashboard file
-README.md # Project documentation




ML Model
- Features: store, item, year, month, day, day of week, is_weekend
- Model: Random Forest Regressor
- Validation RMSE: 8.66


Dashboard Highlights (Power BI)
- Total & average sales per store
- Item-wise sales trends
- Date-wise sales overview
- Actual vs Predicted comparison


How to Run
1. Clone the repo
2. Run `main.py` to train and predict
3. Open `PowerBI_Dashboard.pbix` in Power BI to explore visuals


Author
Aditya Kutte
