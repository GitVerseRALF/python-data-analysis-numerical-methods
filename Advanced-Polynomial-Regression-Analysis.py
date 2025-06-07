import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

class RegressionAnalysis:
    def __init__(self):
        # Black Pepper Export data
        self.pepper_years = np.array([2012, 2013, 2014, 2015, 2016, 2017])
        self.pepper_values = np.array([298.13, 186.05, 136.47, 302.02, 220.68, 87.18])
        
        # Telkom Stock Price data
        self.telkom_years = np.array([2017, 2018, 2019, 2020, 2021])
        self.telkom_values = np.array([3980, 4440, 3750, 3970, 3310])
        
        # Hotel Occupation data
        self.hotel_months = np.array([3, 4, 5, 6, 7])
        self.hotel_values = np.array([32.24, 12.67, 14.45, 19.7, 28.07])

    def perform_regression(self, x, y, degree=1):
        """Perform polynomial regression of specified degree"""
        x = x.reshape(-1, 1)
        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(x)
        
        model = LinearRegression()
        model.fit(x_poly, y)
        
        # Calculate R-squared
        y_pred = model.predict(x_poly)
        r2 = r2_score(y, y_pred)
        
        return model, poly, r2

    def predict_values(self, model, poly, x_new):
        """Make predictions using the fitted model"""
        x_new = np.array(x_new).reshape(-1, 1)
        x_new_poly = poly.transform(x_new)
        return model.predict(x_new_poly)

    def plot_regression(self, x, y, model, poly, title, future_years=2):
        """Plot original data and regression line with future predictions"""
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='blue', label='Original Data')
        
        # Generate points for smooth curve
        x_range = np.linspace(min(x), max(x) + future_years, 100).reshape(-1, 1)
        x_poly = poly.transform(x_range)
        y_pred = model.predict(x_poly)
        
        plt.plot(x_range, y_pred, 'r-', label='Regression Line')
        plt.title(title)
        plt.xlabel('Year/Month')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    analysis = RegressionAnalysis()
    
    # Black Pepper Export Analysis
    print("Black Pepper Export Analysis:")
    pepper_model, pepper_poly, pepper_r2 = analysis.perform_regression(
        analysis.pepper_years, analysis.pepper_values, degree=2)
    analysis.plot_regression(
        analysis.pepper_years, 
        analysis.pepper_values,
        pepper_model,
        pepper_poly,
        'Black Pepper Export Regression'
    )
    print(f"R-squared score: {pepper_r2:.4f}")
    
    # Predict next year
    next_year = 2018
    prediction = analysis.predict_values(pepper_model, pepper_poly, [next_year])
    print(f"Predicted value for {next_year}: {prediction[0]:.2f}")
    
    # Telkom Stock Price Analysis
    print("\nTelkom Stock Price Analysis:")
    telkom_model, telkom_poly, telkom_r2 = analysis.perform_regression(
        analysis.telkom_years, analysis.telkom_values, degree=2)
    analysis.plot_regression(
        analysis.telkom_years,
        analysis.telkom_values,
        telkom_model,
        telkom_poly,
        'Telkom Stock Price Regression'
    )
    print(f"R-squared score: {telkom_r2:.4f}")
    
    # Hotel Occupation Analysis
    print("\nHotel Occupation Analysis:")
    hotel_model, hotel_poly, hotel_r2 = analysis.perform_regression(
        analysis.hotel_months, analysis.hotel_values, degree=3)
    analysis.plot_regression(
        analysis.hotel_months,
        analysis.hotel_values,
        hotel_model,
        hotel_poly,
        'Hotel Occupation Regression',
        future_years=1
    )
    print(f"R-squared score: {hotel_r2:.4f}")

if __name__ == "__main__":
    main()
