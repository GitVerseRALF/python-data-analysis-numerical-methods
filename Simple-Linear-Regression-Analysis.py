
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# a. Black Pepper Export Data
years_pepper = np.array([2012, 2013, 2014, 2015, 2016, 2017]).reshape(-1, 1)
export_values = np.array([298.13, 186.05, 136.47, 302.02, 220.68, 87.18])

# b. Telkom Stock Price Data
years_telkom = np.array([2017, 2018, 2019, 2020, 2021]).reshape(-1, 1)
stock_prices = np.array([3980, 4440, 3750, 3970, 3310])

# c. Hotel Occupation Data
months = np.array([3, 4, 5, 6, 7]).reshape(-1, 1)
occupation = np.array([32.24, 12.67, 14.45, 19.7, 28.07])

# Create regression models
model_pepper = LinearRegression()
model_telkom = LinearRegression()
model_hotel = LinearRegression()

# Fit the models
model_pepper.fit(years_pepper, export_values)
model_telkom.fit(years_telkom, stock_prices)
model_hotel.fit(months, occupation)

# Create plots
plt.figure(figsize=(15, 5))

# Plot Black Pepper Export
plt.subplot(131)
plt.scatter(years_pepper, export_values, color='blue')
plt.plot(years_pepper, model_pepper.predict(years_pepper), color='red')
plt.title('Black Pepper Export Regression')
plt.xlabel('Year')
plt.ylabel('Export Value (Million USD)')

# Plot Telkom Stock Price
plt.subplot(132)
plt.scatter(years_telkom, stock_prices, color='blue')
plt.plot(years_telkom, model_telkom.predict(years_telkom), color='red')
plt.title('Telkom Stock Price Regression')
plt.xlabel('Year')
plt.ylabel('Stock Price (Rupiah)')

# Plot Hotel Occupation
plt.subplot(133)
plt.scatter(months, occupation, color='blue')
plt.plot(months, model_hotel.predict(months), color='red')
plt.title('Hotel Occupation Regression')
plt.xlabel('Month')
plt.ylabel('Occupation Percentage')

plt.tight_layout()
plt.show()

# Print regression equations and R-squared values
print("\nRegression Results:")
print("\n1. Black Pepper Export:")
print(f"y = {model_pepper.coef_[0]:.2f}x + {model_pepper.intercept_:.2f}")
print(f"R² = {model_pepper.score(years_pepper, export_values):.4f}")

print("\n2. Telkom Stock Price:")
print(f"y = {model_telkom.coef_[0]:.2f}x + {model_telkom.intercept_:.2f}")
print(f"R² = {model_telkom.score(years_telkom, stock_prices):.4f}")

print("\n3. Hotel Occupation:")
print(f"y = {model_hotel.coef_[0]:.2f}x + {model_hotel.intercept_:.2f}")
print(f"R² = {model_hotel.score(months, occupation):.4f}")
    