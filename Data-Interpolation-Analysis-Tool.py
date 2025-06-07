import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

class InterpolationAnalysis:
    def __init__(self):
        # Indonesian Population data
        self.population_years = np.array([2010, 2015, 2016, 2017, 2018])
        self.population_values = np.array([238, 255, 258, 261, 264])
        
        # Rice Import data
        self.rice_years = np.array([2017, 2018, 2020, 2022, 2023])
        self.rice_values = np.array([0.14, 1.04, 0.20, 0.20, 1.79])

    def interpolate_data(self, x, y, x_new, method='cubic'):
        """Perform interpolation using specified method"""
        f = interpolate.interp1d(x, y, kind=method)
        return f(x_new)

    def calculate_errors(self, predicted, actual):
        """Calculate absolute and relative errors"""
        absolute_error = np.abs(predicted - actual)
        relative_error = absolute_error / actual * 100
        return absolute_error, relative_error

    def compare_interpolation_methods(self, x, y, x_test, y_test, methods=['linear', 'quadratic', 'cubic']):
        """Compare different interpolation methods and their errors"""
        results = {}
        for method in methods:
            try:
                y_pred = self.interpolate_data(x, y, x_test, method)
                abs_err, rel_err = self.calculate_errors(y_pred, y_test)
                results[method] = {
                    'predicted': y_pred,
                    'absolute_error': abs_err,
                    'relative_error': rel_err
                }
            except:
                continue
        return results

    def plot_interpolation(self, x, y, method, title=''):
        """Plot original data and interpolated results"""
        x_new = np.linspace(min(x), max(x), 100)
        y_new = self.interpolate_data(x, y, x_new, method)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='blue', label='Original Data')
        plt.plot(x_new, y_new, 'r-', label=f'{method.capitalize()} Interpolation')
        plt.title(title)
        plt.xlabel('Year')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    analysis = InterpolationAnalysis()
    
    # Test data for validation (you can input actual data here)
    test_year_population = 2014
    test_value_population = 252  # Example actual value
    
    test_year_rice = 2019
    test_value_rice = 0.5  # Example actual value
    
    # Population data analysis
    print("Indonesian Population Analysis:")
    population_methods = ['linear', 'quadratic', 'cubic']
    pop_results = analysis.compare_interpolation_methods(
        analysis.population_years, 
        analysis.population_values,
        np.array([test_year_population]),
        np.array([test_value_population]),
        population_methods
    )
    
    # Display results for population data
    for method, result in pop_results.items():
        print(f"\n{method.capitalize()} Interpolation:")
        print(f"Predicted value for {test_year_population}: {result['predicted'][0]:.2f}")
        print(f"Absolute Error: {result['absolute_error'][0]:.2f}")
        print(f"Relative Error: {result['relative_error'][0]:.2f}%")
        
        # Plot the best method
        analysis.plot_interpolation(
            analysis.population_years,
            analysis.population_values,
            method,
            f'Indonesian Population Interpolation ({method})'
        )
    
    # Rice Import data analysis
    print("\nRice Import Analysis:")
    rice_methods = ['linear', 'cubic', 'quadratic']
    rice_results = analysis.compare_interpolation_methods(
        analysis.rice_years,
        analysis.rice_values,
        np.array([test_year_rice]),
        np.array([test_value_rice]),
        rice_methods
    )
    
    # Display results for rice import data
    for method, result in rice_results.items():
        print(f"\n{method.capitalize()} Interpolation:")
        print(f"Predicted value for {test_year_rice}: {result['predicted'][0]:.2f}")
        print(f"Absolute Error: {result['absolute_error'][0]:.2f}")
        print(f"Relative Error: {result['relative_error'][0]:.2f}%")
        
        # Plot the best method
        analysis.plot_interpolation(
            analysis.rice_years,
            analysis.rice_values,
            method,
            f'Rice Import Interpolation ({method})'
        )

if __name__ == "__main__":
    main()
