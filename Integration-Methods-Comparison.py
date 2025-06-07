import numpy as np
import matplotlib.pyplot as plt

class NumericalIntegration:
    def __init__(self):
        self.true_value = None
    
    def function(self, x, choice):
        functions = {
            1: lambda x: x**2,            # x^2
            2: lambda x: np.sin(x),       # sin(x)
            3: lambda x: np.exp(x),       # e^x
            4: lambda x: 1/x,             # 1/x
            5: lambda x: x**3             # x^3
        }
        return functions[choice](x)
    
    def midpoint_method(self, a, b, n, func_choice):
        dx = (b - a) / n
        x_mid = np.linspace(a + dx/2, b - dx/2, n)
        return dx * np.sum(self.function(x_mid, func_choice))
    
    def trapezoid_method(self, a, b, n, func_choice):
        x = np.linspace(a, b, n+1)
        y = self.function(x, func_choice)
        return (b - a) / (2 * n) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    
    def calculate_true_error(self, numerical_result):
        return abs(self.true_value - numerical_result)
    
    def plot_methods_comparison(self, a, b, func_choice, max_iterations=20):
        n_points = np.arange(2, max_iterations + 1)
        midpoint_errors = []
        trapezoid_errors = []
        
        for n in n_points:
            mid_result = self.midpoint_method(a, b, n, func_choice)
            trap_result = self.trapezoid_method(a, b, n, func_choice)
            
            midpoint_errors.append(self.calculate_true_error(mid_result))
            trapezoid_errors.append(self.calculate_true_error(trap_result))
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(n_points, midpoint_errors, 'b-', label='Midpoint Method')
        plt.semilogy(n_points, trapezoid_errors, 'r-', label='Trapezoid Method')
        plt.axhline(y=0.00001, color='g', linestyle='--', label='Error Threshold')
        plt.xlabel('Number of Intervals')
        plt.ylabel('True Error (log scale)')
        plt.title('Error Convergence Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()

def get_true_value(func_choice, a, b):
    true_values = {
        1: lambda a, b: (b**3 - a**3) / 3,           # x^2
        2: lambda a, b: -np.cos(b) + np.cos(a),      # sin(x)
        3: lambda a, b: np.exp(b) - np.exp(a),       # e^x
        4: lambda a, b: np.log(b) - np.log(a),       # 1/x
        5: lambda a, b: (b**4 - a**4) / 4            # x^3
    }
    return true_values[func_choice](a, b)

def main():
    integrator = NumericalIntegration()
    
    print("\nAvailable functions for integration:")
    print("1. x^2")
    print("2. sin(x)")
    print("3. e^x")
    print("4. 1/x")
    print("5. x^3")
    
    func_choice = int(input("\nSelect function (1-5): "))
    
    while True:
        a = float(input("Enter lower bound: "))
        b = float(input("Enter upper bound: "))
        if b > a:
            break
        print("Upper bound must be greater than lower bound. Try again.")
    
    # Set true value based on selected function
    integrator.true_value = get_true_value(func_choice, a, b)
    
    # Test both methods with increasing number of intervals
    intervals = [10, 100, 1000, 10000]
    
    print("\nComparison of Integration Methods:")
    print("\nNumber of Intervals | Midpoint Error  | Trapezoid Error")
    print("-" * 50)
    
    for n in intervals:
        mid_result = integrator.midpoint_method(a, b, n, func_choice)
        trap_result = integrator.trapezoid_method(a, b, n, func_choice)
        
        mid_error = integrator.calculate_true_error(mid_result)
        trap_error = integrator.calculate_true_error(trap_result)
        
        print(f"{n:^17} | {mid_error:.8f} | {trap_error:.8f}")
    
    # Plot error convergence
    integrator.plot_methods_comparison(a, b, func_choice)

if __name__ == "__main__":
    main()
