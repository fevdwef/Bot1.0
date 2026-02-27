import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy.optimize import linprog

# Function for basic data analysis
def data_analysis(data):
    print("Descriptive Statistics:", data.describe())
    data.hist()

# Function for anomaly detection using Isolation Forest
def detect_anomalies(data):
    model = IsolationForest()
    model.fit(data)
    anomalies = model.predict(data)
    return anomalies

# Function for clustering using DBSCAN
def cluster_data(data):
    model = DBSCAN()
    clusters = model.fit_predict(data)
    return clusters

# Function for linear optimization
def optimize_linear(c, A, b):
    res = linprog(c, A_ub=A, b_ub=b)
    return res.optimal_value, res.x

# Example usage of the functions
if __name__ == '__main__':
    # Sample data
    data = pd.DataFrame(np.random.randn(100, 5), columns=list('ABCDE'))
    data_analysis(data)
    anomalies = detect_anomalies(data)
    print("Anomalies detected:", anomalies)
    clusters = cluster_data(data)
    print("Cluster labels:", clusters)
    c = [1, 2]  # Coefficients for optimization
    A = [[-1, -1], [1, 2], [1, -1]]  # Inequality constraints
    b = [-1, 3, 1]
    optimal_value, solution = optimize_linear(c, A, b)
    print("Optimal value:", optimal_value, "at solution:", solution)