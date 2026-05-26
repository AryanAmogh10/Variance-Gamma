import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace
import warnings

import candlestick as cs

# # Function to plot histogram
# def plotHist(data: list[float], bins=100, normalize=True):
#     weights = np.ones_like(data) / len(data)
#     values, bins, patches = plt.hist(data, weights=weights if normalize else None, bins=bins, density=False)
#     return values, bins, patches

# Function to plot Laplace distribution PDF
def plotLaplace(bins: list[float], params: tuple[float]):
    loc, scale = params
    probs = []
    
    for i in range(1, len(bins)):
        x1 = bins[i - 1]
        x2 = bins[i]
        prob = laplace.cdf(x2, loc=loc, scale=scale) - laplace.cdf(x1, loc=loc, scale=scale)
        plt.plot([x1, x2], [prob, prob], color="black")
        probs.append(prob)

    return probs

# Function to fit data to Laplace distribution using MLE
def bestFitLaplace(data: list[float]):
    # Fit to Laplace distribution
    print("\nFitting to Laplace Distribution using MLE:")
    loc, scale = laplace.fit(data, method="mle")
    print(f"Estimated Location (loc): {loc}")
    print(f"Estimated Scale (scale): {scale}")

    # Plot Laplace PDF
    # plotLaplace(bins, (loc, scale))
    # plt.title("Laplace Fit to Data")
    
    return loc, scale