from candlestick import *
from typing import List
from datetime import datetime
import matplotlib.pyplot as plt
import math 
import numpy as np
from distribs import *
import scipy 
import scipy.stats as st
import vg

# from gofs import *

from laplace import *

# Fractional returns of a list of candles
def getReturns(candles: list[CandleStick]) -> list[float]:
    returns = []
    for i in range(1, len(candles)):
        returns.append((candles[i].close - candles[i - 1].close) / candles[i - 1].close)
    return returns

def getLogReturns(candles: list[CandleStick]) -> list[float]:
    returns = []
    for i in range(1, len(candles)):
        returns.append(math.log(candles[i].close / candles[i - 1].close))
    return returns

def normalDistrib(x: float, mean: float, std: float) -> float:
    return math.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi))

def normalCDF(x: float, mean: float, std: float) -> float:
    return 0.5 * (1 + math.erf((x - mean) / (std * math.sqrt(2))))

def calculateNormalCDFs(lower: float, upper: float, mean: float, std: float, step: float): 
    cdfs = []
    originalLower = lower

    while lower < upper: 
        cdfs.append(normalCDF(lower, mean, std))
        lower += step

    return cdfs

def getNormalParams(returns: list[float]) -> tuple[float, float]:
    mean = sum(returns) / len(returns)
    std = math.sqrt(sum([(x - mean) ** 2 for x in returns]) / len(returns))
    return mean, std

def getCumulativeReturns(returns: list[float], lower: float, upper: float, binSize: float) -> list[float]:
    cdfs = []

    while lower < upper: 
        cdfs.append(sum([1 for x in returns if x < lower]) / len(returns))
        lower += binSize

    return cdfs

def calculateKolmogorovStat(empCdfs: list[float], theoCdfs: list[float]) -> float:
    return max([abs(empCdfs[i] - theoCdfs[i]) for i in range(len(empCdfs))])

def calculateMSE(empValues: list[float], theoValues: list[float]) -> float: 
    return sum([(empValues[i] - theoValues[i]) ** 2 for i in range(len(empValues))]) / len(empValues)

def calculateR2(empValues: list[float], theoValues: list[float]) -> float:
    mean = sum(empValues) / len(empValues)
    return 1 - sum([(empValues[i]-theoValues[i])**2 for i in range(len(empValues))]) / sum([(x - mean) ** 2 for x in empValues])

def calculatePearsonChiSquared(empValues: list[float], theoValues: list[float]) -> float:
    return sum([((empValues[i] - theoValues[i]) ** 2) / theoValues[i] for i in range(len(empValues))])

# Remove top and bottom 0.5% of returns
def removeOutliers(returns: list[float]):
    returns.sort()
    return returns[int(0.5/100.0 * len(returns)):int((1-(0.5/100.0)) * len(returns))]

def createDatasets(candles: list[CandleStick], trainingFraction: float = 0.7) -> list[list[float]]:
    validationSet = []
    trainingSet   = []

    x = trainingFraction
    # From every 10% of the data, x% is used for training and (10-x)% for validation

    returns = getLogReturns(candles)

    # Shuffle the 10%, but keep the chronological order for the 10 groups 
    for i in range(0, len(returns), len(returns) // 10): 
        focus = returns[i:i + len(returns) // 10]

        np.random.shuffle(focus)

        for j in range(len(focus)):
            if j < x * len(focus):
                trainingSet.append(focus[j])
            else: 
                validationSet.append(focus[j])

    return removeOutliers(trainingSet), removeOutliers(validationSet)

def main(): 
    candles = readCandles("data/NIFTY__1d.txt")

    allReturns = getLogReturns(candles)
    allReturns = removeOutliers(allReturns)
    nBins   = 30

    # Randomly shuffle the returns
    np.random.shuffle(allReturns)
    print(f"Read {len(allReturns)} returns")

    # print("Fitting VarGamma")
    # params = vg.fit(allReturns)
    # print("Params: ", params)
    params=(0.0008243822827291212, 0.011302473367647399, -0.00020185045936517996, 0.7721750006691915)
    
    # for strike in range(23000, 26000, 100):
    #     spot = 23688.95
    #     print(f"{strike} CALL option price: {calculateCallOptPrice(params, strike, spot, 3)} PUT option price: {calculatePutOptPrice(params, strike, spot, 3)}") 
    values, bins, patches = plotHist(allReturns, "grey", 20)
    theValues = vg.plotVG(bins, params, 1)

    print("MSE for Var-Gamma: ", calculateMSE(values, theValues))

    # print("R2 for Var-Gamma: ", calculateR2(values, theValues))

    # params = bestFitLaplace(allReturns)
    # theValues = plotLaplace(bins, params)

    # print("MSE for Laplace: ", calculateMSE(values, theValues))
    # print("R2 for Laplace: ", calculateR2(values, theValues))

    plt.show()

if __name__ == "__main__":
    main()