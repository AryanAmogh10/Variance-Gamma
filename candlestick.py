from dataclasses import dataclass
from datetime import datetime
import pandas

@dataclass
class CandleStick:
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: pandas.Timestamp

    def __str__(self):
        return f"{self.open},{self.high},{self.low},{self.close},{self.volume},{self.timestamp}"

# Find smaller time frame candle in larger time frame
def findCandleIndex(largerCandles: list[CandleStick], smallCandle: CandleStick) -> int: 
    for i in range(len(largerCandles) - 1, -1, -1):
        if smallCandle.timestamp.date() > largerCandles[i].timestamp.date():
            return i       
    return None

def readCandles(fileName: str) -> list[CandleStick]:
    candles = []
    lines = open(fileName, "r").readlines()
    
    for line in lines:
        data = line.split(",")
        candles.append(CandleStick(float(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4]), pandas.Timestamp(data[5])))
    
    return candles