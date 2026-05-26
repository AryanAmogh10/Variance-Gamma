import yfinance as yf
from candlestick import CandleStick
import pandas
from pandas import DataFrame
import datetime
import os

MAX_DAILY_PERIOD = '2y'
RAW_STOCK_FILE = "rawStockList.txt"
DAILY_STOCK_FILE = "dailyStockList.txt"

def makeStockFileName(stockCode: str, interval) -> str:
    return stockCode[:-3] + "__" + interval + ".txt"

def extractRawStockCode(rawFileLine: str):
    if ".NS" not in rawFileLine:
        return None

    return rawFileLine.split(".NS")[0] + ".NS"

def extractTimeStamp(dataFileLine: str) -> pandas.Timestamp:
    timestampStr = dataFileLine.split(",")[-1].strip()
    return pandas.to_datetime(timestampStr)

def addOneDay(lastDate: pandas.Timestamp) -> pandas.Timestamp:
    return lastDate + pandas.DateOffset(days=1)

def extractLatestTimeStamp(fileName: str) -> pandas.Timestamp:
    if not fileExists(fileName):
        raise FileNotFoundError("File does not exist")

    lines = open(fileName, "r").readlines()

    if len(lines) == 0:
        raise ValueError("File is empty")

    lastLine = lines[-1]
    return extractTimeStamp(lastLine)

def lastWorkingDate() -> datetime.date:
    today = pandas.Timestamp.today().date()

    if today.weekday() == 5:
        return pandas.Timestamp(today - pandas.DateOffset(days=1)).date()
    elif today.weekday() == 6:
        return pandas.Timestamp(today - pandas.DateOffset(days=2)).date()
    else:
        return pandas.Timestamp(today).date()

def appendStockData(fileName: str, data, mode="a"):
    f = open(fileName, mode)

    for i in range(len(data)):
        candle = CandleStick(round(data.iloc[i]['Open'], 6),
                                round(data.iloc[i]['High'], 6),
                                round(data.iloc[i]['Low'], 6),
                                round(data.iloc[i]['Close'], 6),
                                data.iloc[i]['Volume'],
                                data.iloc[i].name)

        f.write(candle.__str__() + "\n")

    f.close()

def fileExists(fileName) -> bool:
    exists = True

    try:
        dataFile = open(fileName, "r")
    except FileNotFoundError:
        exists = False

    return exists

def fileIsEmpty(fileName) -> bool:
    return len(open(fileName, "r").readlines()) == 0

def update5mData(stockCode: str, errorStockList: list[str]):
    fileName = makeStockFileName(stockCode, "5m")

    if fileExists(fileName):
        return

    # Find date 59 days ago
    startDate = (lastWorkingDate() - datetime.timedelta(days=10)).strftime("%Y-%m-%d")
    data = yf.download(stockCode, interval="5m", start=startDate, progress=False, auto_adjust=True, repair=True)

    if len(data) == 0:
        errorStockList.append(stockCode)
        return

    appendStockData(fileName, data, "w")

def updateDailyData(stockCode: str, errorStocks: list[str]):
    # fileName = makeStockFileName(stockCode, "1d") 

    # data = []

    # if not fileExists(fileName) or fileIsEmpty(fileName):
    #     data = yf.download(stockCode, interval="1d", period=MAX_DAILY_PERIOD, progress=False, repair=True, auto_adjust=True)
    # else:
    #     file = open(fileName, "r")

    #     lastTimeStamp = extractLatestTimeStamp(fileName)
    #     if lastTimeStamp.date() == lastWorkingDate():
    #         return

    #     nextDateStr = addOneDay(lastTimeStamp).date().strftime("%Y-%m-%d")
    #     data = yf.download(stockCode, interval="1d", start=nextDateStr, progress=False, repair=True)

    # if len(data) == 0:
    #     errorStocks.append(stockCode)
    #     return

    # appendStockData(fileName, data)

    fileName = "NIFTY__5m.txt"

    # start today-59 days, end today 
    start = (lastWorkingDate() - datetime.timedelta(days=59)).strftime("%Y-%m-%d")
    end   = lastWorkingDate().strftime("%Y-%m-%d")

    print("Extracting data from " + start + " to " + end, end="... ", flush=True)
    data = yf.download("^NSEI", interval="5m", start=start, end=end, progress=True, repair=True, auto_adjust=True)
    print("Done")   

    appendStockData(fileName, data, "w")

# remove error stocks from raw file
def cleanRawFile(errorStocks: list[str]):
    # Write to a new file then rename it
    newFile = open("newRawStockList.txt", "w")
    rawFile = open(RAW_STOCK_FILE, "r")

    for line in rawFile.readlines():
        # Remove BO stock codes from the raw file
        if ".BO" in line:
            continue

        if not ".NS" in line:
            continue

        if extractRawStockCode(line) not in errorStocks:
            newFile.write(line)

    rawFile.close()
    newFile.close()

    os.remove(RAW_STOCK_FILE)
    os.rename("newRawStockList.txt", RAW_STOCK_FILE)


def updateData():
    # rawFile =      open(RAW_STOCK_FILE, "r")
    # dailyStockList = open(DAILY_STOCK_FILE, "w")

    # errorStocks = []

    # count = 0
    # lines = rawFile.readlines()
    # for line in lines:
    #     count += 1
    #     print("\r" + " "*100, end="")
    #     print(f"\rUpdating data... {count/len(lines)*100:.2f}%", end="")

    #     stockCode = extractRawStockCode(line)

    #     if stockCode is None:
    #         continue

    #     try:
    #         updateDailyData(stockCode, errorStocks)
    #         # update5mData(stockCode, errorStocks)
    #     except Exception as e:
    #         print(f"Error updating {stockCode}: {e}")
    #         errorStocks.append(stockCode)
    #         continue

    #     if fileExists(makeStockFileName(stockCode, "1d")):
    #         dailyStockList.write(stockCode + "\n")

    # rawFile.close()
    # dailyStockList.close()

    # cleanRawFile(errorStocks)
    updateDailyData("^NSEI", [])


if __name__ == "__main__":
    updateData()