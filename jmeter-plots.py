import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import re
from scipy import stats
import sys
import time
from urllib.parse import urlparse
import warnings
warnings.filterwarnings("ignore")

def main():
    p = argparse.ArgumentParser(description='Takes an input CSV from a JMeter load test and graphs the performance of each thread', formatter_class=HelpFormatOverride)
    p.add_argument('-r','--results',type=str,required=True,help='The file path to the test results CSV')
    p.add_argument('-c','--compare',type=str,required=False,help='The file path to a comparision results CSV, using -r\'s argument as a baseline')
    p.add_argument('-p','--plot',type=str,required=False,default='candlestick',help='R|The type of plot to be generated from the test results. Options include:\n -"Candlestick": side-by-side box-and-whisker plots of each column\'s latencies\n -"CorrelationBPlot": a box plot of distribution correlations for each column for an aggregate of command-line outputs from the creation of Q-Q plots\n -"Histogram": histograms showing details on the distribution in each column\n -"Q-Q": quantile-quantile plots comparing the distributions of latency irrespective of magnitude. Requires --compare')
    p.add_argument('-o','--output',type=str,required=False,help='The filename to save a chart as')
    args = p.parse_args()

    match args.plot.lower():
        case 'candlestick':
            threadResponses, pathResponses = jmeterCSV(args.results)
            if args.compare:
                comparisionThreadResponses, comparisionPathResponses = jmeterCSV(args.compare)
            plt.figure(figsize=[16,9],dpi=120)
            threadPlot = threadResponses.tablePlot()
            if args.compare:
                comparisionThreadResponses = threadResponses.scaleToSelf(comparisionThreadResponses)
                comparisionThreadPlot = comparisionThreadResponses.tablePlot()
                translatePlot(threadPlot,-0.1)
                translatePlot(comparisionThreadPlot,0.1)
            plt.subplots_adjust(bottom=.16, top=.96)
            plt.xticks(range(len(threadResponses.table)+1)[1:], list(threadResponses.table), fontsize=10, rotation=30)
            plt.gca().set_ylim([0,400])
            if args.output:
                plt.savefig(args.output + '-' + str(time.time()) + '.png')
            else:
                plt.show()

            plt.figure(figsize=[16,9],dpi=120)
            pathPlot = pathResponses.tablePlot(outliers=True)
            if args.compare:
                comparisionPathResponses = pathResponses.scaleToSelf(comparisionPathResponses)
                comparisionPathPlot = comparisionPathResponses.tablePlot(outliers=True)
                translatePlot(pathPlot,-0.1)
                translatePlot(comparisionPathPlot,0.1)
            plt.subplots_adjust(bottom=.16, top=.96)
            plt.xticks(range(len(pathResponses.table)+1)[1:], list(pathResponses.table), fontsize=10, rotation=30)
            plt.gca().set_ylim([0,400])
            if args.output:
                plt.savefig(args.output + '-' + str(time.time()) + '.png')
            else:
                plt.show()
        case 'q-q':
            if args.compare:
                threadResponses, pathResponses = jmeterCSV(args.results)
                comparisionThreadResponses, comparisionPathResponses = jmeterCSV(args.compare)
                results = enumTable.join([threadResponses, pathResponses]).zScores()
                compare = enumTable.join([comparisionThreadResponses, comparisionPathResponses]).zScores()
                intersection = np.isin(list(results.table), list(compare.table), assume_unique=True)
                shared_indexes = []
                for i in range(len(intersection)):
                    if intersection[i]:
                        shared_indexes.append(list(results.table)[i])
                max_rows = np.uint8(np.round(np.sqrt(len(shared_indexes))))
                max_cols = np.uint8(np.ceil(np.sqrt(len(shared_indexes))))

                fig, axes = plt.subplots(nrows=max_rows,ncols=max_cols,figsize=[16,9],dpi=120,squeeze=False)
                residuals = {}
                for row in range(max_rows):
                    for col in range(max_cols):
                        index = row * max_cols + col
                        if index < len(shared_indexes):
                            column = list(shared_indexes)[index]
                            axes[row,col], res = qQPlot(results.table[column][0], compare.table[column][0], axes[row, col])
                            residuals.update({column : res})
                            axes[row, col].set_title(column, fontsize=10)
                plt.subplots_adjust(bottom=0.05, top=0.95, hspace=0.3)
                if args.output:
                    plt.savefig(args.output + '-' + str(time.time()) + '.png')
                else:
                    plt.show()

                fig, axes = plt.subplots(nrows=max_rows,ncols=max_cols,figsize=[16,9],dpi=120,squeeze=False)
                for row in range(max_rows):
                    for col in range(max_cols):
                        index = row * max_cols + col
                        if index < len(shared_indexes):
                            column = list(shared_indexes)[index]
                            controlData = np.random.normal(0,1,len(residuals[column]))
                            qQPlot(residuals[column], controlData, axes[row, col])
                            axes[row, col].set_title(column, fontsize=10)
                            correlation = 100 * np.corrcoef(residuals[column], controlData)[0][1]
                            print(','.join([args.results, args.compare, column, str(abs(correlation))]))
                plt.subplots_adjust(bottom=0.05, top=0.95, hspace=0.3)
                if args.output:
                    plt.savefig(args.output + '-residual-' + str(time.time()) + '.png', dpi=300)
                else:
                    plt.show()
            else:
                print('ERROR: Q-Q plot requires a comparision CSV!')
        case 'correlationbplot':
            with open(args.results, 'r') as csvfile:
                f = csv.reader(csvfile, delimiter=',')
                data = list(f)

            cData = {}
            for label in data[0]:
                cData.update({label : []})

            for row in data[1:]:
                for i in range(len(row)):
                    cData[list(cData)[i]].append(row[i])

            columnSort = enumTable(cData, 'Column', 'Correlation')
            for column in list(columnSort.table):
                columnSort.table[column] = np.round(np.array(columnSort.table[column]).astype(np.float64)).astype(int,casting='unsafe')
                for i in range(len(columnSort.table[column])):
                    columnSort.table[column][i] = columnSort.table[column][i] if columnSort.table[column][i]>=0 else 0 # eliminates NaNs which would otherwise appear as -9223372036854775808% p-values

            plt.figure(figsize=[16,9],dpi=120)
            columnSort.tablePlot(outliers=True)
            plt.subplots_adjust(bottom=.2, top=.96,left=.05,right=.95)
            plt.xticks(range(len(columnSort.table)+1)[1:], list(columnSort.table), fontsize=10, rotation=50)
            if args.output:
                plt.savefig(args.output + '-' + str(time.time()) + '.png')
            else:
                plt.show()
        case 'histogram':
            fullResponses = enumTable.join(jmeterCSV(args.results))
            max_rows = np.uint8(np.round(np.sqrt(len(list(fullResponses.table)))))
            max_cols = np.uint8(np.ceil(np.sqrt(len(list(fullResponses.table)))))
            fig, axes = plt.subplots(nrows=max_rows,ncols=max_cols,figsize=[16,9],dpi=120,squeeze=False)

            for row in range(max_rows):
                for col in range(max_cols):
                    index = row * max_rows + col
                    if index < len(list(fullResponses.table)):
                        column = list(fullResponses.table)[index]
                        axes[row, col].hist(fullResponses.table[column], 25)
                        axes[row, col].set_title(column, fontsize=10)
                    axes[row, col].tick_params(axis='x', labelbottom=False, bottom=False)
            plt.subplots_adjust(bottom=0.05, top=0.95, hspace=0.3)
            if args.output:
                plt.savefig(args.output + '-' + str(time.time()) + '.png')
            else:
                plt.show()

        case _:
            print('Unspecified plot type!')

# a class that assigns a column of data a value for a user-defined enumerated type
class enumTable():
    # takes a dict with keys enumKey and dataKey and returns the enumTable where all values of rawData[enumKey] are enumerated and list their corresponding rawData[dataKey] values
    def __init__(self, rawData, enumKey, dataKey):
        self.table = {}
        for i in range(len(rawData[enumKey])):
            curKey = rawData[enumKey][i]
            curData = rawData[dataKey][i]
            if curKey not in self.table:
                self.table.update({curKey : [curData]})
            else:
                self.table[curKey].append(curData)

    # returns an enumTable which joins together all columns in a list of enumTables
    @staticmethod
    def join(tables):
        rawInput = {'enum' : [], 'data': []}
        for t in tables:
            for column in list(t.table):
                rawInput['enum'].append(column)
                rawInput['data'].append(t.table[column])
        return enumTable(rawInput, 'enum', 'data')

    # copies another enumTable and scales all of its data to the mean and standard deviation of the intersecting enum values between the tables, returning this copy
    def scaleToSelf(self, other):
        rawInput = {'enum' : [], 'data': []}
        for column in list(self.table):
            if column in list(other.table):
                sCol = np.array(self.table[column], dtype=np.int32)
                oCol = np.array(other.table[column], dtype=np.int32)
                scalarStDev = np.std(sCol, ddof=0)
                scalarMean = np.average(sCol)
                baseStDev = np.std(oCol, ddof=0)
                baseMean = np.average(oCol)
                for i in oCol:
                    rawInput['enum'].append(column)
                    zScore = (i-baseMean)/baseStDev
                    rawInput['data'].append(np.int32(zScore*scalarStDev + scalarMean))
        return enumTable(rawInput, 'enum', 'data')

    # creates a box plot for each enum value of the table; highlights the box red if it does not strongly correlate to a normal distribution
    def tablePlot(self, outliers=False):
        dataset = []
        pStats = []
        for column in list(self.table):
            columnData = np.array(self.table[column], dtype=np.int32)
            untrimmedNormality = 100*stats.normaltest(columnData).pvalue
            outlierStr = ''
            if not outliers:
                columnData = trimOutliers(columnData)
                normality = 100*stats.normaltest(columnData).pvalue
                outlierStr = '# Outliers: ' + str(len(self.table[column]) - len(columnData)) + '\nNormality excluding outliers: '+ str(normality) + '%\n'
            else:
                normality = untrimmedNormality
            print(column + ': [' +  ', '.join(map(str, self.table[column])) + ']\nNormality: ' + str(untrimmedNormality) + '%\n' + outlierStr)
            dataset.append(columnData)
            pStats.append(normality)
        if not outliers:
            bplot = plt.boxplot(dataset, whis=sys.maxsize, widths=0.2)
        else:
            bplot = plt.boxplot(dataset, widths=0.2)
        for patch, pStat in zip(bplot['boxes'], pStats):
            if pStat < .5: # Rejects the null hypothesis of a normal distribution; alpha = .005
                patch.set_color('red')
        return bplot

    # returns an enumTable containing the z-scores of each value in each column of an enumTable object
    def zScores(self):
        rawInput = {'enum': [], 'data': []}
        for column in self.table:
            arr = np.array(self.table[column], dtype=np.int32)
            sigma = np.std(arr,ddof=0)
            mu = np.average(arr)
            for x in arr:
                rawInput['enum'].append(column)
                zScore = (x-mu)/sigma
                rawInput['data'].append(zScore)
        return enumTable(rawInput, 'enum', 'data')

# class used by argument parser to create line breaks within the help message
class HelpFormatOverride(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)

# maps threadName fields to their corresponding application names
def appMap(threads):
    threadNames = ['Thread Group 1-1', 'Thread Group 1-2', 'Thread Group 1-3', 'Thread Group 1-4', 'Thread Group 1-5', 'Thread Group 1-6', 'Thread Group 1-7', 'Thread Group 1-8', 'Thread Group 1-9']
    appNames = ['Device Health',  'EverSmart Clean', 'EverSmart Rodent', 'EverSmart Space Indoor Foot Traffic', 'EverSmart Space Occupancy', 'IoT Monitor - IAQ', 'Merchant Guidance', 'Settings', 'Settings - EverSmart Clean']
    threadMap = dict(zip(threadNames, appNames))
    output = []
    for i in threads:
        output.append(threadMap[i])
    return output

# reads out a matrix-form CSV and returns the corresponding data as a dict where the first row contains labels for all following rows; should work with the CSV output of jmeter's "view results as table" listener
def jmeterCSV(filename):
    with open(filename, 'r') as csvfile:
        f = csv.reader(csvfile, delimiter=',')
        data = list(f)

    output = {}
    for label in data[0]:
        output.update({label : []})

    for row in data[1:]:
        for i in range(len(row)):
            output[list(output)[i]].append(row[i])

    output.update({'path' : pathsFromURL(output['URL'])})
    output.update({'appName' : appMap(output['threadName'])})
    appTable = enumTable(output, 'appName', 'Latency')
    pathTable = enumTable(output, 'path', 'Latency')
    return appTable, pathTable

# removes hostname and protocol from URL fields
def pathsFromURL(urls):
    output = []
    for i in urls:
        urlPath = urlparse(i).path
        if (urlPath.startswith('/device') and len(urlPath.split('/'))==4) or urlPath.startswith('/apps'):
            urlPath = '/'.join(urlPath.split('/')[:-1])# removes specific ID
        output.append(urlPath)
    return output

# takes two lists and returns an uninterpolated plot of their positions
def plotPositions(distA, distB):
    distA.sort()
    distB.sort()
    if len(distA) == len(distB):
        return distA, distB
    else:
        x = []
        y = []
        smaller = distA if len(distA) < len(distB) else distB
        larger = distA if len(distA) > len(distB) else distB
        ratio = (len(smaller)-1)/len(larger)
        for lIndex in range(len(larger)):
            sIndex = int(np.round(lIndex*ratio))# creates aliasing but avoids false positives
            x.append(smaller[sIndex])
            y.append(larger[lIndex])
        return x, y

# generates a quantile-quantile plot with the axes restricted to the sample size of each dataset and returns the updated matplotlib axes alongside a list of its least-squares regression's residuals
def qQPlot(distA, distB, axis):
    distA = np.array(distA)[~np.isnan(distA)]
    distB = np.array(distB)[~np.isnan(distB)]
    residual = []
    if len(distA)>0 and len(distB)>0:
        sLim = min(stats.t.ppf(q=1-.005, df=(len(distA)-2)), stats.t.ppf(q=1-.005, df=(len(distB)-2)))
        lLim = max(stats.t.ppf(q=1-.005, df=(len(distA)-2)), stats.t.ppf(q=1-.005, df=(len(distB)-2)))
        axis.set_xlim(-sLim, sLim)
        axis.set_ylim(-lLim, lLim)

        plotX, plotY = plotPositions(distA, distB)
        dataLen = len(plotX)
        axis.scatter(plotX, plotY)

        left, right = axis.get_xlim()
        xRange = np.linspace(left,right,dataLen)
        axis.plot(xRange,xRange,linestyle=(0,(5,5)),color='blue')

        concat = (plotX, np.ones(dataLen))
        coefficients = np.vstack(concat).T
        regression = np.linalg.lstsq(coefficients, plotY)[0]
        regFunc = regression[0] * xRange + regression[1]
        axis.plot(xRange,regFunc,color='red')

        for i in range(len(plotX)):
            iResid = plotY[i] - regression[0] * plotX[i] - regression[1]
            residual.append(iResid)

    return axis, residual

# iterates across all lines in all components of a matplotlib plot and shifts all of their x-positions by a given constant
def translatePlot(bPlot, offset):
    for component in list(bPlot):
        for line in bPlot[component]:
            xData = line.get_xdata()
            xShifted = []
            for x in xData:
                xShifted.append(x+offset)
            line.set_xdata(xShifted)

# takes a given list of unsigned 32-bit integers and recursively removes any outliers in a copy thereof per the modified thompson tau test, returning this copy
def trimOutliers(data):
    if len(data)==1:
        return data
    else:
        data = np.array(list(map(int, data)), dtype=np.int32)
        data.sort()
        n = len(data)
        tau = stats.t.ppf(q=1-.005, df=(n-2))
        rejection = tau*(n-1)/((n*(n-2+tau**2))**.5)
        mu = np.average(data)
        sigma = np.std(data, ddof=1)
        min_cutoff = mu-sigma*rejection
        max_cutoff = mu+sigma*rejection
        if min(data) <= min_cutoff or max(data) >= max_cutoff:
            new_data = data
            index = 0
            while new_data[index] < min_cutoff:
                index += 1
            new_data = new_data[index:]
            index = len(new_data) - 1
            while new_data[index] > max_cutoff:
                index -= 1
            new_data = new_data[:index]
            return trimOutliers(new_data)
        else:
            return data

if __name__ == "__main__":
    main()
