import matplotlib
matplotlib.use("TkAgg")
from pylab import *
import numpy as np
import scipy
import csv

import pylab
import time
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
#import statistics
import os
import glob
import re
import pandas as pd
from datetime import date
import mpu
import random

numOfSamples = 1000

'''Reads a set of IQ files with different gains, and shows the mean and std deviation of tone power at each gain'''
class Utilities:
    peakValues = None
    numOfPoints = 0
    meanDict = {}
    varianceDict = {}

    def totalpower(self, psd):
        avg_psd_dB = 10 * np.log10(np.average(psd) / 10.0)
        return avg_psd_dB

    def get_peak_pos(self, reals, imags, sample_rate, fc, NFFT):
        numFFTs = int(len(reals) / NFFT)
        maxValues = np.zeros(numFFTs)
        averageValues = np.zeros(numFFTs)
        positions = np.zeros(numFFTs)
        errors = 0

        i = 0
        iq_samples = np.array(
            [(re + 1j * co) for re, co in zip(reals[i * NFFT:(i + 1) * NFFT], imags[i * NFFT:(i + 1) * NFFT])])
        x = pylab.psd(iq_samples, NFFT=NFFT, Fs=sample_rate / 1e6, Fc=fc, window=mlab.window_hanning)

        np.set_printoptions(threshold=np.infty)

        firsthalf = 10 * np.log10(x[0][0:int(1 * len(x[0]) / 2) - 1])
        peakPoint = np.argmax(firsthalf)
        return peakPoint

    def collect_peaks(self, reals, imags, sample_rate, fc, NFFT, peakPoint):
        numFFTs = int(len(reals) / NFFT)
        maxValues = np.zeros(numFFTs)
        averageValues = np.zeros(numFFTs)
        positions = np.zeros(numFFTs)
        errors = 0
        for i in range(0, numFFTs): #len(reals)/NFFT

            iq_samples = np.array([(re + 1j * co) for re, co in zip(reals[i * NFFT:(i + 1) * NFFT], imags[i * NFFT:(i + 1) * NFFT])])
            x = pylab.psd(iq_samples, NFFT=NFFT, Fs=sample_rate / 1e6, Fc=fc, window=mlab.window_hanning)

            np.set_printoptions(threshold=np.infty)

            firsthalf = 10 * np.log10(x[0][0:int(1 * len(x[0]) / 2) - 1])
            self.peakValues[self.numOfPoints] = firsthalf[peakPoint]
            self.numOfPoints += 1

    def post_process(self, filename):
        mean = np.mean(self.peakValues[:self.numOfPoints])
        variance = np.var(self.peakValues[:self.numOfPoints])
        std = np.std(self.peakValues[:self.numOfPoints])
        self.meanDict[filename] = mean
        self.varianceDict[filename] = variance

    def __init__(self):
        self.peakValues = np.zeros(numOfSamples)

import weakref
class RTL_IQ_analysis:


    def __init__(self, iqfile, datatype, block_length, sample_rate):
        self.iqfile = iqfile
        self.datatype = datatype
        self.sizeof_data = self.datatype.nbytes  # number of bytes per sample in file
        self.block_length = block_length
        self.sample_rate = sample_rate
        self.hfile = open(self.iqfile, "rb")
        self.call_count = 0
        def on_die(killed_ref):
            print('on_die')
            self.hfile.close()
        self._del_ref = weakref.ref(self, on_die)

    def read_samples(self):
        self.hfile.seek(self.block_length * self.call_count)
        self.call_count += 1
        try:
            iq = scipy.fromfile(self.hfile, dtype=self.datatype, count=self.block_length, )
        except MemoryError:
            print("End of File")
        else:
            reals = scipy.array([(r) for index, r in enumerate(iq) if index % 2 == 0])
            imags = scipy.array([(i) for index, i in enumerate(iq) if index % 2 == 1])

        #self.hfile.close()
        return reals, imags

def convert_locations_to_filenames():
    allFiles = glob.glob("*.csv")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, index_col=None, header=0, parse_dates=['time'])
        list_.append(df)
    frame = pd.concat(list_)
    df['seconds'] = [datetime.datetime.utcfromtimestamp((long)(time.mktime(t.timetuple())))
                     - datetime.timedelta(hours=4) for t in df.time]
    df['dup_seconds'] = [t for t in df.seconds]

    base_lat = df['lat'].iloc[0]
    base_lon = df['lon'].iloc[0]
    print(base_lat, base_lon)
    df['distance'] = [mpu.haversine_distance((base_lat, base_lon),
                                             (df['lat'].iloc[rowNum], df['lon'].iloc[rowNum]))
                      for rowNum in range(len(df))]
    return df

def list_iq_files():
    listOfIQs = glob.glob("*.iq")
    #print([(filename[:-3] for filename in listOfIQs])
    timestamps = [datetime.datetime.utcfromtimestamp(long(filename[:-3])/1e9) for filename in listOfIQs]
    df1 = pd.DataFrame(list(zip(timestamps, listOfIQs)), columns=['timestamps', 'filename'])
    return df1

def merge_dfs(df1, df2):
    #B1 = df1.set_index('time').reindex(df2.set_index('timestamps').index, method='nearest').reset_index()
    df1['seconds'] = pd.to_datetime(df1.seconds)
    df2['timestamps'] = pd.to_datetime(df2.timestamps)
    df1.set_index('seconds')
    df2.set_index('timestamps')
    df1.sort_values(by='seconds', inplace=True)
    df2.sort_values(by='timestamps', inplace=True)
    df1 = df1.drop(columns=['lat', 'lon', 'elevation', 'accuracy', 'bearing', 'speed',
                      'satellites', 'provider',  'hdop', 'vdop', 'pdop', 'geoidheight',
                            'ageofdgpsdata', 'dgpsid', 'activity', 'battery', 'annotation'])
    df2.columns=['seconds', 'filename']
    df1 = pd.merge_asof(df1, df2, on='seconds', direction='nearest')

    df1 = df1.drop_duplicates(['distance'])
    df1 = df1.drop_duplicates(['filename'])
    return (df1)

def process_iq(filename, NFFT):
    datatype = scipy.uint8
    block_length = NFFT * 200
    # block_offset = NFFT*i #<---change to random offsets between 0 to (max_no_of_iq_samples - block_length)
    sample_rate = 1e6
    fc = 916e6
    utils = Utilities()

    # fullFileName = "/" + str(filename) +".iq"
    # print(fullFileName)
    rtl = RTL_IQ_analysis(filename, datatype, block_length, sample_rate)
    for j in range(1):
        r, i = rtl.read_samples()
        # print (r,i)
        peakPoint = utils.get_peak_pos(r, i, sample_rate, fc, NFFT)
        utils.collect_peaks(r, i, sample_rate, fc, NFFT, peakPoint)
        utils.post_process(filename)
        return utils.peakValues[:utils.numOfPoints]

        #utils.peakValues
import random as rand

def compute_cov(df):
    print(df['stdvalues'])
    cov = np.zeros((number_of_sensors, number_of_sensors))
    cov_file = open('cov', 'w')
    print (len(df))
    for i in range(number_of_sensors):
        for j in range(number_of_sensors):
            if (i == j):
                cov[i, j] = df['stdvalues'].iloc[random.randint(0, 10)] ** 2
                # if cov[i, j] < 1.5 * 1.5: #lower limit of std dev
                #     cov[i, j] = 1.5 * 1.5
                # elif cov[i, j] > 2.5 * 2.5: # upper limit of std dev
                #     cov[i, j] = 2.5 * 2.5
            print(cov[i, j], end=' ', file=cov_file)
        print(file=cov_file)
    return cov
rand.seed(1)

def generate_hypothesis_data():
    df1 = convert_locations_to_filenames()
    df2 = list_iq_files()
    df = merge_dfs(df1, df2)
    df = df.reset_index()
    print (df)
    mean_var_arrays = {}
    var_var_arrays = {}
    start_logNFFT = 8
    end_logNFFT = 15 #change start and end_logNFFT to same value for homogeneous sensors
    models = {}
    predictions= {}
    delmean = {}
    for logNFFT in range(start_logNFFT, end_logNFFT + 1):
        NFFT = 2 ** logNFFT
        df['meanvalues' + str(NFFT)] = [np.mean(process_iq(filename, NFFT)) for filename in df.filename]
        if (end_logNFFT == logNFFT):
            df['stdvalues'] = [np.std(process_iq(filename, NFFT)) for filename in df.filename]

        # df = df.drop(df.index[19])
        # df = df.drop(df.index[18])
        # df = df.drop(df.index[17])

        print(df)

        # try:
        X = np.log10(df['distance'] * 1000 + 0.5)
        print(X)
        # except:
        #     X = np.log2(0.0001)
        y = df['meanvalues' + str(NFFT)]
        #print (X, y)
        delta = [(df['meanvalues' + str(NFFT)][i] - df['meanvalues' + str(NFFT)][i - 1]) /
                 (X[i] - X[i - 1]) for i in range(1, len(df))]
        delta = [delvalue if delvalue > 0 else 0 for delvalue in delta ]
        positivedelvalues = [delvalue for delvalue in delta if delvalue > 0]
        print(delta, positivedelvalues)
        delmean[NFFT] = np.mean(np.array(positivedelvalues))

    cov = compute_cov(df)

    sensor_locations = random.sample(range(length * length), number_of_sensors)
    sensor_configs = [2 ** random.randrange(start_logNFFT, end_logNFFT + 1) for i in range(len(sensor_locations))]
    sensor_file = open('sensors', 'w')
    energy_cost = np.array([2.9935, 2.9799, 3.0657, 3.2532, 3.5475, 4.0937, 5.1648, 7.6977])
    energy_cost_max = np.max(energy_cost)
    energy_cost = np.array([cost / energy_cost_max for cost in energy_cost])

    for sensorNum in range(len(sensor_locations)):
        sensor = sensor_locations[sensorNum]
        yloc = sensor // length
        xloc = sensor % length
        sensor_config = sensor_configs[sensorNum]
        print(xloc, yloc, math.sqrt(cov[sensorNum, sensorNum]), energy_cost[int(np.log2(sensor_config) + 0.5)
                                                       - start_logNFFT], file=sensor_file)

    print(sensor_configs)
    file_handle = open('hypothesis', 'w')
    distance_unit = 5 #increase this to make the means smaller; affects very quickly
    for trans_i in range(0, length):
        for trans_j in range(0, length):
            for sensorNum in range(len(sensor_locations)):
                sensor = sensor_locations[sensorNum]
                yloc = sensor // length
                xloc = sensor % length
                distance = math.sqrt((xloc - trans_i)**2 + (yloc - trans_j)**2)
                if (distance == 0):
                    distance = 0.5
                transmitter_power = df['meanvalues' + str(sensor_configs[sensorNum])].max()

                actual_power = transmitter_power - delmean[256] * np.log10(distance) / distance_unit
                if (actual_power < 0):
                    actual_power = 0

                print(trans_i, trans_j, xloc, yloc, actual_power, sensor_configs[sensorNum], file=file_handle)

        #models[NFFT] = sm.OLS(y, X).fit()
        # predictions[NFFT] = models[NFFT].predict(X)
        # predictions[NFFT][0] = 10000
        # df['predictions'] = [predictions[NFFT][i] for i in range(0, len(predictions[NFFT]))]
        # print(df)
        # # # values = np.array(values)
        # means = [np.mean(values) for i in range((len(values)))]
        # cov = np.cov(values)
        # var_arrays = np.zeros(len(cov))
        # for i in range(len(cov)):
        #     var_arrays = cov[i, i]
        # mean_var_arrays[logNFFT] = np.mean(var_arrays)
        # var_var_arrays[logNFFT] = np.var(var_arrays)

# actual_data = []
# X = [arange(0, 0.5, 0.01)]
# for logNFFT in range(start_logNFFT, end_logNFFT + 1):
#     predictions = models[256].predict(X)
#     print(predictions)
#
# print(mean_var_arrays, df)
#
#

def plot_histogram(filename):
    NFFT = 512
    utils = Utilities()
    datatype = scipy.uint8
    # block_offset = NFFT*i #<---change to random offsets between 0 to (max_no_of_iq_samples - block_length)
    sample_rate = 1e6
    fc = 916e6
    utils = Utilities()
    datatype = scipy.uint8


    block_length = NFFT * 100
    # block_offset = NFFT*i #<---change to random offsets between 0 to (max_no_of_iq_samples - block_length)
    sample_rate = 1e6
    fc = 916e6
    utils = Utilities()

    # fullFileName = "/" + str(filename) +".iq"
    # print(fullFileName)
    rtl = RTL_IQ_analysis(filename, datatype, block_length, sample_rate)
    r, i = rtl.read_samples()
    # print (r,i)
    peakPoint = utils.get_peak_pos(r, i, sample_rate, fc, NFFT)
    utils.collect_peaks(r, i, sample_rate, fc, NFFT, peakPoint)
    utils.post_process(filename)
    print(utils.peakValues[:utils.numOfPoints])
    plt.close()
    plt.hist(utils.peakValues[:utils.numOfPoints], 200)

    plt.savefig(filename + '.png')

import glob

length = 64  # change number of cells
number_of_sensors = 1000  # change number of sensors

#for filename in glob.iglob('*.iq'):
#    plot_histogram(filename)

generate_hypothesis_data()



