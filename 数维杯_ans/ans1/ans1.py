import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import scipy.stats as stats
from scipy.stats import poisson
# from scipy.stats import gamma
from scipy.stats import nbinom
from numpy import log as ln
import statsmodels.api as sm
from datetime import datetime, timedelta
import pylab as mpl

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体字

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same') #计算移动平均值和卷积
    return y_smooth

# for state in states:
from scipy.stats import \
    gamma  # 每次都需要导入gamma

fig, ax = plt.subplots()

# 打开表格数据
g = open('000400.SZ.csv', 'r')
g1 = open('002281.SZ.csv', 'r')
g2 = open('600519.SH.csv', 'r')
# reader = csv.reader(g)
reader = csv.reader(g)
reader1 = csv.reader(g1)
reader2 = csv.reader(g2)

# 第1只股票
t_open = []
high = []
low = []
close = []
volume = []
turn = []
day = []
dates = []
ii = 0


for row in reader:
    if (ii > 0):
        date_object = datetime.strptime(row[0], '%Y-%m-%d').date()
        day.append(float(ii))
        # print(date_object,float(row[8]),float(row[7]),float(row[6]))
        dates.append(date_object)
        t_open.append(float(row[1]))
        high.append(float(row[2]))
        low.append(float(row[3]))
        close.append(float(row[4]))
        volume.append(float(row[5]))
        turn.append((float(row[6])))
    ii += 1

g.close()

ndays = []
for i in range(len(volume)):
    if (volume[i] > 0):
        ndays.append(day[i])

##### 估计和预测
dvolume = np.diff(volume)
dt_open = np.diff(t_open)
dhigh = np.diff(high)
dlow = np.diff(low)
dclose = np.diff(close)
dturn = np.diff(turn)

# print(dvolume)
for ii in range(len(dvolume)):
    if dvolume[ii] < 0.: dvolume[ii] = 0.
    if dt_open[ii] < 0. : t_open[ii] = 0.
    if dhigh[ii] < 0. : high[ii] = 0.
    if dlow[ii] < 0. : low[ii] = 0.
    if dclose[ii] < 0. : close[ii] = 0.
    # if dturn[ii] < 0. : turn[ii] = 0.

xd = dates[1:]

# 第二只股票
t_open1 = []
high1 = []
low1 = []
close1 = []
volume1 = []
turn1 = []
day1 = []
dates1 = []
ii = 0

for row in reader1:
    if (ii > 0):
        date_object1 = datetime.strptime(row[0], '%Y-%m-%d').date()
        day1.append(float(ii))
        # print(date_object,float(row[8]),float(row[7]),float(row[6]))
        dates1.append(date_object1)
        t_open1.append(float(row[1]))
        high1.append(float(row[2]))
        low1.append(float(row[3]))
        close1.append(float(row[4]))
        volume1.append(float(row[5]))
        turn1.append((float(row[6])))
    ii += 1

g1.close()

ndays1 = []
for i in range(len(volume1)):
    if (volume1[i] > 0):
        ndays1.append(day[i])

##### 估计和预测
dvolume1 = np.diff(volume1)
dt_open1 = np.diff(t_open1)
dhigh1 = np.diff(high1)
dlow1 = np.diff(low1)
dclose1 = np.diff(close1)
# dturn1 = np.diff(turn1)

# print(dvolume)
for ii in range(len(dvolume1)):
    if dvolume1[ii] < 0.: dvolume1[ii] = 0.
    if dt_open1[ii] < 0. : t_open1[ii] = 0.
    if dhigh1[ii] < 0. : high1[ii] = 0.
    if dlow1[ii] < 0. : low1[ii] = 0.
    if dclose1[ii] < 0. : close1[ii] = 0.
    # if dturn1[ii] < 0. : turn1[ii] = 0.

xd1 = dates1[1:]

# 第三只股票
t_open2 = []
high2 = []
low2 = []
close2 = []
volume2 = []
turn2 = []
day2 = []
dates2 = []
ii = 0

for row in reader2:
    if (ii > 0):
        date_object2 = datetime.strptime(row[0], '%Y-%m-%d').date()
        day2.append(float(ii))
        # print(date_object,float(row[8]),float(row[7]),float(row[6]))
        dates2.append(date_object2)
        t_open2.append(float(row[1]))
        high2.append(float(row[2]))
        low2.append(float(row[3]))
        close2.append(float(row[4]))
        volume2.append(float(row[5]))
        turn2.append((float(row[6])))
    ii += 1

g2.close()

ndays2 = []
for i in range(len(volume2)):
    if (volume2[i] > 0):
        ndays2.append(day[i])

##### 估计和预测
dvolume2 = np.diff(volume2)
dt_open2 = np.diff(t_open2)
dhigh2 = np.diff(high2)
dlow2 = np.diff(low2)
dclose2 = np.diff(close2)
dturn2 = np.diff(turn2)

# print(dvolume)
for ii in range(len(dvolume1)):
    if dvolume2[ii] < 0.: dvolume2[ii] = 0.
    if dt_open2[ii] < 0. : t_open2[ii] = 0.
    if dhigh2[ii] < 0. : high2[ii] = 0.
    if dlow2[ii] < 0. : low2[ii] = 0.
    if dclose2[ii] < 0. : close2[ii] = 0.
    # if dturn2[ii] < 0. : turn2[ii] = 0.

xd2 = dates2[1:]

# print(xd)

# plt.plot(xd, dvolume, 'go', alpha=0.5, markersize=8, label='哈哈哈')
# plt.plot(xd, dvolume)
sdays = 7
sdays1 = 30

# volume交易量
yy = smooth(dvolume,
            sdays)  # 平滑sdays(天数)移动窗口，在连续的天报告中平均大块
tmp_yy = smooth(dvolume, sdays1) #30d
yy[-2] = (dvolume[-4] + dvolume[-3] + dvolume[
    -2]) / 3.  # 最后两行数据
yy[-1] = (dvolume[-3] + dvolume[-2] + dvolume[-1]) / 3.

yy1 = smooth(dvolume1,
            sdays)  # 平滑sdays(天数)移动窗口，在连续的天报告中平均大块
tmp_yy1 = smooth(dvolume1, sdays1) #30d
yy1[-2] = (dvolume1[-4] + dvolume1[-3] + dvolume1[
    -2]) / 3.  # 最后两行数据
yy1[-1] = (dvolume1[-3] + dvolume1[-2] + dvolume1[-1]) / 3.

yy2 = smooth(dvolume2,
            sdays)  # 平滑sdays(天数)移动窗口，在连续的天报告中平均大块
tmp_yy2 = smooth(dvolume2, sdays1) #30d
yy2[-2] = (dvolume2[-4] + dvolume2[-3] + dvolume2[
    -2]) / 3.  # 最后两行数据
yy2[-1] = (dvolume2[-3] + dvolume2[-2] + dvolume2[-1]) / 3.

# volume成交量
# 1d
plt.plot(dates, volume, label = 'Stock 1')
plt.plot(dates1, volume1, label = 'Stock 2')
plt.plot(dates2, volume2, label = 'Stock 3')
plt.title('1d volume')
plt.xlabel('date')
plt.ylabel('price')
plt.legend(loc = 'best')
plt.savefig('1d')
plt.show()
plt.clf()

# 7d
# plt.plot(dates, volume, label = '1d')
plt.plot(xd, yy, 'b', label = 'Stock 1')
plt.plot(xd1, yy1, label = 'Stock 2')
plt.plot(xd2, yy2, label = 'Stock 3')
plt.title('7d volume')
plt.xlabel('date')
plt.ylabel('price')
plt.legend(loc = 'best')
plt.savefig('7d')
plt.show()
plt.clf()

# 30d
# plt.plot(dates, volume, label = '1d')
plt.plot(xd, tmp_yy, 'b', label = 'Stock 1')
plt.plot(xd1, tmp_yy1, label = 'Stock 2')
plt.plot(xd2, tmp_yy2, label = 'Stock 3')
plt.title('30d volume')
plt.xlabel('date')
plt.ylabel('price')
plt.legend(loc = 'best')
plt.savefig('30d')
plt.show()
plt.clf()

# turn周转率
# print(turn1)
# print(dates2)
plt.plot(dates, turn, 'red', linestyle = '--', label = 'Stock 1')
plt.plot(dates1, turn1, 'yellow', label = 'Stock 1')
plt.plot(dates2, turn2, 'blue', label = 'Stock 1')
plt.title('Turn of three stocks')
plt.xlabel('date')
plt.ylabel('rate')
plt.legend(loc = 'best')
plt.savefig('三只股票的周转率')
plt.show()
plt.clf()
