# code

~~~python
#6.2中的（b）
import matplotlib.pyplot as plt
import numpy as np
import csv

data_x = []

with open('test1.csv', 'r', encoding='gbk') as f:
    reader = csv.reader(f)

    for i in reader:
        data_x.append(i)

x = [int(i) for i in data_x[0]]
y1 = [float(i) for i in data_x[1]]
y2 = [float(i) for i in data_x[2]]

plt.xlabel('year x')
plt.ylabel('popularity of music_genre y')

plt.figure(num=1, figsize=(8, 5))
#gca = 'get current axis'
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.plot(x, y1, color='blue', linewidth=1.5, linestyle='--', label='Pop/Rock')
plt.plot(x, y2, color='red', linewidth=1.5, linestyle='--', label='R&B')

plt.legend(loc='best')
plt.show()  #出图
~~~



~~~python
#图8.1
import matplotlib.pyplot as plt
import numpy as np
import csv

data_x = []

with open('test1.csv', 'r', encoding='gbk') as f:
    reader = csv.reader(f)

    for i in reader:
        data_x.append(i)

x = [int(i) for i in data_x[0]]
y1 = [float(i) for i in data_x[1]]
y2 = [float(i) for i in data_x[2]]
y3 = [float(i) for i in data_x[3]]

plt.xlabel('year x')
plt.ylabel('popularity of music_genre y')

plt.figure(num=1, figsize=(8, 5))
#gca = 'get current axis'
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.plot(x, y1, color='blue', linewidth=1.5, linestyle='--', label='Pop/Rock')
plt.plot(x, y2, color='red', linewidth=1.5, linestyle='--', label='R&B')
plt.plot(x, y3, color='green', linewidth=1.5, linestyle='--', label='Country')
plt.legend(loc='best')
plt.show()  #出图
~~~

~~~python
#图9.1
import matplotlib.pyplot as plt
import numpy as np
import csv

data_x = []

with open('test1.csv', 'r', encoding='gbk') as f:
    reader = csv.reader(f)

    for i in reader:
        data_x.append(i)

x = [int(i) for i in data_x[0]]
y1 = [float(i) for i in data_x[1]]

plt.xlabel('year x')
plt.ylabel('popularity of music_genre y')

plt.figure(num=1, figsize=(8, 5))
#gca = 'get current axis'
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.plot(x, y1, color='blue', linewidth=1.5, linestyle='--', label='Pop/Rock')


plt.legend(loc='best')
plt.show()  #出图
~~~

