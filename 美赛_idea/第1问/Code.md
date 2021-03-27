```python
import csv

edge_to = [0] * 200000
edge_next = [0] * 200000
head = [-1] * 200000
num_edge = 0

def add_edge(fro, to):
    #print(fro, to)
    global num_edge
    num_edge += 1
    #print(num_edge)
    edge_to[num_edge] = to
    edge_next[num_edge] = head[fro]
    head[fro] = num_edge
    
    par = [_ for _ in range(5604)]

class info:    
    def _init_(self, num, num_id, name, genre, start):
        self.num = num       
        self.id = num_id      
        self.name = name    
        self.genre = genree  
        self.start = strat    
        
data_list = [] 

with open('influence_data.csv', 'r', encoding='gbk', errors='ignore') as f:
    reader = csv.reader(f)
    # print(reader)
        
    for row in reader:
        data_list.append(row)
    
data_dic = {}   
all_genre = {}     
influencer_genre = {}  
follower_genre = {}   
all_people = []  
inn = [0] * 5604
out = [0] * 5604

lar_num_genre = []                
lar_num_genre_influencer = []          
lar_num_genre_follower = []       
filt = {'Pop/Rock': 1, 'Comedy/Spoken': 2, 'R&B;': 3}

temp = info()
all_people.append(temp)

n = [0]   

for i in range(1, len(data_list)):
    x, y = data_list[i][0], data_list[i][4]
    
    if x not in data_dic:   
        n[0] += 1
        data_dic[x] = n[0]    
        
        p = info()   
        p.num = data_dic[x]
        p.id = data_list[i][0]
        p.name = data_list[i][1]
        p.genre = data_list[i][2]
        p.start = data_list[i][3]
        all_people.append(p)
        
        if data_list[i][2] in all_genre:      
            all_genre[data_list[i][2]] += 1
        else:
            all_genre[data_list[i][2]] = 1
            
        
        
    if y not in data_dic:   
        n[0] += 1
        data_dic[y] = n[0]    
        
        q = info()
        q.num = data_dic[y]
        q.id = data_list[i][4]
        q.name = data_list[i][5]
        q.genre = data_list[i][6]
        q.start = data_list[i][7]
        all_people.append(q)
            
    
        if data_list[i][6] in all_genre:      
            all_genre[data_list[i][6]] += 1
        else:
            all_genre[data_list[i][6]] = 1
        
    if data_list[i][2] in influencer_genre:       
        influencer_genre[data_list[i][2]] += 1
    else:
        influencer_genre[data_list[i][2]] = 1

    if data_list[i][6] in follower_genre:       
        follower_genre[data_list[i][6]] += 1
    else:
        follower_genre[data_list[i][6]] = 1
            
            

    if data_list[i][2] in filt:
        year_gener1 = []
        year_gener1.append(data_list[i][3])
        year_gener1.append(data_list[i][2])
        lar_num_genre.append(year_gener1)
        lar_num_genre_influencer.append(year_gener1)
    if data_list[i][6] in filt:
        year_gener2 = []
        year_gener2.append(data_list[i][7])
        year_gener2.append(data_list[i][6])
        lar_num_genre_follower.append(year_gener2)
        lar_num_genre.append(year_gener2)
        
    u, v = data_dic[x], data_dic[y]
    add_edge(u, v)  
          
    out[u] += 1
    inn[v] += 1
    
num = n[0]  

gener_boos = {}

for i in all_genre:
    gener_boos[i] = 0

for i in range(1, num + 1):
    if inn[i] == 0:
        gener_boos[all_people[i].genre] += 1

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
mpl.rcParams['axes.unicode_minus'] = False 
plt.rcParams['figure.figsize'] = (15.0, 5.0)

gener_type = [i for i in all_genre]
all_people1 = [all_genre[i] for i in all_genre]
boos = [gener_boos[i] for i in gener_boos]


plt.title('Number of people in different genres of music')
plt.plot(gener_type, all_people1, color='red',  linestyle='--', label='Number of people')
my_x_ticks = np.arange(len(gener_type))
plt.xticks(my_x_ticks)

for i,y in enumerate(all_people1):
    #plt.text(ax, i, y, y)
    plt.text(i, y, y, ha='center', color='b', va= 'baseline',fontsize=13)

plt.legend() # 显示图例
plt.xticks(rotation=90)
plt.xlabel('Genre')
plt.ylabel('Number of people')
plt.show()


plt.title('Number of different genres of music among influencers and followers')
plt.plot(gener_type, influencer, color='red',  linestyle='--', label='influencers')
plt.plot(gener_type, follower, color='blue',  linestyle='-', label='followers')
my_x_ticks = np.arange(len(gener_type)) 
plt.xticks(my_x_ticks)

plt.legend()
plt.xticks(rotation=90)
plt.xlabel('Genre')
plt.ylabel('Number of people')
plt.show()

plt.title('Number of experts in different genres of music')
plt.plot(gener_type, boos, color='red',  linestyle='--', label='Number of people')
my_x_ticks = np.arange(len(gener_type))
plt.xticks(my_x_ticks)
for i,y in enumerate(boos):
    #plt.text(ax, i, y, y)
    plt.text(i, y, y, ha='center', color='b', va= 'baseline',fontsize=13)

plt.legend() 
plt.xticks(rotation=90)
plt.xlabel('Genre')
plt.ylabel('Number of people')
plt.show()



year = ['1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010']
Pop_Rock = [0] * 9
Comedy_Spoken = [0] * 9
R_B = [0] * 9

for i in lar_num_genre:
    index = int((int(i[0]) - 1930) / 10)    
    if i[1] == 'Pop/Rock':
        Pop_Rock[index] += 1
    elif i[1] == 'Comedy/Spoken':
        Comedy_Spoken[index] += 1
    elif i[1] == 'R&B;':
        R_B[index] += 1
        
Pop_influencer_num = [0] * 9
Pop_follower_num = [0] * 9

for i in lar_num_genre_influencer:
    index = int((int(i[0]) - 1930) / 10)    
    if i[1] == 'Pop/Rock':
        Pop_influencer_num[index] += 1

for i in lar_num_genre_follower:
    index = int((int(i[0]) - 1930) / 10)    
    if i[1] == 'Pop/Rock':
        Pop_follower_num[index] += 1


plt.title('Number of influencers and followers in different years of genre Pop/Rock')
plt.plot(year, Pop_influencer_num, color='red',  linestyle='--', label='Number of influencer')
plt.plot(year, Pop_follower_num, color='blue',  linestyle='--', label='Number of follower')
my_x_ticks = np.arange(len(year)) 
plt.xticks(my_x_ticks)

'''
for i,y in enumerate(Pop_influencer_num):
    #plt.text(ax, i, y, y)
    plt.text(i, y, y, ha='center', color='b', va= 'baseline',fontsize=13)
    
for i,y in enumerate(Pop_follower_num):
    #plt.text(ax, i, y, y)
    plt.text(i, y, y, ha='center', color='r', va= 'baseline',fontsize=13)
'''

plt.legend()
plt.xlabel('Year')
plt.ylabel('Number of people')
plt.show()

plt.title('Number of people in different years of the three genres ')
plt.plot(year, Pop_Rock, color='red',  linestyle='--', label='Pop/Rock')
plt.plot(year, Comedy_Spoken, color='blue',  linestyle=':', label='Comedy/Spoken')
plt.plot(year, R_B, color='brown',  linestyle='--', label='R&B')

my_x_ticks = np.arange(len(year))
plt.xticks(my_x_ticks)

'''
for i,y in enumerate(Pop_Rock):
    #plt.text(ax, i, y, y)
    plt.text(i, y, y, ha='center', color='r', va= 'baseline',fontsize=13)
for i,y in enumerate(R_B):
    #plt.text(ax, i, y, y)
    plt.text(i, y, y, ha='center', color='r', va= 'baseline',fontsize=13)
'''

for i,y in enumerate(Comedy_Spoken):
    plt.text(i, y, y, ha='center', color='r', va= 'baseline',fontsize=13)

plt.legend(
plt.xlabel('Genre')
plt.ylabel('Number of people')
plt.show()
```

