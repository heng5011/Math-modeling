###清洗数据

对full_music_data文件里的数据进行了过滤，以包括前1000个最频繁列出的类型中的个人喜好子类型。选定的子流派可以减少到4种流派（显示Tunes, Classical, Country, Rap/Pop）。进行此过滤以查看是否通过群集形成在数据中显示了每种流派元素的相似性。

将数据集减少到’duration_ms‘合理范围内的歌曲。从之前（〜8秒到〜1小时），减少到1-10分钟的歌曲范围。有助于确保数据来自实际的歌曲，而不是剪辑片段或其中有很多死角的“隐藏”轨道。 

检查并剔除所给influence_data文件数据中的Nan/NULL所在行，接着把influence_data文件中的influencer_main_genre列中元素通过id合并到full_music_data文件中。

箱型图显示，此列已删除了大多数outliner，但由于较长的音乐可能在群集中起重要作用，因此将保留其余异常值。比如，古典流派的歌曲的时长往往比此时的传统音乐长得多。

![](https://cdn.luogu.com.cn/upload/image_hosting/z8jcvces.png) 

用数据绘制成对图以观察分布和列之间的关系, 查看是否可以预测任何可能的群集。 

![](F:\Code\jupyter_notebook\美赛\idea\photos\pro2\relationshipsBetweenColumns.png)

- 上图表明，要研究的数据中存在很大的相关性。
- 我们还可以看到acousticness, instrumentalness, energy, and popularity 是双峰的。
- Liveness and speechiness 向左倾斜， loudness 向右倾斜

### 建立UMAP模型

使用原始数据的子集（用于模拟用户播放列表），实施了一些聚类算法，以查看他们如何对音乐数据进行聚类。选择的算法将是UMAP-Kmeans模型。在审查每个类的前十名最频繁的特征时，UMAP在类的聚类方面似乎做得更好，并且产生了最佳的轮廓分数。 

### 绘制出热力图(流派内)

---

![](F:\Code\jupyter_notebook\美赛\idea\photos\pro2\heatmap.png)

上面的热图显示，danceablility, loudness, energy and acousticness 相互关联。根据他们的描述，energy是可变的，需要考虑loudness和acousticness。故舍弃energy



![](F:\Code\jupyter_notebook\美赛\idea\photos\pro2\varianceRatio.png)

观察上图可得,大约有83％的方差可以用6个成分来证明同一类型的艺术家相似度更高，而仅有17%的不同派别的艺术家具有相似特征。

---

### 聚类 (流派间)

#### 数据降维

接下来进行数据降维。降维是进行建模中的通用惯例。大部分使用的列都是相同的比例，从0-1开始，但“popularity ”列的范围是从0至100。这里需要缩放使每列具有相等的权重。

首先，尝试通过K-Means进行一些聚类。在评估结果之后，使用已经进行了降维处理的数据重新运行k-means算法，并比较两个结果。 

![](F:\Code\jupyter_notebook\美赛\idea\photos\pro2\elbowMethod.png)



![](F:\Code\jupyter_notebook\美赛\idea\photos\pro2\UmapElbowMethod.png)



上图1和图2分别显示了每个聚类结果的欧式距离和使用UMAP精简的欧式距离。有上图可得，该数据不会产生非常急剧的弯头(说明数据均匀分布在黎曼流形上)，相反，在第二个群集之后，每个群集的SSD只会逐渐下降。将选择四个群集，这意味着SSD约为11，000。 在K-Means模型中，使用UMAP精简数据生成的4个群集的SSD值约为1000。之前的SSD仅低至6000。 

| name               | SSD   |
| ------------------ | ----- |
| 未精简数据进行群集 | 11000 |
| 4个群集            | 6000  |
| UMAP精简数据       | 1000  |

------



![](https://cdn.luogu.com.cn/upload/image_hosting/qdfrtlii.png)

由图可以明显的得到:

- 第一组的特点是更快，更响亮，跳舞能力更高且受欢迎。可能为“ R＆B /pop”。
- 第二个群集的特点是高声学，生动感和较低的混浊度。可能为“country”
- 第三类群的特征是高舞蹈能力，活跃性，响度，言语和受欢迎程度。可能为“Rap/hip-hop”。
- 最后一簇的特征是高声学，持续时间，工具性。可能为“Classical”。

### summary

把K-Means算法优化为高斯混合模型，该算法中使用欧式距离变量。将欧式距离变量绘制成图以后，可以显而易见的看出，我们所使用的的数据均匀分布在黎曼流形上; 随后进行反向缩放，找到每个聚类的平均值，然后将它们放入具有背景渐变的数据框中进行数据结果的可视化，并且定义每个聚类的某些特征。 添加一个群集标签列，对观察到的群集成员计数，使用matplotlib绘制结果。对此图来观察上面分级的聚类均值表中定义的特征。由于群集特征与使用K-Means形成的群集的特征不太相同，所以我们又使用PCA进行降维，接下来又使用UMAP得到了最佳的结果：Silhouette score for two cluster k-means: 0.513827383518219。在K-Means模型中，使用UMAP精简数据生成的4个群集的SSD值约为4000，远远小于之前的SSD7000。 

综上所述:

-  同一流派音乐的关联性比不同流派音乐的关联系数要大，这也间接的说明了处于同一流派内的两个音乐更为相似。
- 大约有83％的方差可以用6个成分来证明同一类型的艺术家相似度更高，而仅有17%的不同派别的艺术家具有相似特征。

---



## 后面几问可能用到的数据

### UMAP 轮廓分数(UMAP silhouette score )

轮廓分数将告诉我们群集相对于自身的定义程度。Solhouette测量元素到其分配的聚类中心的距离与它们到其他聚类中心的距离。分数越高，聚类越好。 

```
Silhouette score for two cluster k-means: 0.513827383518219
Silhouette score for three cluster k-means: 0.4256020486354828
Silhouette score for four cluster k-means: 0.3742857575416565
```



### UMAP流派集群

每个群集的十大最常见流派 

```
Pop/Rock      309
R&B;           71
Country        48
Jazz           39
Electronic     18
Vocal          15
Folk           15
Blues          14
Latin           7
Reggae          4
---------------------------------------------------

Pop/Rock      118
R&B;           35
Country        22
Jazz           20
Electronic     11
Vocal          11
Folk           10
Reggae          6
Latin           5
Blues           5
---------------------------------------------------

Pop/Rock      330
R&B;           82
Country        45
Jazz           37
Blues          21
Vocal          17
Electronic     16
Folk           13
Reggae         10
Latin           7
---------------------------------------------------

Pop/Rock         336
R&B;              87
Country           45
Jazz              37
Vocal             21
Electronic        16
Blues             11
Folk               9
Reggae             9
International      6

```



