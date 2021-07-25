
#RFM ile K-Means Kıyaslanması (n_cluster = 10)

rfm.groupby("Cluster_No").agg({"Cluster_No": "count"})
rfm.groupby("Segment").agg({"Segment": "count"})


for i in range(1,11):
    print(f"{i}. Küme".center(50,"*"))
    print(rfm.loc[rfm["Cluster_No"] == i,"Segment"].value_counts())


#rfm'e göre aynı segmentte olan (hibernating) kişiler k-means'e göre aynı kümede değil. Örneğin hibernating segmentindeki 229 kişi 2 numaralı kümedeyken, 155 kişi 3 numaralı kümede.

scaler = MinMaxScaler(feature_range=(1,10))
rfm["recency"] = scaler.fit_transform(rfm[["recency"]])
rfm["frequency"] = scaler.fit_transform(rfm[["frequency"]])
rfm["monetary"] = scaler.fit_transform(rfm[["monetary"]])

rfm.groupby(["Cluster_No", "Segment"]).agg({"frequency": ["mean", "count"],
                                            "recency" : ["mean", "count"],
                                            "monetary": ["mean", "count"]})

#                                      Frequency        Recency          Monetary
#Cluster_No (1) : cant_loose            1.29067    12   5.20442    12    1.08344    12
#Cluster_No (4) :cant_loose            1.31989     28   3.04835    28    1.09396    28

#Öncelikle değerler arasındaki mesafeyi iyi anlayabilmek için recency, frequency ve monetary değişkenleri için standartlaştırma işlemi gerçekleştirdik.

#Şimdi, yukarıdaki her iki küme için tüm metriklere baktık. Frequency ve monetary değerleri arasında bir fark olmadığını gördük. Ardından recency değeri için inceleme yaptığımızda; aralarında bir farklılık olduğunu gözlemledik ve bu farklılığı da k-means'in uzaklık temelli olmasına ve bu iki segmentte bulunan kişilerin recency değerleri arasında bir uzaklık olmasına dayandırdık. Yani RFM'in aynı segmente koyduğu kişileri, K-Means uzaklık temelli olduğundan recency dolayısı ile aynı kümeye koymuyor. Buradan çıkardığımız yorum ise bu modelde recency değerinin ayırt edici olmasıdır.

#Peki bu iki recency değeri arasında gerçekten de istatistiksel olarak anlamlı bir fark var mı ;)) (Bonus)

test_stat, pvalue = shapiro(rfm.loc[rfm["Cluster_No"] == 1, "recency"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))  # H0 RED, dağılım normal değil

test_stat, pvalue = shapiro(rfm.loc[rfm["Cluster_No"] == 4, "recency"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) # H0 RED, dağılım normal değil

#Nonparametrik AB testi:
test_stat, pvalue = mannwhitneyu(rfm.loc[rfm["Cluster_No"] == 1, "recency"],
                                 rfm.loc[rfm["Cluster_No"] == 4, "recency"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) #H0 red, iki kümenin cant_lose segmentindeki kişilerin recency değerleri arasında gerçekten de istatistiksel olarak anlamlı bir fark vardır.
##################################################################