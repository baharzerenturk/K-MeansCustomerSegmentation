import os
import pickle
import argparse # argparse kütüphanesini yükledik
import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from yellowbrick.cluster import KElbowVisualizer
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal


# Parser nesnesini oluşturduk.
parser = argparse.ArgumentParser(prog="K_means ile RFM Analizi", # dosyanın adı
                                 description="RFM analizi ile yapılan müşteri segmentasyonu, k_means ile yapılmıştır.") # dosya açıklaması

parser.add_argument("--datapath", # argüman ismi
                    help="Veri Seti Yolu", # argüman açıklaması
                    required=True, # argüman ismi yazılmalı mı: Evet
                    type=os.path.abspath) # argümanın tipi: os kütüphanesinin path modülündeki abspath sınıfı
parser.add_argument("--isdump", # argüman ismi
                    help="İşlenmiş veriyi diske basar.", # argüman açıklaması
                    type=str, # argümanın tipi: string
                    required=False, # argüman ismi yazılmalı mı: Hayır
                    choices=["pickle_yes", "pickle_no"], # argüman seçeneği
                    default="pickle_no") # varsayılan argüman değeri

args = parser.parse_args() # alınan argümanları ayrı ayrı kullanmamızı sağlar.

is_pickle = False # varsayılan olarak "veri seti pkl dosyasına basılmasın" dedik (Bayrak Değişkeni)
if str(args.isdump).lower() == "pickle_yes":
    is_pickle = True # eğer cmd'de argüman pickle_yes olarak girildiyse bayrak değişkenini True yap dedik

def knn_kmeans(dataframe):
    dataframe.dropna(inplace=True)
    dataframe.shape  # gözlem sayısı 541910den 406830a düştü.

    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]

    today_date = dt.datetime(2011, 12, 11)
    dataframe["Customer ID"] = dataframe["Customer ID"].astype(int)

    rfm = dataframe.groupby("Customer ID").agg({"InvoiceDate": lambda x: (today_date - x.max()).days,
                                                "Invoice": lambda x: x.nunique(),
                                                "TotalPrice": lambda x: x.sum()})

    rfm.columns = ["recency", "frequency", "monetary"]

    rfm = rfm[rfm["monetary"] > 0]

    # RFM ile
    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])

    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

    rfm["RFM_Score"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm["Segment"] = rfm["RFM_Score"].replace(seg_map, regex=True)

    # K-Means ile

    df_kmeans = pd.DataFrame()
    df_kmeans["Recency"] = rfm["recency"]
    df_kmeans["Frequeny"] = rfm["frequency"]
    df_kmeans["Monetary"] = rfm["monetary"]

    sc = MinMaxScaler((0, 1))
    df_kmeans = sc.fit_transform(df_kmeans)

    kmeans = KMeans(n_clusters=10)
    k_fit = kmeans.fit(df_kmeans)  # modeli fit ediyoruz.
    kumeler = k_fit.labels_

    k_fit.get_params()  # parametreleri getirir.

    k_fit.n_clusters  # cluster sayısı
    k_fit.cluster_centers_  # bu clusterların merkezleri (8 farklı kümenin merkezi yani 8 farklı gözlem birimi)
    k_fit.labels_  # tüm gözlem birimlerinin 8 adet sınıfa dağılımı (0dan 7ye kadar = 8 sınıf)
    k_fit.inertia_  # 8.161555920215077

    # Final Cluster'ının Oluşturulması

    df_kmeans = pd.DataFrame()
    df_kmeans["Recency"] = rfm["recency"]
    df_kmeans["Frequeny"] = rfm["frequency"]
    df_kmeans["Monetary"] = rfm["monetary"]

    pd.DataFrame({"Customer": df_kmeans.index, "Cluster": kumeler})  # bir dataframe oluşturduk.

    rfm["Cluster_No"] = kumeler
    rfm["Cluster_No"] = rfm["Cluster_No"] + 1  # labelların sıfırdan başlıyor olmaması için yaptık.
    rfm.head()
    return rfm

df = pd.read_excel(args.datapath) # argüman olarak alınan veri setinin yolu. (Absolute Path olarak girilmeli)

df_prep = knn_kmeans(df)

if is_pickle: # Bayrak değişkenimiz True ise
    filename= "knn_kmeans.pkl" # dosya ismini oluştur
    pickle.dump(df_prep, open(filename, "wb")) # ön işlemden geçmiş veri setini dosayaya bas


print(df_prep.head())
