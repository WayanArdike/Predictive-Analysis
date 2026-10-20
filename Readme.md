# Laporan Proyek Machine Learning - Wayan Ardike

## Kesehatan (Heart Attack)
![Hasil Akurasi Model](https://storage.googleapis.com/kaggle-datasets-images/7297699/11631410/36769dd44ac446c2b03fa30732466480/dataset-cover.jpg?t=2025-04-30-22-09-22)

Serangan jantung merupakan salah satu penyebab kematian utama di dunia, dengan lebih dari 17 juta kematian setiap tahunnya (WHO, 2021). Deteksi dini sangat penting untuk meningkatkan peluang keselamatan pasien. Namun, keterbatasan sumber daya di banyak rumah sakit sering kali menghambat diagnosis cepat.

Dengan memanfaatkan dataset dari Zheen Hospital, Erbil, yang mencakup indikator medis seperti usia, tekanan darah, denyut jantung, dan biomarker jantung, proyek ini bertujuan mengembangkan model klasifikasi untuk memprediksi risiko serangan jantung. Model ini diharapkan dapat membantu tenaga medis mengambil keputusan lebih cepat dan akurat melalui teknologi berbasis data.

Masalah ini harus diselesaikan karena deteksi dini memiliki dampak besar dalam menurunkan angka kematian dan biaya pengobatan. Teknologi machine learning memungkinkan pengolahan data medis secara efisien untuk membangun model klasifikasi yang dapat mendeteksi kemungkinan serangan jantung dengan tingkat akurasi tinggi.

Penggunaan teknik data mining dan machine learning, seperti Decision Tree dan Naive Bayes, telah terbukti efektif dalam memprediksi penyakit jantung dengan akurasi lebih dari 85% (Soni et al., 2011). Dengan otomatisasi diagnosis menggunakan model ini, proses pengambilan keputusan medis bisa menjadi lebih cepat dan efisien, membantu tenaga medis dalam memberikan perawatan yang tepat waktu dan meningkatkan kualitas layanan kesehatan.

Sumber :

-   World Health Organization. (2021). Cardiovascular diseases (CVDs). Retrieved from [https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds) "null")
    
-   Soni, J., Ansari, U., Sharma, D., & Soni, S. (2011). Predictive Data Mining for Medical Diagnosis: An Overview of Heart Disease Prediction. International Journal of Computer Applications, 17(8), 44-50.[https://www.academia.edu/download/79534142/5a18f6653b56138cd5196d20e2f39de189e3.pdf](https://www.academia.edu/download/79534142/5a18f6653b56138cd5196d20e2f39de189e3.pdf "null")
    

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:

-   Bagaimana cara mengidentifikasi pasien yang berisiko mengalami serangan jantung secara dini berdasarkan data medis yang tersedia?
    
-   Apakah model klasifikasi berbasis data dapat memberikan hasil diagnosis yang akurat dan membantu proses pengambilan keputusan medis?
    

### Goals

Menjelaskan tujuan dari pernyataan masalah:

-   Mengembangkan model klasifikasi untuk memprediksi kemungkinan serangan jantung pada pasien berdasarkan atribut medis seperti usia, jenis kelamin, tekanan darah, denyut jantung, kadar gula darah, dan biomarker jantung.
    
-   Mengevaluasi performa model dalam hal akurasi.
    

### Solution Statement

Menjelaskan cara untuk meraih goals

-   Menggunakan lebih dari 2 algoritma dalam pembuatan model untuk melakukan klasifikasi yaitu 'DecisionTree', 'RandomForest', 'GradientBoosting', 'KNN', 'LogisticRegression'.
    
-   Melakukan evaluasi dengan membandingkan hasil akurasi dari beberapa algoritma yang digunakan dan memilih yang terbaik.
    

## Data Understanding

Dataset Heart Attack diunduh dari [Kaggle](https://www.kaggle.com/datasets/fatemehmohammadinia/heart-attack-dataset-tarik-a-rashid "null") dimana datasets ini dikumpulkan dari Zheen Hospital di Erbil, Iraq, antara Januari hingga Mei 2019, dan bertujuan untuk mengklasifikasikan apakah seorang pasien mengalami serangan jantung atau tidak berdasarkan sejumlah parameter medis. Dataset ini terdiri dari sembilan atribut, termasuk usia, jenis kelamin, denyut jantung, tekanan darah sistolik dan diastolik, kadar gula darah, serta dua biomarker jantung yaitu CK-MB dan troponin.

### Data Loading

Melakukan loading/ menampilkan data yang akan digunakan, dimana datasets yang dipakai bernama `Medicaldataset.csv`. Setelah data dimuat, dapat diketahui bahwa dataset ini memiliki **1.319 baris (records)** dan **9 kolom (fitur)**.

### Exploratory Data Analysis

#### Deskripsi Variabel

Berdasarkan informasi dari [Kaggle](https://www.kaggle.com/datasets/fatemehmohammadinia/heart-attack-dataset-tarik-a-rashid "null"), variabel-variabel pada Heart Attack Dataset adalah sebagai berikut:

-   **Age**: Usia pasien. Bertipe data `integer`. Rentang usia pasien berkisar antara **14 hingga 103 tahun**, dengan usia rata-rata sekitar **56 tahun**. Sebanyak 50% pasien berusia 58 tahun ke bawah.
    
-   **Gender**: Jenis kelamin biologis pasien (laki-laki diberi nilai 1 dan perempuan 0). Bertipe data `integer`.
    
-   **Heart Rate**: Jumlah detak jantung per menit. Bertipe data `integer`. Rata-rata detak jantung adalah **75 bpm**, namun nilai maksimum mencapai **1111 bpm**, yang sangat tidak wajar dan kemungkinan merupakan outlier atau kesalahan data.
    
-   **Systolic Blood Pressure**: Tekanan pada arteri saat jantung berkontraksi. Bertipe data `integer`. Rata-rata tekanan darah sistolik sekitar **127 mmHg**, dengan rentang **42–223 mmHg**.
    
-   **Diastolic Blood Pressure**: Tekanan pada arteri di antara detak jantung. Bertipe data `integer`. Rata-rata tekanan darah diastolik sekitar **72 mmHg**, dengan rentang **38–154 mmHg**.
    
-   **Blood Sugar**: Tingkat glukosa darah pasien. Bertipe data `float`. Rata-rata kadar glukosa darah adalah **146.6 mg/dL**, yang berarti sebagian besar pasien memiliki kadar gula darah tinggi (lebih dari ambang normal 120 mg/dL). Nilai maksimum mencapai **541 mg/dL** yang sangat tinggi.
    
-   **CK-MB**: Enzim jantung yang dilepaskan saat terjadi kerusakan otot jantung. Bertipe data `float`. Rata-rata nilai CK-MB adalah **15.27**, dengan maksimum **300**, menunjukkan bahwa beberapa pasien mengalami kerusakan otot jantung yang signifikan.
    
-   **Troponin**: Penanda protein yang sangat spesifik untuk cedera otot jantung. Bertipe data `float`. Rata-rata nilai troponin adalah **0.36**, dengan maksimum **10.3**. Troponin biasanya sangat rendah pada individu sehat sehingga nilai tinggi mencerminkan adanya cedera jantung yang serius.
    
-   **Result**: Label hasil yang menunjukkan apakah pasien mengalami serangan jantung atau tidak (1: serangan jantung, 0: tidak ada serangan jantung). Bertipe data `object` (akan diubah menjadi numerik jika diperlukan untuk pemodelan).
    

Dari hasil `data.info()`, terlihat bahwa:

-   Terdapat 1 kolom bertipe `object` yaitu kolom `Result`. Kolom ini merupakan fitur kategorikal dan akan menjadi target prediksi model klasifikasi.
    
-   Terdapat 5 kolom bertipe `integer` yaitu kolom `Age`, `Gender`, `Heart Rate`, `Systolic Blood Pressure`, dan `Diastolic Blood Pressure`. Ini adalah fitur numerik.
    
-   Terdapat 3 kolom bertipe `float` yaitu kolom `Blood Sugar`, `Ck-mb`, dan `Troponin`. Ini juga merupakan fitur numerik.
    

Secara umum, tipe data pada setiap kolom sudah sesuai dengan jenis informasi yang disimpannya.

#### Kondisi/ Rangkuman Data

-   Usia pasien berkisar antara 14 hingga 103 tahun, dengan usia rata-rata sekitar 56 tahun.
    
-   Sebanyak 50% pasien berusia 58 tahun ke bawah, menunjukkan mayoritas pasien berusia setengah baya ke atas.
    
-   Nilai maksimum Heart Rate 1111 bpm sangat tidak wajar dan kemungkinan merupakan outlier atau kesalahan data.
    
-   Systolic Blood Pressure rata-rata sekitar 127 mmHg, dengan rentang 42–223 mmHg & Diastolic Blood Pressure rata-rata sekitar 72 mmHg, dengan rentang 38–154 mmHg.
    
-   Rata-rata kadar glukosa darah adalah 146.6 mg/dL, yang berarti sebagian besar pasien memiliki kadar gula darah tinggi (lebih dari ambang normal 120 mg/dL).
    
-   Nilai maksimum Blood Sugar mencapai 541 mg/dL yang sangat tinggi dan menunjukkan potensi kondisi kesehatan yang sangat buruk.
    
-   Rata-rata nilai CK-MB adalah 15.27, dengan maksimum 300, menunjukkan bahwa beberapa pasien mengalami kerusakan otot jantung yang signifikan.
    
-   Rata-rata nilai troponin adalah 0.36, dengan maksimum 10.3. Troponin biasanya sangat rendah pada individu sehat sehingga nilai tinggi mencerminkan adanya cedera jantung yang serius.
    

### Memeriksa kebersihan data

**Data hilang** 
Dengan menjalankan kode `data.isnull().sum()`, dapat diketahui bahwa **tidak terdapat data yang kosong atau hilang** di seluruh kolom dataset. Ini menunjukkan kualitas data yang baik dari segi kelengkapan.

**Data duplikat** 
Dengan menjalankan kode `data.duplicated().sum()`, dapat diketahui bahwa **tidak terdapat data yang duplikat atau sama** dalam dataset. Setiap baris data unik, yang penting untuk memastikan integritas analisis.

**Data outlier** 
Berdasarkan visualisasi yang telah dibuat, dapat dilihat bahwa terdapat beberapa fitur yang memiliki data _outlier_. Setelah dihitung, jumlah _outlier_ mencapai **530 baris**, yang merupakan sekitar **45% dari keseluruhan data**. Meskipun proporsi _outlier_ ini signifikan, diputuskan untuk tetap mempertahankan data _outlier_ tersebut dengan mempertimbangkan beberapa faktor krusial:

-   **Kondisi Klinis Ekstrem yang Valid**: Dalam konteks data medis, nilai-nilai ekstrem (outlier) seringkali mencerminkan kondisi klinis yang memang nyata dan penting, seperti kasus pasien dengan detak jantung sangat tinggi, tekanan darah abnormal, atau kadar biomarker jantung yang sangat elevated. Menghapus data ini dapat menghilangkan informasi vital tentang kasus-kasus berisiko tinggi.
    
-   **Informasi Penting**: _Outlier_ bisa jadi bukan kesalahan data, melainkan representasi dari variasi biologis alami atau kondisi penyakit yang parah. Menghapusnya berisiko menghilangkan pola atau hubungan yang mungkin krusial untuk deteksi dini atau pemahaman penyakit.
    
-   **Proporsi Besar**: Dengan 45% data teridentifikasi sebagai _outlier_, penghapusan akan menyebabkan hilangnya sebagian besar dataset, yang dapat bias dan mengurangi kemampuan model untuk menggeneralisasi.
    
-   **Model yang Robust**: Algoritma berbasis pohon (seperti Decision Tree, Random Forest, dan Gradient Boosting) cenderung lebih robust terhadap _outlier_ dibandingkan model berbasis jarak atau regresi linier. Mereka tidak terlalu terpengaruh oleh nilai ekstrem karena keputusan split didasarkan pada ambang batas fitur, bukan pada nilai rata-rata atau jarak absolut.
    
-   **Deteksi Kasus Berisiko Tinggi**: _Outlier_ dalam fitur seperti CK-MB dan Troponin (biomarker jantung) justru sangat relevan untuk mendeteksi kasus serangan jantung yang serius. Mempertahankan mereka dapat membantu model mengidentifikasi pasien dengan risiko tertinggi.
    
-   **Generalisasi Model**: Untuk memastikan model dapat bekerja dengan baik di dunia nyata yang penuh variasi, penting bagi model untuk "melihat" dan belajar dari berbagai kondisi, termasuk yang ekstrem. Penghapusan _outlier_ dapat menurunkan kemampuan generalisasi model.
    

Oleh karena itu, alih-alih menghapus, fokus akan diberikan pada penggunaan model yang lebih toleran terhadap _outlier_ atau mempertimbangkan transformasi data jika diperlukan di tahap selanjutnya.

### Univariate Analysis

**Categorical** 
Dari analisis yang dilakukan pada kolom `Result` diketahui bahwa data pada kolom `Result` memiliki jumlah yang hampir seimbang antara kelas positif (serangan jantung) dan negatif (tidak ada serangan jantung). Distribusi yang seimbang ini sangat baik untuk pelatihan model klasifikasi, karena mengurangi risiko model bias terhadap salah satu kelas.

**Numerical** 
Dari analisis yang dilakukan terhadap beberapa kolom numerik dapat diketahui:

-   **Age**: Distribusi cenderung normal, dengan mayoritas pasien berusia antara 50–70 tahun.
    
-   **Gender**: Terdapat lebih banyak pasien berjenis kelamin pria (nilai 1) dibanding wanita (nilai 0) dalam dataset.
    
-   **Heart Rate**: Distribusi menunjukkan adanya banyak _outlier_ ekstrem dengan nilai detak jantung di atas 1000 bpm, yang kemungkinan besar merupakan _error_ dalam pengumpulan data.
    
-   **Systolic BP**: Distribusi hampir normal, namun terdapat sebagian pasien dengan tekanan darah sistolik yang sangat tinggi (di atas 200 mmHg).
    
-   **Diastolic BP**: Distribusi cenderung _skewed_ ke kanan, dengan mayoritas nilai tekanan darah diastolik berada di rentang 60–90 mmHg.
    
-   **Blood Sugar**: Distribusi sangat _skewed_ ke kanan, mengindikasikan banyak pasien memiliki kadar gula darah yang tinggi, dengan beberapa nilai ekstrem di atas 300 mg/dL.
    
-   **CK-MB**: Distribusi sangat _skewed_ ke kanan secara ekstrem, menunjukkan bahwa sebagian besar pasien memiliki kadar CK-MB yang rendah, tetapi ada beberapa kasus dengan kadar yang sangat tinggi, yang merupakan indikasi kuat adanya kerusakan otot jantung.
    
-   **Troponin**: Mirip dengan CK-MB, banyak nilai troponin yang nol atau sangat rendah, namun ada beberapa nilai ekstrem di atas 10, yang merupakan penanda klinis serius untuk cedera jantung.
    

### Multivariate Analysis

**Categorical** 
Dari analisis data kategorikal yaitu kolom `Result` dengan beberapa kolom numerik diketahui bahwa:

-   **Age**: Pasien yang didiagnosis positif serangan jantung cenderung memiliki usia rata-rata yang lebih tua dibandingkan pasien yang didiagnosis negatif. Ini mengindikasikan bahwa usia merupakan faktor risiko yang relevan.
    
-   **Heart Rate, Systolic & Diastolic BP**: Tidak ada perbedaan signifikan pada rata-rata nilai detak jantung, tekanan darah sistolik, dan diastolik antara pasien dengan hasil positif dan negatif. Hal ini menunjukkan bahwa fitur-fitur ini mungkin tidak menjadi prediktor tunggal yang kuat untuk serangan jantung dalam dataset ini.
    
-   **Blood Sugar**: Rata-rata kadar gula darah sedikit lebih rendah pada pasien positif, namun perbedaannya sangat kecil dan mungkin tidak signifikan secara klinis.
    
-   **CK-MB**: Rata-rata nilai CK-MB jauh lebih tinggi pada pasien dengan hasil positif serangan jantung. Ini merupakan indikasi kuat adanya kerusakan otot jantung yang menjadi penanda utama serangan jantung.
    
-   **Troponin**: Rata-rata nilai troponin juga jauh lebih tinggi pada pasien positif serangan jantung. Troponin adalah biomarker yang sangat spesifik dan relevan untuk diagnosis cedera otot jantung, sehingga peningkatan signifikan ini sangat penting.
    

**Numerical** 
Dari analisis korelasi antar kolom numerik dapat diketahui bahwa:

-   **Systolic vs Diastolic Blood Pressure**: Terlihat korelasi positif yang cukup kuat (nilai korelasi sekitar 0.59). Ini berarti tekanan darah sistolik dan diastolik cenderung naik dan turun bersamaan, yang merupakan hubungan yang diharapkan secara fisiologis.
    
-   **CK-MB vs Troponin**: Terlihat korelasi positif, meskipun tidak terlalu kuat. Ini mengindikasikan bahwa peningkatan pada salah satu biomarker ini seringkali disertai dengan peningkatan pada yang lain, karena keduanya adalah penanda masalah jantung.
    
-   **CK-MB dan Troponin dengan variabel lain**: Kedua biomarker ini tidak menunjukkan hubungan linear yang kuat dengan variabel lain seperti `Age`, `Heart Rate`, atau `Blood Sugar`. Hal ini mengindikasikan bahwa CK-MB dan Troponin lebih spesifik sebagai penanda klinis langsung untuk kerusakan jantung, dan tidak terlalu berkorelasi dengan faktor-faktor risiko umum lainnya.
    
-   **Mayoritas Fitur Numerik**: Sebagian besar fitur numerik lainnya memiliki korelasi yang sangat rendah atau hampir nol satu sama lain. Ini menunjukkan bahwa fitur-fitur ini bersifat independen secara linear dan dapat memberikan informasi yang unik dalam analisis atau pemodelan, tanpa masalah multikolinearitas yang signifikan.
    

## Data Preparation

### Encoding data kolom gender

```
encoder = OneHotEncoder(sparse_output=False, feature_name_combiner='concat') 
encoded = encoder.fit_transform(data[['Gender']])

encoded = pd.DataFrame(encoded, columns=['Woman', 'Man'])

```

Melakukan encoding pada kolom `Gender` karena pada dasarnya `Gender` merupakan tipe data nominal (kategorikal tanpa urutan). Oleh karena itu, perlu dilakukan encoding dengan metode One-Hot Encoding. Metode ini akan membuat dua kolom baru (`Woman` dan `Man`), di mana setiap baris akan memiliki nilai 1 di salah satu kolom dan 0 di kolom lainnya, merepresentasikan jenis kelamin pasien tanpa memberikan urutan atau nilai numerik yang salah.

### Membagi data

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)

```

Data dibagi menjadi data latih (`X_train`, `y_train`) dan data uji (`X_test`, `y_test`) terlebih dahulu sebelum proses standarisasi. Hal ini dilakukan untuk mencegah _data leakage_ (kebocoran data), di mana informasi dari data uji "bocor" ke data latih selama proses pra-pemrosesan. Pembagian dilakukan menggunakan fungsi `train_test_split` dari modul `sklearn`, dengan ukuran data uji sebesar **30%** dari keseluruhan data dan data latih sebesar **70%**. Parameter `random_state = 123` digunakan untuk memastikan hasil pembagian data yang konsisten setiap kali kode dijalankan.

### Standarisasi

```
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)

```

Standarisasi dilakukan untuk menyamakan skala nilai pada dataset. Sebelumnya, terlihat bahwa setiap kolom/fitur memiliki rentang nilai yang sangat berbeda (misalnya, `Age` dalam puluhan, `Heart Rate` bisa mencapai ratusan, `Troponin` dalam desimal). Dengan standarisasi menggunakan `StandardScaler()` dari modul `sklearn`, nilai-nilai pada semua fitur akan ditransformasi sehingga memiliki rata-rata 0 dan standar deviasi 1. Proses `fit()` hanya dilakukan pada data latih (`X_train`) untuk mempelajari parameter scaling (rata-rata dan standar deviasi), dan kemudian transformasi (`transform()`) diterapkan pada `X_train` dan nantinya pada `X_test` menggunakan parameter yang sama. Ini penting agar model tidak bias terhadap fitur dengan skala nilai yang lebih besar dan untuk memastikan algoritma berbasis jarak atau gradien bekerja secara optimal.

## Modeling

```
models = pd.DataFrame(index=['train_acc', 'test_acc'],
                      columns=['DecisionTree', 'RandomForest', 'GradientBoosting', 'KNN', 'LogisticRegression'])

```

Sebelum melakukan proses pemodelan, sebuah tabel (`DataFrame`) bernama `models` dibuat. Tabel ini berfungsi untuk menampung dan membandingkan hasil akurasi pelatihan (`train_acc`) dan akurasi pengujian (`test_acc`) dari setiap algoritma klasifikasi yang akan digunakan. Ini mempermudah evaluasi komparatif.

### Decision Tree

```
DC = DecisionTreeClassifier(
    random_state=123,
    criterion='gini',         
    max_depth=25,             
    min_samples_split=8,      
    min_samples_leaf=5,       
    max_features='sqrt',      
)

```

**Cara Kerja Algoritma:** 
Decision Tree adalah algoritma klasifikasi yang membangun model dalam bentuk struktur pohon. Setiap "node" dalam pohon merepresentasikan sebuah fitur (atribut), setiap "cabang" merepresentasikan aturan keputusan, dan setiap "daun" (leaf node) merepresentasikan hasil klasifikasi. Algoritma ini bekerja dengan membagi dataset secara rekursif menjadi subset-subset yang lebih homogen berdasarkan nilai fitur, dengan tujuan memaksimalkan kemurnian kelas di setiap daun. Proses pembagian ini terus berlanjut hingga kondisi berhenti terpenuhi (misalnya, kedalaman maksimum pohon tercapai atau jumlah sampel di node terlalu kecil).

**Tahapan dan Parameter:**

-   **`random_state=123`**: Digunakan untuk mengontrol keacakan proses pembuatan pohon, sehingga hasil yang diperoleh akan konsisten dan dapat direproduksi setiap kali model dijalankan ulang dengan parameter yang sama.
    
-   **`criterion='gini'`**: Menentukan fungsi untuk mengukur kualitas "split" atau pemisahan pada setiap node. Indeks Gini mengukur seberapa sering elemen yang dipilih secara acak dari dataset akan salah diklasifikasikan jika diberi label secara acak sesuai dengan distribusi label di subset tersebut.
    
-   **`max_depth=25`**: Membatasi kedalaman maksimum pohon hingga 25 level. Pembatasan ini penting untuk mencegah _overfitting_, di mana pohon menjadi terlalu kompleks dan "menghafal" data latih, sehingga performanya buruk pada data baru.
    
-   **`min_samples_split=8`**: Menentukan jumlah minimum sampel yang diperlukan untuk membagi sebuah node. Jika jumlah sampel di sebuah node kurang dari 8, node tersebut tidak akan dibagi lebih lanjut, dan akan menjadi daun. Ini juga membantu mengontrol kompleksitas pohon dan mencegah _overfitting_.
    
-   **`min_samples_leaf=5`**: Menentukan jumlah minimum sampel yang harus ada di setiap node daun (leaf node) setelah pemisahan. Jika pemisahan akan menghasilkan daun dengan jumlah sampel kurang dari 5, pemisahan tersebut tidak akan dilakukan. Ini memastikan setiap daun memiliki representasi data yang cukup.
    
-   **`max_features='sqrt'`**: Menentukan jumlah fitur yang akan dipertimbangkan saat mencari pemisahan terbaik pada setiap node. `sqrt` berarti algoritma hanya akan mempertimbangkan akar kuadrat dari total jumlah fitur yang tersedia. Ini memperkenalkan elemen keacakan dan membantu mengurangi korelasi antar pohon jika digunakan dalam ensemble seperti Random Forest.
    

**Kelebihan & Kekurangan:**

-   **Kelebihan**: Mudah diinterpretasikan dan divisualisasikan, dapat menangani data numerik dan kategorikal, serta tidak memerlukan penskalaan fitur. Dapat digunakan untuk regresi maupun klasifikasi.
    
-   **Kekurangan**: Rentan terhadap _overfitting_ pada data pelatihan, terutama jika pohon menjadi terlalu dalam dan kompleks. Sedikit perubahan pada data dapat menghasilkan pohon yang sangat berbeda.
    

### Random Forest

```
RF = RandomForestClassifier(
    random_state=123,
    criterion='gini',         
    max_depth=25,             
    min_samples_split=8,      
    min_samples_leaf=5,       
    max_features='sqrt')

```

**Cara Kerja Algoritma:** 
Random Forest adalah metode _ensemble learning_ yang membangun banyak Decision Tree (`n_estimators`) secara independen dan kemudian menggabungkan prediksi mereka untuk menghasilkan hasil akhir. Setiap pohon dilatih pada subset data yang berbeda (dengan _bootstrapping_) dan hanya mempertimbangkan subset fitur acak (`max_features`) pada setiap split. Dengan menggabungkan banyak pohon yang bervariasi, Random Forest mengurangi varians dan _overfitting_ yang menjadi kelemahan Decision Tree tunggal. Untuk klasifikasi, prediksi akhir adalah hasil voting mayoritas dari semua pohon.

**Tahapan dan Parameter:**

-   **`random_state=123`**: Menjamin hasil yang dapat direproduksi (acak tetap konsisten) untuk proses pemilihan sampel dan fitur.
    
-   **`criterion='gini'`**: Mengukur kualitas split berdasarkan indeks Gini, sama seperti Decision Tree.
    
-   **`max_depth=25`**: Batas kedalaman pohon individu agar tidak terlalu kompleks. Meskipun Random Forest lebih tahan _overfitting_, membatasi kedalaman masih dapat membantu efisiensi komputasi.
    
-   **`min_samples_split=8`**: Minimum jumlah data untuk membagi node.
    
-   **`min_samples_leaf=5`**: Minimum jumlah data di setiap daun.
    
-   **`max_features='sqrt'`**: Fitur yang dipertimbangkan per split adalah akar jumlah total fitur. Ini adalah aspek kunci "randomness" dalam Random Forest, memastikan setiap pohon dalam hutan memiliki variasi dan tidak terlalu berkorelasi.
    

**Kelebihan & Kekurangan:**

-   **Kelebihan**: Secara signifikan mengurangi _overfitting_ dibandingkan Decision Tree tunggal, memberikan akurasi yang tinggi, dan relatif robust terhadap _outlier_ dan _missing value_. Dapat menangani data berdimensi tinggi.
    
-   **Kekurangan**: Model lebih besar dan lambat saat prediksi karena melibatkan banyak pohon. Kurang mudah diinterpretasikan dibandingkan Decision Tree tunggal karena sifatnya yang merupakan "kotak hitam" (black box).
    

### Gradient Boosting

```
GB = GradientBoostingClassifier(
    random_state=123,
    n_estimators=100,         
    learning_rate=0.1,        
    max_depth=3,              
    min_samples_split=5,      
    min_samples_leaf=3,       
)

```

**Cara Kerja Algoritma:** 
Gradient Boosting adalah metode _ensemble learning_ lain yang juga membangun banyak Decision Tree secara sekuensial. Namun, berbeda dengan Random Forest yang membangun pohon secara independen, Gradient Boosting membangun pohon baru untuk "mengoreksi" kesalahan prediksi dari pohon sebelumnya. Setiap pohon baru dilatih untuk memprediksi _residual_ (perbedaan antara nilai sebenarnya dan prediksi kumulatif dari pohon-pohon sebelumnya). Proses ini berulang, secara bertahap meningkatkan kinerja model.

**Tahapan dan Parameter:**

-   **`random_state=123`**: Menjamin hasil yang konsisten di setiap eksekusi.
    
-   **`n_estimators=100`**: Jumlah pohon (estimators) yang dibangun secara bertahap. Semakin banyak pohon, semakin kuat modelnya, tetapi juga semakin lama waktu pelatihan dan semakin tinggi risiko _overfitting_.
    
-   **`learning_rate=0.1`**: Seberapa besar kontribusi tiap pohon terhadap hasil akhir. Nilai yang lebih kecil berarti setiap pohon membuat koreksi yang lebih kecil, membutuhkan lebih banyak pohon (`n_estimators`) tetapi dapat menghasilkan model yang lebih robust.
    
-   **`max_depth=3`**: Mengontrol kedalaman tiap pohon agar tetap sederhana. Pohon dalam Gradient Boosting biasanya dangkal (`weak learners`) karena mereka fokus pada koreksi kesalahan.
    
-   **`min_samples_split=5`**: Minimum jumlah sampel untuk membagi node.
    
-   **`min_samples_leaf=3`**: Minimum jumlah sampel di setiap daun pohon.
    

**Kelebihan & Kekurangan:**

-   **Kelebihan**: Sangat efektif dan seringkali memberikan akurasi yang sangat tinggi, terutama pada data terstruktur. Dapat menyesuaikan terhadap error model sebelumnya (boosting) secara iteratif.
    
-   **Kekurangan**: Waktu training relatif lama karena proses sekuensial. Lebih rentan terhadap _overfitting_ jika parameter tidak diatur dengan baik (terutama `n_estimators` dan `learning_rate`). Lebih kompleks dan kurang mudah diinterpretasikan dibandingkan Decision Tree atau Random Forest.
    

### KNN

```
KNN = KNeighborsClassifier(
    n_neighbors=5,          
    metric='manhattan',     
    weights='distance',     
    algorithm='auto',       
    leaf_size=30,           
)

```

**Cara Kerja Algoritma:** 
K-Nearest Neighbors (KNN) adalah algoritma klasifikasi non-parametrik yang bekerja berdasarkan prinsip "tetangga terdekat". Untuk mengklasifikasikan sebuah titik data baru, KNN akan mencari `k` titik data terdekat di data latih berdasarkan metrik jarak tertentu. Kelas dari titik data baru kemudian ditentukan oleh mayoritas kelas di antara `k` tetangga terdekat tersebut. Jika `weights='distance'`, tetangga yang lebih dekat akan memiliki bobot lebih besar dalam penentuan kelas.

**Tahapan dan Parameter:**

-   **`n_neighbors=5`**: Jumlah tetangga terdekat (`k`) yang dipertimbangkan dalam proses prediksi. Pemilihan nilai `k` yang tepat sangat penting dan seringkali memerlukan tuning.
    
-   **`metric='manhattan'`**: Menggunakan jarak Manhattan (L1 distance) untuk mengukur kedekatan antar data. Jarak Manhattan dihitung sebagai jumlah perbedaan absolut antara koordinat titik-titik. Alternatif lain adalah Euclidean distance (`'euclidean'`).
    
-   **`weights='distance'`**: Menentukan bagaimana bobot diberikan kepada tetangga. `'distance'` berarti tetangga yang lebih dekat akan memiliki bobot yang lebih besar dalam penentuan kelas, sedangkan `'uniform'` berarti semua tetangga memiliki bobot yang sama.
    
-   **`algorithm='auto'`**: Algoritma pencarian tetangga ditentukan secara otomatis berdasarkan jenis data. Pilihan lain termasuk `'ball_tree'`, `'kd_tree'`, atau `'brute'`.
    
-   **`leaf_size=30`**: Ukuran _leaf_ pada struktur data pencarian (jika `algorithm` bukan `'brute'`). Parameter ini memengaruhi kecepatan dan penggunaan memori, terutama untuk dataset besar.
    

**Kelebihan & Kekurangan:**

-   **Kelebihan**: Sederhana dan mudah diimplementasikan. Adaptif terhadap data non-linear. Tidak ada asumsi tentang distribusi data.
    
-   **Kekurangan**: Lambat pada prediksi untuk data besar karena perlu menghitung jarak ke semua titik data latih. Sensitif terhadap skala fitur (membutuhkan standarisasi data) dan terhadap _outlier_. Pemilihan nilai `k` yang tepat sangat krusial.
    

### Logistic Regression

```
LR = LogisticRegression(
    penalty='elasticnet', 
    solver='saga', 
    C=0.01, 
    l1_ratio=0.01
    )

```

**Cara Kerja Algoritma:** 
Logistic Regression adalah algoritma klasifikasi linier yang digunakan untuk memprediksi probabilitas suatu kejadian (misalnya, probabilitas serangan jantung) dan kemudian mengklasifikasikannya berdasarkan ambang batas probabilitas tersebut. Meskipun namanya mengandung "Regression", ini adalah model klasifikasi. Algoritma ini menggunakan fungsi logistik (sigmoid) untuk memetakan output linier ke dalam rentang antara 0 dan 1, yang dapat diinterpretasikan sebagai probabilitas. Regularisasi digunakan untuk mencegah _overfitting_ dengan menambahkan penalti pada ukuran koefisien model.

**Tahapan dan Parameter:**

-   **`penalty='elasticnet'`**: Menggabungkan penalti L1 (Lasso) dan L2 (Ridge) untuk regularisasi. Penalti L1 mendorong beberapa koefisien menjadi nol (seleksi fitur), sedangkan penalti L2 mencegah koefisien menjadi terlalu besar. `elasticnet` adalah pilihan yang baik ketika ada banyak fitur yang berkorelasi.
    
-   **`solver='saga'`**: Solver yang mendukung penalti `elasticnet`. Solver ini cocok untuk dataset besar dan ketika ada banyak fitur.
    
-   **`C=0.01`**: Kebalikan dari kekuatan regularisasi. Nilai `C` yang kecil berarti regularisasi yang lebih kuat (penalti yang lebih besar pada koefisien), yang membantu mencegah _overfitting_.
    
-   **`l1_ratio=0.01`**: Rasio campuran antara penalti L1 dan L2 ketika `penalty='elasticnet'`. Nilai `0.01` berarti penalti didominasi oleh L2 (Ridge), dengan sedikit kontribusi dari L1 (Lasso).
    

**Kelebihan & Kekurangan:**

-   **Kelebihan**: Mudah diinterpretasikan karena memberikan probabilitas dan hubungan linier antar fitur. Cepat untuk dilatih dan cocok untuk klasifikasi biner. Memberikan dasar yang kuat untuk pemodelan.
    
-   **Kekurangan**: Asumsi linearitas antara fitur dan log-odds dari variabel target. Sensitif terhadap _outlier_ dan multikolinearitas (meskipun regularisasi dapat membantu). Mungkin tidak berkinerja baik pada dataset dengan hubungan non-linier yang kompleks.
    

## Evaluation

Matrik evaluasi yang digunakan adalah akurasi, berikut hasil dari beberapa akurasi algoritma: | Model | train_acc | test_acc | | Model               
| Model               | train_acc | test_acc |
|---------------------|-----------|----------|
| DecisionTree        | 0.971831  | 0.631313 |
| RandomForest        | 0.989166  | 0.631313 |
| GradientBoosting    | 1.0       | 0.631313 |
| KNN                 | 1.0       | 0.631313 |
| LogisticRegression  | 0.67714   | 0.651515 |
 

-   **Overfitting yang Parah**: DecisionTree, RandomForest, GradientBoosting, dan KNN menunjukkan akurasi pelatihan yang sangat tinggi (≥97%, bahkan 100% untuk GradientBoosting dan KNN) tetapi akurasi pengujian yang jauh lebih rendah (sekitar 63%). Fenomena ini adalah indikasi jelas dari _overfitting_. Model-model ini terlalu "menghafal" pola dan noise pada data latih, sehingga gagal menggeneralisasi dengan baik pada data baru yang belum pernah dilihat sebelumnya. Ini berarti model-model tersebut tidak akan dapat diandalkan dalam skenario dunia nyata.
    
-   **Logistic Regression sebagai Model Terbaik (dalam konteks ini)**: Meskipun akurasi pengujiannya (65.2%) tidak secara numerik tertinggi dibandingkan akurasi pelatihan model lain, Logistic Regression menunjukkan performa yang paling stabil dan seimbang antara akurasi pelatihan (67.7%) dan akurasi pengujian. Perbedaan yang kecil antara akurasi latih dan uji menunjukkan bahwa model ini tidak mengalami _overfitting_ yang signifikan. Dalam konteks aplikasi medis, di mana keandalan dan kemampuan generalisasi model sangat penting, model yang stabil dan tidak _overfit_ lebih diutamakan daripada model dengan akurasi pelatihan yang tinggi tetapi _overfit_. Logistic Regression, dalam kasus ini, berfungsi sebagai _baseline_ yang lebih realistis dan dapat diandalkan untuk prediksi risiko serangan jantung. Model ini lebih mampu menggeneralisasi pola yang sebenarnya dari data, bukan hanya mengingat data latih.
