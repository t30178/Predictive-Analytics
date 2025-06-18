# Laporan Proyek Pertama Machine Learning Terapan - Theodorus Andang Jatmiko
---
## Domain Proyek
Proyek ini berada dalam domain kesehatan, dengan fokus utama pada deteksi dini penyakit diabetes menggunakan pendekatan machine learning. Dengan memanfaatkan data klinis pasien seperti kadar glukosa, tekanan darah, dan BMI, sistem prediksi ini bertujuan untuk membantu profesional medis dalam mengidentifikasi potensi diabetes secara lebih cepat dan efisien.

## Latar Belakang
Diabetes merupakan salah satu penyakit kronis yang banyak dialami masyarakat dunia. Banyak kasus menunjukkan bahwa penderita diabetes baru menyadari penyakitnya setelah mengalami komplikasi serius. Ini disebabkan karena kurangnya deteksi dini yang efektif.

Deteksi dini sangat penting agar pengobatan dapat diberikan lebih awal, mencegah komplikasi, dan mengurangi beban biaya kesehatan dalam jangka panjang. Oleh karena itu, pendekatan otomatis seperti machine learning dibutuhkan untuk membantu memprediksi risiko diabetes berdasarkan data klinis secara cepat dan akurat.

## Business Understanding

## Problem Statements
Berdasarkan latar belakang yang telah disampaikan, terdapat beberapa rumusan masalah yang akan diselesaikan pada proyek ini:
1. Bagaimana memanfaatkan data klinis pasien untuk memprediksi apakah seseorang berisiko menderita diabetes?
2. Model machine learning apa yang paling efektif untuk memprediksi penyakit diabetes berdasarkan data yang tersedia?
3. Bagaimana memastikan bahwa model yang dibangun cukup akurat untuk digunakan dalam skenario deteksi awal?

### Goals
Proyek ini dibangun dengan tujuan:
1. Mengembangkan model klasifikasi berbasis machine learning untuk memprediksi diabetes menggunakan data klinis pasien.
2. Membandingkan performa algoritma machine learning Logistic Regression dan Random Forest untuk menemukan model terbaik.
3. Memberikan solusi yang dapat membantu proses skrining awal pasien dengan hasil yang cepat dan mudah dipahami.

### Solution Statements
Untuk mencapai tujuan dalam studi kasus ini, dilakukan beberapa tahapan solusi sebagai berikut:
1. Menggunakan dataset yang memuat informasi kesehatan seperti tekanan darah, kadar gula, BMI, dan lainnya.
2. Menerapkan model klasifikasi: Logistic Regression dan Random Forest.
3. Melakukan preprocessing data, tuning hyperparameter, dan membandingkan performa model menggunakan metrik seperti akurasi, precision, recall, dan F1-score.
4. Menyediakan rekomendasi model terbaik untuk prediksi risiko diabetes.

## Data Understanding
Dataset diabetes yang digunakan pada proyek ini berjumlah 768 baris(sample) dan 8 Fitur yang diambil pada:
<br/>
https://www.kaggle.com/uciml/pima-indians-diabetes-database
<br/>
### EDA - Deskripsi Variabel
## Deskripsi Dataset
Dataset ini berisi data klinis pasien dan status diabetes mereka. Berikut adalah deskripsi fitur-fiturnya:

Pregnancies: Jumlah kehamilan.
Glucose: Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral.
BloodPressure: Tekanan darah diastolik (mm Hg).
SkinThickness: Ketebalan lipatan kulit trisep (mm).
Insulin: Insulin serum 2 jam (mu U/ml).
BMI: Indeks Massa Tubuh (berat dalam kg / (tinggi dalam m)^2).
DiabetesPedigreeFunction: Fungsi silsilah diabetes (menunjukkan kemungkinan diabetes berdasarkan riwayat keluarga).
Age: Usia (tahun).
Outcome: Variabel target, menunjukkan apakah pasien memiliki diabetes (1) atau tidak (0).
Jumlah total data (setelah penanganan outliers) adalah 636. Dataset ini tidak memiliki nilai yang hilang atau data duplikat.

## Data Preparation
Dalam proyek ini, melakukan dua langkah utama dalam tahap Data Preprocessing:

1. Penanganan Outliers:
Identifikasi Outliers: mengidentifikasi outliers (nilai ekstrem) pada setiap fitur numerik menggunakan metode Interquartile Range (IQR). Metode ini menghitung kuartil pertama (Q1) dan kuartil ketiga (Q3) dari data, kemudian menentukan rentang interkuartil (IQR = Q3 - Q1). Batas bawah dan atas untuk mendeteksi outliers dihitung sebagai Q1 - 1.5 * IQR dan Q3 + 1.5 * IQR. Setiap data point yang berada di luar rentang ini dianggap sebagai outlier.

Penghapusan Outliers: kemudian menghapus baris-baris data yang mengandung outliers dari DataFrame df_healthcare. Hal ini dilakukan untuk mencegah outliers memengaruhi pelatihan model, karena nilai ekstrem dapat mendistorsi perhitungan dan menyebabkan model menjadi kurang robust.

Visualisasi Setelah Penghapusan:memvisualisasikan kembali distribusi fitur numerik menggunakan boxplot setelah menghapus outliers untuk memastikan bahwa nilai-nilai ekstrem sebagian besar sudah dihilangkan. Boxplot yang lebih "rapi" menunjukkan bahwa outliers telah berhasil ditangani.

2. Scaling Fitur Numerik:
Tujuan Scaling: Algoritma machine learning, terutama yang berbasis jarak seperti Logistic Regression, sensitif terhadap skala fitur. Fitur dengan rentang nilai yang besar dapat mendominasi fitur dengan rentang nilai yang kecil, meskipun fitur yang lebih kecil sebenarnya lebih informatif. Scaling membuat semua fitur numerik memiliki rentang nilai yang serupa.

Metode StandardScaler: menggunakan StandardScaler dari library scikit-learn. StandardScaler melakukan standarisasi fitur dengan menghapus rata-rata dan menskalakan ke varians unit. Ini menghasilkan data dengan rata-rata 0 dan standar deviasi 1.

Implementasi: mengidentifikasi kolom-kolom numerik yang perlu di-scale (kecuali kolom 'Outcome' yang merupakan target) dan kemudian menerapkan StandardScaler menggunakan fit_transform pada kolom-kolom tersebut. Hasil scaling ini menggantikan nilai asli dalam DataFrame.

Hasil: Setelah scaling, nilai-nilai pada fitur numerik akan berada dalam rentang yang mirip, seperti yang terlihat pada output df_healthcare.head() setelah proses scaling.


## Modeling
Dalam proyek ini, tujuannya adalah membangun model klasifikasi untuk memprediksi apakah seorang pasien menderita diabetes (Outcome = 1) atau tidak (Outcome = 0).

1. Membagi Data menjadi Set Pelatihan dan Pengujian:

Tujuan: Sebelum melatih model, penting untuk membagi dataset menjadi dua bagian terpisah: set pelatihan (training set) dan set pengujian (testing set). Model dilatih hanya menggunakan data pelatihan, dan kemudian dievaluasi menggunakan data pengujian yang belum pernah dilihat oleh model sebelumnya. Hal ini dilakukan untuk menguji kemampuan generalisasi model pada data baru dan menghindari overfitting (model terlalu baik pada data pelatihan tetapi buruk pada data baru).

Implementasi: Digunakan fungsi train_test_split dari scikit-learn. Dipisahkan fitur (X) dari variabel target (y). Data dibagi menjadi 80% untuk pelatihan dan 20% untuk pengujian (test_size=0.2). Parameter random_state=42 digunakan untuk memastikan bahwa pembagian data selalu sama setiap kali kode dijalankan, sehingga hasilnya reproducible. Parameter stratify=y sangat penting untuk masalah klasifikasi seperti ini, terutama jika kelas target tidak seimbang. stratify=y memastikan bahwa proporsi kelas target (diabetes vs non-diabetes) dipertahankan sama di set pelatihan dan pengujian.

2. Membangun dan Mengevaluasi Model Logistic Regression:

Algoritma: Logistic Regression adalah algoritma klasifikasi linier yang memprediksi probabilitas suatu instance termasuk dalam kelas tertentu. Meskipun namanya "Regression", ini digunakan untuk tugas klasifikasi biner atau multinomial.

Implementasi: Dibuat instance dari LogisticRegression. Parameter max_iter=200 ditambahkan untuk memastikan algoritma memiliki cukup iterasi untuk konvergensi. Model dilatih menggunakan data pelatihan (model_lr.fit(X_train, y_train)).

Evaluasi: Setelah dilatih, model digunakan untuk memprediksi kelas pada data pengujian (y_pred_lr = model_lr.predict(X_test)). Kinerja model dievaluasi menggunakan metrik:
- Accuracy: Proporsi prediksi yang benar secara keseluruhan.
- Confusion Matrix: Tabel yang menunjukkan jumlah True Positives (TP), True Negatives (TN), False Positives (FP), dan False Negatives (FN). Ini memberikan gambaran detail tentang jenis kesalahan yang dibuat model.
- Classification Report: Ringkasan metrik precision, recall, dan F1-score untuk setiap kelas. Precision adalah kemampuan model untuk tidak memprediksi positif palsu, recall adalah kemampuan model untuk menemukan semua instance positif, dan F1-score adalah rata-rata harmonik dari precision dan recall.

3. Membangun dan Mengevaluasi Model Random Forest:

Algoritma: Random Forest adalah algoritma ensemble yang membangun banyak pohon keputusan selama pelatihan dan menghasilkan output (prediksi) yang merupakan mode (klasifikasi) atau rata-rata (regresi) dari prediksi pohon-pohon individual. Ini cenderung lebih robust dan kurang rentan terhadap overfitting dibandingkan pohon keputusan tunggal.

Implementasi: Dibuat instance dari RandomForestClassifier. Parameter n_estimators=100 menentukan jumlah pohon dalam forest. random_state=42 digunakan untuk reproducibility. Model dilatih menggunakan data pelatihan (model_rf.fit(X_train, y_train)).

Evaluasi: Sama seperti Logistic Regression, model Random Forest dievaluasi menggunakan accuracy, confusion matrix, dan classification report pada data pengujian.

4. Membandingkan Kinerja Model:

Tujuan: Setelah mengevaluasi kedua model, dibandingkan metrik kinerja mereka (terutama akurasi awal) untuk menentukan model mana yang memiliki performa lebih baik pada data pengujian.

Hasil: Berdasarkan akurasi awal, Logistic Regression menunjukkan kinerja yang sedikit lebih tinggi dibandingkan Random Forest. Oleh karena itu, Logistic Regression dipilih sebagai model terbaik untuk tahap tuning selanjutnya.

5. Hyperparameter Tuning pada Logistic Regression dengan GridSearchCV:

Tujuan: Hyperparameter adalah parameter yang tidak dipelajari dari data selama pelatihan, tetapi ditetapkan sebelum pelatihan dimulai (misalnya, C, penalty, solver pada Logistic Regression, atau n_estimators pada Random Forest). Memilih kombinasi hyperparameter yang tepat dapat secara signifikan meningkatkan kinerja model. Hyperparameter tuning adalah proses mencari kombinasi hyperparameter terbaik.

Metode: Digunakan GridSearchCV. Metode ini mencoba semua kombinasi hyperparameter yang ditentukan dalam param_grid. Untuk setiap kombinasi, dilakukan cross-validation (dalam kasus ini, 5-fold cross-validation) pada data pelatihan. Cross-validation membagi data pelatihan menjadi beberapa lipatan (folds), melatih model pada sebagian lipatan, dan mengevaluasi pada lipatan yang tersisa. Ini memberikan estimasi kinerja model yang lebih robust.

Implementasi: Didefinisikan param_grid_lr dengan berbagai nilai untuk C, penalty, dan solver. Dibuat instance GridSearchCV dengan model Logistic Regression, parameter grid, jumlah lipatan CV (cv=5), dan metrik evaluasi (scoring='accuracy'). GridSearchCV kemudian dilatih pada data pelatihan (grid_search_lr.fit(X_train, y_train)).

Hasil Tuning: GridSearchCV akan menemukan kombinasi hyperparameter terbaik (best_params_) dan skor akurasi terbaik yang dicapai selama cross-validation (best_score_). Model dengan parameter terbaik ini kemudian dievaluasi lagi pada data pengujian yang belum pernah dilihat sebelumnya untuk mendapatkan metrik kinerja akhir setelah tuning.

6. Confusion Matrix untuk Model Terbaik:

Tujuan: Setelah tuning, dibuat visualisasi heatmap dari confusion matrix untuk model Logistic Regression yang telah di-tuning.

Implementasi: Digunakan seaborn.heatmap untuk menampilkan confusion matrix secara visual, dengan label yang jelas untuk prediksi dan label aktual. Ini membantu dalam memahami secara intuitif seberapa baik model membedakan antara pasien diabetes dan non-diabetes.

7. Cross-Validation dan Metrik Tambahan:

Tujuan: Untuk mendapatkan estimasi kinerja model yang lebih stabil dan komprehensif, dilakukan cross-validation dan menghitung metrik evaluasi tambahan.

Cross-Validation: Digunakan cross_val_score untuk menghitung akurasi rata-rata dan standar deviasi dari akurasi di seluruh lipatan cross-validation pada data pelatihan untuk kedua model (Logistic Regression dan Random Forest). Ini memberikan gambaran yang lebih baik tentang seberapa konsisten kinerja model.

Metrik Tambahan: Dihitung AUC-ROC (Area Under the Receiver Operating Characteristic Curve), Precision, Recall, dan F1-score pada data pengujian untuk kedua model. AUC-ROC mengukur kemampuan model untuk membedakan antara kelas positif dan negatif di berbagai ambang batas klasifikasi. Metrik-metrik ini memberikan perspektif yang berbeda tentang kinerja model selain akurasi sederhana.

## Evaluasi

1. Logistic Regression (Model Awal):

Accuracy: 0.80. Ini berarti model Logistic Regression awal mampu memprediksi status diabetes dengan benar pada sekitar 80% dari data pengujian.
Confusion Matrix:
[[80  8]
 [18 22]]
True Negative (TN): 80. Model dengan benar memprediksi 80 pasien tidak menderita diabetes.
False Positive (FP): 8. Model salah memprediksi 8 pasien tidak menderita diabetes padahal sebenarnya menderita. (Ini disebut Type I error).
False Negative (FN): 18. Model salah memprediksi 18 pasien menderita diabetes padahal sebenarnya tidak menderita. (Ini disebut Type II error).
True Positive (TP): 22. Model dengan benar memprediksi 22 pasien menderita diabetes.
Classification Report:
Menunjukkan precision, recall, dan F1-score untuk masing-masing kelas (0: tidak diabetes, 1: diabetes).
Untuk kelas 0 (tidak diabetes), model memiliki precision 0.82 dan recall 0.91, menunjukkan kinerja yang baik dalam mengidentifikasi pasien yang tidak diabetes.
Untuk kelas 1 (diabetes), model memiliki precision 0.73 dan recall 0.55, menunjukkan bahwa meskipun cukup baik dalam prediksi positif, model masih melewatkan sebagian besar kasus diabetes yang sebenarnya (recall yang lebih rendah).

2. Random Forest (Model Awal):

Accuracy: 0.77. Akurasi model Random Forest awal sedikit lebih rendah dibandingkan Logistic Regression awal.
Confusion Matrix:
[[78 10]
 [20 20]]
TN: 78, FP: 10, FN: 20, TP: 20. Dibandingkan Logistic Regression awal, Random Forest memiliki sedikit lebih banyak False Positives dan False Negatives.
Classification Report:
Untuk kelas 0, precision 0.80 dan recall 0.89.
Untuk kelas 1, precision 0.67 dan recall 0.50. Kinerja dalam mengidentifikasi kasus diabetes (kelas 1) terlihat sedikit lebih rendah pada Random Forest awal dibandingkan Logistic Regression awal.

3. Logistic Regression (Setelah Tuning Hyperparameter):

Parameter terbaik: {'C': 0.1, 'penalty': 'l1', 'solver': 'saga'}. Ini adalah kombinasi hyperparameter yang ditemukan oleh GridSearchCV yang memberikan akurasi terbaik pada data pelatihan selama cross-validation.
Skor akurasi terbaik (dari cross-validation): 0.79. Ini adalah estimasi kinerja model dengan parameter terbaik pada data pelatihan.
Accuracy setelah Tuning (pada data pengujian): 0.80. Akurasi pada data pengujian tetap 0.80.
Confusion Matrix setelah Tuning:
[[81  7]
 [18 22]]
TN: 81, FP: 7, FN: 18, TP: 22. Dibandingkan Logistic Regression awal, tuning berhasil sedikit mengurangi jumlah False Positives (dari 8 menjadi 7) sambil mempertahankan jumlah True Negatives dan True Positives, serta jumlah False Negatives.
Classification Report setelah Tuning:
Untuk kelas 0, precision 0.82 dan recall 0.92.
Untuk kelas 1, precision 0.76 dan recall 0.55. Tuning sedikit meningkatkan precision untuk kelas 1 (dari 0.73 menjadi 0.76), yang berarti ketika model memprediksi positif, kemungkinan besar itu benar. Recall untuk kelas 1 tetap 0.55.
Metrik Tambahan (setelah Tuning):
Cross-Validation Accuracy: 0.79 (+/- 0.04). Menunjukkan akurasi rata-rata yang stabil di seluruh lipatan cross-validation.
AUC-ROC: 0.83. Nilai yang cukup baik (mendekati 1.00) menunjukkan kemampuan model yang baik dalam membedakan antara kelas positif dan negatif.
Precision: 0.76. Dari semua prediksi positif, 76% benar.
Recall: 0.55. Model berhasil mengidentifikasi 55% dari total kasus diabetes yang sebenarnya.
F1-score: 0.64. Rata-rata harmonik precision dan recall.

Kesimpulan Hasil Model:

Berdasarkan perbandingan akurasi awal dan hasil setelah tuning, Logistic Regression setelah tuning adalah model terbaik dalam proyek ini dengan akurasi 0.80 pada data pengujian dan AUC-ROC 0.83. Meskipun akurasi pada data pengujian tidak meningkat secara signifikan setelah tuning, perubahan pada confusion matrix dan classification report menunjukkan peningkatan kecil dalam kemampuan model untuk menghindari False Positives.

Penting untuk dicatat bahwa recall untuk kelas 1 (diabetes) masih relatif rendah (0.55). Ini berarti model masih melewatkan hampir setengah dari pasien yang sebenarnya menderita diabetes (False Negatives). Dalam konteks deteksi dini penyakit, False Negatives bisa menjadi masalah serius karena pasien yang sakit tidak teridentifikasi. Meskipun demikian, precision untuk kelas 1 cukup baik (0.76), yang berarti ketika model memprediksi seseorang menderita diabetes, kemungkinan besar prediksi itu benar.

Analisis lebih lanjut terhadap metrik ini penting tergantung pada prioritas bisnis: apakah lebih penting untuk meminimalkan False Positives (memprediksi orang sehat sebagai sakit) atau meminimalkan False Negatives (melewatkan orang sakit). Dalam kasus deteksi dini, meminimalkan False Negatives mungkin lebih penting, bahkan jika itu berarti meningkatkan False Positives. Ini bisa dieksplorasi lebih lanjut dengan menyesuaikan ambang batas klasifikasi atau mencoba teknik lain.

Secara keseluruhan, model Logistic Regression yang telah di-tuning menunjukkan kinerja yang menjanjikan sebagai alat bantu deteksi dini diabetes, namun perlu dipertimbangkan keterbatasannya, terutama pada recall untuk kelas positif.

### Penyelesaian permasalahan
Beginilah cara mendeteksi penyakit diabetes lebih dini berdasarkan data klinis pasien menggunakan pendekatan machine learning :

1. Penggunaan Data Klinis: Proyek ini secara langsung memanfaatkan data klinis pasien (seperti glukosa, tekanan darah, BMI, dll.) sebagai input untuk model machine learning. Data ini merupakan dasar informasi yang digunakan untuk membuat prediksi.
2. Pendekatan Machine Learning Klasifikasi: Masalah deteksi diabetes (membedakan antara penderita dan bukan penderita) diformulasikan sebagai masalah klasifikasi. Digunakan algoritma klasifikasi (Logistic Regression dan Random Forest) yang memang dirancang untuk tugas seperti ini, yaitu mengkategorikan data ke dalam kelas-kelas yang telah ditentukan.
3. Pengembangan Model Prediktif: Melalui tahapan preprocessing data (penanganan outliers, scaling) dan pemodelan, dibangun model machine learning yang belajar pola dari data klinis. Model ini dilatih untuk mengenali hubungan antara fitur-fitur klinis dan status diabetes pasien.
4. Prediksi Dini: Model yang telah dilatih kemudian digunakan untuk memprediksi kemungkinan seseorang menderita diabetes berdasarkan data klinis baru. Hasil prediksi ini dapat digunakan sebagai indikasi awal atau sinyal peringatan bagi individu dan profesional medis untuk melakukan pemeriksaan lebih lanjut.
5. Pemilihan Model Terbaik: Dengan membandingkan kinerja beberapa model (Logistic Regression dan Random Forest) dan melakukan tuning hyperparameter pada model terbaik, dipilih model yang paling efektif dalam melakukan prediksi. Model terbaik ini menjadi alat utama untuk menjawab rumusan masalah.

Secara ringkas, penyelesaian masalah ini adalah dengan membangun dan menerapkan model klasifikasi machine learning yang dilatih pada data klinis pasien untuk memprediksi status diabetes. Model ini bertindak sebagai alat bantu untuk deteksi dini dengan memberikan prediksi berdasarkan informasi klinis yang tersedia.

## Referensi
[1]: World Health Organization. (2023). Diabetes. Retrieved from https://www.who.int/news-room/fact-sheets/detail/diabetes

[2]: American Diabetes Association. (2023). Standards of Medical Care in Diabetes. Retrieved from https://diabetes.org/

[3]: Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes       mellitus. Proceedings of the Annual Symposium on Computer Application in Medical Care, 261–265. https://doi.org/10.1016/0029-6554(88)90039-X

