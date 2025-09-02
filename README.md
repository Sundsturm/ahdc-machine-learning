# EKG Holter: Aplikasi Monitoring Detak Jantung dan Pendeteksian & Pengklasifikasian Aritmia

## Tentang Proyek
Proyek EKG Holter ini terbatas pada pembuatan antarmuka/GUI EKG Holter dan kecerdasan buatan untuk pendeteksian dan pengklasifikasian aritmia. Antarmuka/GUI dibuat dengan *framework* Python Tkinter. Kecerdasan buatan memanfaatkan model 1D-CNN untuk mendeteksi dan mengklasifikasikan aritmia. Dengan demikian, implementasi EKG Holter dilaksanakan di sisi perangkat lunak dan sisi perangkat keras sudah tersedia.

Proyek ini dilaksanakan selama kurang lebih dua bulan dari 7 Juli 2025 sampai kurang lebih 29 Agustus 2025.

## Peserta Kerja Praktik
| Nama Lengkap            | NIM      | Bidang          | Tanggung Jawab |
| -------- | -------- | -------- | -------- |
| Kean Malik Aji Santoso    | 13222083     | Teknik Elektro     | Kecerdasan Buatan Pendeteksian & Pengklasifikasian Aritmia |
| Prajnagastya Adhyatmika | 18322005 | Teknik Biomedis | GUI/Antarmuka EKG Holter |

## Repository Proyek
Ada beberapa *repository* yang dibuat dan digunakan untuk mengerjakan proyek ini. Ada juga *repository*/folder yang dipakai sebagai referensi pengerjaan proyek ini. Daftar *repository*/folder dapat dilihat melalui poin-poin berikut.
1. [GUI/Antarmuka EKG Holter](https://github.com/gastyaadhyatmika/-KP-EKG-Holter-Analysis-for-Arrhythmia-Detection)
2. [Kecerdasan Buatan Pendeteksian & Pengklasifikasian Aritmia](https://github.com/Sundsturm/ahdc-machine-learning.git)
3. [Google Colab Pelatihan Kecerdasan Buatan Pendeteksian & Pengklasifikasian Aritmia](https://colab.research.google.com/drive/1miIRfWbtPMGr7ze8EtRrRoscNEkoW6Wq?authuser=1#scrollTo=1SPEEx14l3u8)
4. [Google Drive TA242501010](https://drive.google.com/drive/folders/1cJJ3s59jAI2zLr1WJsEypTzV_RJOREAl)
5. [File Figma untuk GUI/Antarmuka EKG Holter](https://www.figma.com/design/T4ZLz3nBal5QjaI9wrwj3r/Mockup-ECG-Holter_PT-Xirka-Darma-Persada?node-id=3-2&t=fHxEvnm2tz4XCsaE-1)

## Spesifikasi Proyek
### Spesifikasi GUI/Antarmuka EKG Holter
GUI EKG Holter dibuat berdasarkan referensi dari TA242501010. Referensi tersebut memanfaatkan *framework* Python Tkinter. Maka dari itu, GUI diharapkan juga dibuat dengan Python Tkinter. Sebelum berlanjut ke pembuatan GUI dengan Python Tkinter, diperlukan pembuatan *mockup* untuk setiap *window*/tampilan yang akan muncul di GUI.

### Spesifikasi Kecerdasan Buatan Pendeteksian & Pengklasifikasian Aritmia
#### Alur dan Model Pembelajaran Mesin
*Machine learning* dilaksanakan dengan ***supervised learning***. Digunakan tiga buah model pembelajaran mesin, yaitu 
1. **MLP;**
CNN atau *convolutional neural network* adalah jenis model pembelajaran mesin berupa jaringan saraf yang sangat unggul dalam memproses data dengan struktur *grid*, seperti gambar atau sinyal EKG mentah. Model pembelajaran mesin ini dapat bekerja secara otomatis untuk mendeteksi dan mempelajari fitur-fitur hierarkis dari tepi sederhana hingga pola kompleks dengan lapisan konvolusi.
2. **CNN; dan**
MLP atau *multi-layer perceptron* adalah bentuk fundamental dari jaringan saraf tiruan yang terdiri dari beberapa lapisan neuron yang terhubung sepenuhnya. Setiap neuron pada satu lapisan terhubung ke setiap neuron di lapisan berikutnya. Lapisan neuron yang digunakan adalah lapisan *dense* dan lapisan *dropout*.
3. **Random Forest.**
Random Forest adalah model *ensemble learning* yang bekerja dengan membangun banyak decision trees saat pelatihan. Untuk membuat prediksi, model ini mengumpulkan output dari semua pohon lalu mengambil *majority vote* untuk klasifikasi atau mengambil rata-ratanya untuk regresi. Hal ini membuat hasil pelatihan yang lebih akurat dan tahan terhadap *overfitting* daripada satu pohon tunggal.

Setiap sesi pembelajaran/pelatihan mengikuti *pipeline* sesuai gambar berikut.
![PipelineML](https://hackmd.io/_uploads/H14NRFhKgx.png)

*Pipeline* ini merupakan gambaran kasar dalam melakukan pembelajaran mesin.

Selain *pipeline*, hasil pelatihan kecerdasan buatan dapat diketahui berdasarkan metrik-metrik tertentu. Metrik-metrik yang diutamakan untuk pelatihan ini adalah
1. **`Accuracy`**: Akurasi mengukur seringnya model dalam membuat prediksi yang benar dari keseluruhan data dan perhitungannya adalah rasio jumlah prediksi yang benar, baik positif maupun negatif, terhadap total jumlah prediksi;
2. **`Precision`**: Presisi ini fokus pada kualitas/konsistensi prediksi positif atau persentase prediksi yang benar-benar positif dari semua prediksi positif yang dilakukan oleh model;
3. **`Recall/Sensitivity`**: *Recall* atau sensitivitas mengukur kemampuan model untuk menemukan semua sampel kelas positif yang sebenarnya atau persentase prediksi yang berhasil diprediksi benar oleh model dari semua prediksi positif yang dilakukan oleh model;
4. **`F1-Score`**: Rata-rata harmonik antara `Precision` dan `Recall/Sensitivity` yang menyeimbangkan kedua metrik; dan
5. **`Specificity`**: *Spesificity* mengukur kemampuan model untuk mengidentifikasi sampel kelas negatif dengan benar atau persentase prediksi yang benar-benar negatif dari semua prediksi negatif yang dilakukan oleh model.
#### Basis Data
Basis data untuk kecerdasan buatan pendeteksian & pengklasifikasian aritmia adalah data [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/). Secara detail, basis data ini dapat dideskripsikan sebagai berikut.
* Berasal dari dua kabel *lead*
* Direkam pada frekuensi 360 Hz, resolusi 11-bit, dan rentang magnitudo 10 mV
* Basis data tersebut memiliki pelabelan sendiri sehingga pelabelannya perlu diubah untuk mengikuti **standar AAMI EC57** dan penjelasannya dapat dilihat melalui tabel berikut

| Klasifikasi Standar AAMI EC57 | Klasifikasi MIT-BIH |
| -------- | -------- |
| Normal Beat (N)     |  Normal Beat ( N atau . ), Left Bundle Branch Block Beat ( L ), Right Bundle Branch Block Beat ( R ), Atrial Escape Beat ( e ), Nodal/Junction Escape Beat ( j )    |
| Ventricular Ectopic Beat/VEB (V)     |  Premature Ventricular Contraction (V), Ventricular Escape Beat (E)    |
| Supraventricular Ectopic Beat/SVEB (S)     | Atrial Premature Beat (A), Aberrated Atrial Premature Beat (a), Nodal/Junctional Premature Beat (J), Supraventricular Premature Beat (S)     |
| Fusion Beat (F)     |   Fusion of Ventricular and Normal Beat (F)   |
| Unknown Beat (Q)     |  Paced Beat (P atau /), Fusion of Paced Beat and Normal Beat (f), Unclassifiable Beat (U)    |

## Implementasi Proyek
### Implementasi GUI/Antarmuka EKG Holter
Lihat *repository* asli antarmuka EKG Holter untuk mengetahui dan melihat GUI-nya secara detail. GUI-nya dibuat berdasarkan file Figma yang merupakan *mockup* untuk GUI EKG Holter yang kami rancang.
#### Saran/Masukan
Saran/masukan untuk proyek GUI EKG Holter adalah
1. Pecah file utama GUI menjadi ke beberapa file Python yang merepresentasikan fungsi tertentu;
2. Mengatur dan membuat *folder management* yang baik selayaknya sebuah proyek yang rapi dan profesional dari GitHub; dan
3. Mengimplementasikan fitur pada GUI yang belum diimplementasikan berdasarkan file Figma.

### Implementasi Kecerdasan Buatan Pendeteksian & Pengklasifikasian Aritmia
#### Progres Implementasi (31 Agustus 2025 & 1 September 2025)
Sejauh ini, implementasi kecerdasan buatan pendeteksian & pengklasifikasian aritmia itu masih dalam tahap pelatihan yang difokuskan pada pelatihan model 1D-CNN karena model 1D-CNN lebih bagus dalam mendeteksi data mentah, terutama data sinyal EKG. Hasil pelatihan terakhir menunjukkan bahwa performa model yang baik di kelas normal, cukup baik di kelas *ventricular*, dan belum baik di *supraventricular* dan *fusion*. Perlu dilakukan pelatihan lebih lanjut untuk model ini ataupun dua model lainnya. Jika performa model sudah baik, dapat dilanjutkan ke tahapan implementasi model yang diintegrasikan ke GUI EKG Holter.
#### Penjelasan *Repository*
```text
.
├── data
│   ├── external
│   ├── interim
│   ├── processed
│   └── raw
├── models
├── notebooks
├── references
├── reports
│   └── figures
└── src
```
##### `data`
Folder ini berisi semua data untuk proyek yang diatur berdasarkan tahapannya dalam *pipeline* pemrosesan.
* **`raw`**: Data asli yang belum diubah.
* **`interim`**: Data perantara yang telah diubah, tetapi belum siap untuk pemodelan.
* **`processed`**: Dataset akhir yang telah melalui rekayasa fitur dan siap untuk dimasukkan ke dalam model *machine learning*.
* **`external`**: Data dari sumber pihak ketiga.
##### `models`
Folder ini menyimpan model *machine learning* akhir yang sudah dilatih. Menyimpan model di sini memungkinkan model untuk dimuat kembali untuk proses inferensi tanpa perlu dilatih ulang.
##### `notebooks`
Folder ini menyimpan Jupyter Notebooks (file `.ipynb`) yang digunakan untuk pengembangan interaktif, analisis data eksplorasi (EDA), dan pembuatan prototipe model.
##### `references`
Folder ini menyimpan dokumen pendukung, seperti makalah penelitian, dokumentasi proyek, atau lembar data (*datasheet*) untuk perangkat medis.
##### `reports`
Folder ini untuk hasil yang digenerasi seperti visualisasi, metrik performa, dan laporan akhir.
* **`figures`**: Berisi semua plot yang dihasilkan, seperti *confusion matrix*, kurva ROC, atau visualisasi sinyal ECG.
##### `src`
Folder ini berisi skrip Python (`.py`) yang sudah rapi dan dapat digunakan kembali untuk proyek. Ini termasuk kode untuk pipeline pemrosesan data, pelatihan dan evaluasi model, serta *deployment* akhir.

#### Penjelasan Google Colab
File Google Colab berisi pelatihan dengan data `Windowed Features` pada tiga model. Selain itu, terdapat juga fitur *automatic hyperparameter tuning* untuk model MLP. Sayangnya, fitur tersebut tidak diimplementasikan ke masing-masing model karena *tuning* dapat dilakukan secara manual melalui *single-fold validation* ataupun *ten-fold cross validation*.

#### Penjelasan Jupyter Notebook
Ada empat file Jupyter Notebook yang dipakai selama pelatihan, yaitu
1. **`training-A.ipynb`**: Pelatihan tiga model dengan data `Windowed Features` dan *single-lead training* di *lead* MLII;
2. **`training-B.ipynb`**: Pelatihan tiga model dengan data `WPT` untuk model MLP dan model Random Forest dan `Data Mentah` untuk model 1D-CNN serta *dual-lead training* berdasarkan *MIT-BIH Arrhythmia Database*;
3. **`training-C.ipynb`**: Pelatihan tiga model dengan data `DWT` untuk model MLP dan Random Forest dan `Data Mentah` untuk model 1D-CNN serta *dual-lead training* berdasarkan *MIT-BIH Arrhythmia Database*; dan
4. **`training-D.ipynb`**: Pelatihan tiga model dengan `Data Mentah` dan *single-lead training* di *lead* .

#### Penjelasan *Computing Power* Pelatihan Google Colab dan Pelatihan Lokal
Pelatihan dengan Google Colab sering menggunakan CPU yang disediakan oleh Google Colab. Pelatihan secara lokal memanfaatkan laptop **LOQ 15IAX9E** yang memiliki **CPU Intel Core i5-12450HX 2,4GHz**, **GPU NVIDIA *Laptop GPU* RTX3050 6GB**, dan **RAM 12GB**. **Pelatihan secara lokal** **suka** mengalami ***crash*** pada Jupyter Lab ketika **menjalankan dua atau lebih file Jupyter Notebook sekaligus** karena laptop memiliki **RAM yang terbatas** dan menggunakan **OS Windows 11** yang ***bloated* atau *resource-heavy***.  

#### Persiapan & *Preprocessing* Data
Kecerdasan buatan untuk pendeteksian dan pengklasifikasian aritmia dilatih dengan **data tambahan EKG dari *simulator* aritmia** dan ***dataset* *MIT-BIH Arrhythmia Database***. *MIT-BIH Arrhtymia Database* dipisah berdasarkan **De Chazal dkk.** dan pemisahannya dapat diketahui melalui tabel berikut.
| Nama Dataset | ID Pasien | Pemakaian Dataset |
| -------- | -------- | -------- |
| `ds1`     | '101', '106', '108', '109', '112', '114', '115', '116', '118', '119','122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230' | Training & Validation
| `ds2`     | '100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234'     | Testing     |

***Dataset* tambahan EKG dari *simulator* aritmia** dijadikan sebagai ***dataset testing*** bersama dengan *dataset* `ds2`.

Secara berurutan, alur persiapan dan *preprocessing* data adalah
1. Memuat data *MIT-BIH Arrhythmia Database* dan data EKG dari *simulator* aritmia serta melakukan pemisahan *dataset* sesuai penjelasan sebelumnya;
2. Melakukan *preprocessing* data untuk sinyal EKG dari masing-masing *dataset*;
3. *Resampling* frekuensi dasar *MIT-BIH Arrhythmia Database* dari 360 Hz ke 500 Hz agar sesuai dengan data EKG yang dari *simulator*, melakukan transformasi data sesuai bentuk data yang diinginkan, dan memberikan label untuk *dataset* *MIT-BIH Arrhythmia Database*;
4.  Membaca data EKG *simulator* aritmia yang memiliki format .bin, melakukan transformasi data sesuai bentuk data yang diinginkan, dan memberikan label untuk *dataset* sinyal EKG dari *simulator* aritmia;
5.  Melakukan *standard scaling* dan menggabungkan hasil pengolahan data dari langkah 3 dan langkah 4;
6.  Membagi data gabungan menjadi beberapa *split/set* untuk *training set* sebesar 80% dan *validation set* sebesar 20% dan menerapkan algoritma SMOTEEN/SMOTE untuk *training set*; dan 
7.  *Assign* data yang sudah dibagi ke beberapa *set* ke masing-masing model yang ingin dilatih. 

Proses *preprocessing* data memanfaatkan beberapa filter digital untuk menghilangkan derau *baseline* detak jantung, derau *electrical interference*, dan derau frekuensi tinggi di data sinyal EKG. Selama melakukan pelatihan model, ada beberapa bentuk transformasi basis data yang telah dilakukan. Bentuk-bentuk transformasi basis data ini dapat dijelaskan sebagai berikut.
1. **`Windowed Features (Semua model)`**: Data durasi waktu dari sebuah *window* yang menampung setiap 10 interval RR berdasarkan pendeteksian puncak R pada waktu tertentu;
2. **`Data Mentah (1D-CNN)`**: Sinyal EKG mentah dijadikan sebagai data untuk pelatihan dan pengujian model, khususnya untuk model CNN yang digunakan;
3. **`Discrete Wavelet Transform/DWT`** (MLP & Random Forest): Transformasi sinyal EKG mentah ke transformasi diskrit dengan metode Daubechies 4;
4. **`Wavelet Packet Transform/WPT (MLP & Random Forest)`**: Transformasi sinyal EKG mentah ke data-data vektor yang berisi analisis detail morfologi sinyal pada tingkatan frekuensi sehingga diperoleh data vektor berupa energi, entropi *Shannon*, dan statistik dasar morfologinya;
5. **`Scalogram (2D-CNN)`**: Visualisasi sinyal EKG berupa grafik intensitas warna yang merepresentasikan koefisien *wavelet* pada waktu tertentu dan visualisasi ini dibuat berdasarkan aspek domain waktu dan domain frekuensi dari sinyal EKG dengan *continuous wavelet transform* (CWT); dan 
6. **`Tabular Features (MLP & Random Forest)`**: Data sinyal EKG yang diorganisasikan dalam bentuk tabel yang berisi tentang informasi hasil *heart rate variability* dan amplitudo puncak R sehingga tidak ada informasi data tentang bentuk/morfologi sinyal EKG.
Pelatihan *multimodels* terakhir menggunakan kombinasi **data mentah** untuk **model 1D-CNN** dan **WPT** untuk **model MLP dan Random Forest**. Pelatihan terakhir memanfaatkan **data mentah** untuk model **1D-CNN**.

#### Pelatihan Model Pembelajaran Mesin
Pelatihan memanfaatkan tiga buah model dan penjelasan masing-masing model dapat dilihat melalui poin-poin berikut.
1. **`MLP`**
    * Dibuat dengan **TensorFlow Keras**
    * *Layers* yang dipakai adalah
        1. `Dense 1`: **512 neuron**, ada parameter dimensi *input*, dan menggunakan fungsi ***rectified linear unit* (ReLu)**;
        2. `Dropout 1`: Regulator/pencegah *overfitting* yang mematikan **10%/(0.1)** *neuron* dari lapisan sebelumnya;
        3. `Dense 2`: **512 neuron** dengan **ReLu**;
        4. `Dropout 2`: Pencegah *overfitting* yang mematikan **40%/0.4** neuron dari lapisan sebelumnya; dan
        5. `Dense 3`: **Jumlah neuron sama dengan jumlah kelas/tipe label (`output_dim`)** dan mengubah *output* mentah menjadi distribusi probabilitas **(`softmax`)**.
    * Hasil pelatihan terakhir dapat dilihat melalui gambar berikut (31 Agustus 2025).
![Screenshot 2025-08-31 043524](https://hackmd.io/_uploads/r11jONXcll.png)

2. **`Random Forest`**
    * Dibangun dengan **cuML** yang berasal dari RAPIDS AI agar pelatihan Random Forest memanfaatkan GPU 
    * Hanya terdiri dari pengaturan **jumlah *decision trees* (`n_estimators`), yaitu 200**, dengan **tingkat pertumbuhan (`max_depth`) mencapai *level* 30** dan ***random state* (`random_state`)** yang bernilai **42**
    * Hasil pelatihan terakhir dapat dilihat melalui gambar berikut (31 Agustus 2025).
![Screenshot 2025-08-31 043449](https://hackmd.io/_uploads/SkAadVXcxx.png)

3. **`1D-CNN`**
    * Dibangun dengan **TensorFlow Keras**
    * Memanfaatkan **`epoch=150`** dan **`batch_size=100`**
    * *Layers* yang dipakai dan pengaturan nilai parameter per *layer* dapat dilihat melalui gambar berikut.
![Screenshot 2025-08-26 153255](https://hackmd.io/_uploads/H1NvQR0Kxg.png)
    Ada perubahan pada ***layer* yang *trainable***, yaitu **100, 256, dan 0.0001** untuk **filter Conv1D, *layer* Dense (Bukan *output layer Dense*), dan *learning rate optimizer* Adam**.
    * Hasil pelatihan dapat dilihat melalui gambar berikut (1 September 2025).
![image](https://hackmd.io/_uploads/HkhoUNm5le.png)

Segala pelatihan yang telah dilakukan dapat dimodifikasi agar bisa mencapai performa yang lebih baik.

#### Saran/Masukan
1. *Tuning hyperparameters trainable layers* pada 1D-CNN sampai ditemukan hasil yang paling optimal;
2. *Tuning hyperparameters* pada model lain, yaitu MLP dan Random Forest, sampai ditemukan hasil yang optimal;
3. Membuat file-file Python terpisah untuk masing-masing tahapan *pipeline* jika hasil pelatihan sudah baik agar *repository* dapat di-*reproduce* dengan baik;
4. Melihat, memahami, meniru, dan melakukan modifikasi referensi-referensi dari Github yang berkaitan dengan pelatihan model pembelajaran mesin untuk pendeteksian dan pengklasifikasian aritmia;
5. Eksplorasi *paper* lain yang berkaitan dengan kecerdasan buatan pendeteksian dan pengklasifikasian aritmia; dan
6. Implementasi model yang sudah diekspor ke GUI EKG Holter jika performa model sudah sangat bagus.

## Cara Menggunakan Proyek
### Cara Membuat *Virtual Environment* Python dan Instalasi Packages dengan pip
1. Inisiasi *virtual environment* di folder proyek;
```code
python -m venv <nama folder virtual environment>
```
2. Aktifkan *virtual environment*; dan
```code
.\<nama folder virtual environment>\Scripts\activate # Windows
source <nama folder virtual environment>/bin/activate # Linux/MacOS 
```
3. Gunakan manajer paket `pip` untuk instalasi paket-paket yang dibutuhkan atau instalasi melalui *script* .txt tertentu
```code
pip install <nama package 1>==<versi package 1> <nama package 2>==<versi package 2> ....
pip install -r <nama script>.txt # Instalasi dari script tertentu
``` 
### Menggunakan Repository GUI EKG Holter
Langkah-langkah untuk menjalankan GUI EKG Holter adalah
1. *Clone repository* GUI EKG Holter
```Code
git clone https://github.com/gastyaadhyatmika/-KP-EKG-Holter-Analysis-for-Arrhythmia-Detection.git
```
2. Inisiasi *virtual environment* Python dan lakukan instalasi *packages* yang diperlukan melalui file requirement.txt di bawah ini
**`requirement.txt`**
```
numpy
Pillow
matplotlib
scipy
tensorflow[and-cuda]
```
Untuk *package* TensorFlow, Anda perlu melakukan instalasi *prerequisite packages* sebelum melakukan instlasi TensorFlow di pip. `tensorflow[and-cuda]` merupakan sebuah *package* yang dikhususkan untuk pengguna GPU, sedangkan `tensorflow` dikhususkan untuk pengguna CPU. Lihat langkah-langkah instalasi TensorFlow dengan tepat melalui [halaman ini](https://www.tensorflow.org/install).
3. Jalankan file Python GUI EKG Holter
```
python3 "EKG Holter.py" # Python versi 3
python "EKG Holter.py" # Python versi bukan 3
```
### Menggunakan Repository dan Google Colab Kecerdasan Buatan Pendeteksian & Pengklasifikasian Aritmia
Untuk Google Colab, Anda hanya perlu membuka filenya, memilih *computing power* yang tersedia di Google Colab, dan melakukan pelatihan. Pelatihan di Google Colab lebih mudah daripada pelatihan secara lokal, tetapi *computing power* terbatas dengan durasi tertentu. Maka dari itu, pelatihan secara lokal merupakan metode pelatihan yang lebih direkomendasikan.

Langkah-langkah untuk melakukan pelatihan secara lokal adalah
1. *Clone repository* kecerdasan buatan pendeteksian dan pengklasifikasian aritmia
```
git clone https://github.com/Sundsturm/ahdc-machine-learning.git
```
2. Inisiasi *virtual environment* pada *repository*/folder proyek dan aktifkan *virtual environment*;
3. Sebelum melakukan instalasi *packages* untuk proyek ini, harap melakukan instalasi **TensorFlow yang menggunakan GPU ([Tutorial Instalasi TensorFlow](https://www.tensorflow.org/install))**, baik yang AMD maupun yang NVIDIA, karena model Random Forest berasal dari cuML yang memanfaatkan GPU untuk pelatihannya;
4. Instalasi *packages* dengan `pip` melalui `requirement.txt` yang berada di *repository*;
5. Jalankan Jupyter Lab/Jupyter Notebook; dan
```
jupyter lab # Menjalankan Jupyter Lab
jupyter notebook # Menjalankan Jupyter Notebook
```
6. Cek dan jalankan file-file `.ipynb` yang berada di folder `notebooks` untuk melakukan pelatihan model.

## Referensi
1. A. Raza, K. P. Tran, L. Koehl, and S. Li, "Designing ECG monitoring healthcare system with federated transfer learning and explainable AI," Knowledge-Based Systems, vol. 236, p. 107763, Jan. 2022, doi: 10.1016/j.knosys.2021.107763.
2. G. Silva, P. Silva, G. Moreira, V. Freitas, J. Gertrudes, and E. Luz, "A Systematic Review of ECG Arrhythmia Classification: Adherence to Standards, Fair Evaluation, and Embedded Feasibility," arXiv preprint arXiv:2503.07276, 2025. [Online]. Available: https://arxiv.org/abs/2503.07276
3. S. Aziz, S. Ahmed, and M.-S. Alouini, "ECG-based machine-learning algorithms for heartbeat classification," Sci. Rep., vol. 11, no. 1, Art. no. 18738, Sep. 2021, doi: 10.1038/s41598-021-97118-5.
4. Y. Ansari, O. Mourad, K. Qaraqe, and E. Serpedin, "Deep learning for ECG Arrhythmia detection and classification: an overview of progress for period 2017-2023," Front. Physiol., vol. 14, Art. no. 1246746, Sep. 2023, doi: 10.3389/fphys.2023.1246746.

