# Laporan Proyek Machine Learning - Ahmad Sholihin
## _Machine Learning Terapan Dicoding_

[(Perlihatkan Kelas)](https://www.dicoding.com/academies/319)


- Machine Learning dapat digunakan untuk meningkatkan efisiensi dari berbagai pekerjaan.
- Machine Learning dapat diimplementasikan ke berbagai industri dan berbagai jenis data sehingga kegunaannya sangat luas.
- Banyak perusahaan memiliki jumlah data yang sangat besar sehingga perlu diproses dengan machine learning untuk mendapatkan informasi  yang berarti.
- Kebutuhan karier di bidang Machine Learning sangatlah tinggi karena jumlah praktisi yang masih sedikit sehingga peluangnya masih sangat besar.
-  Pemahaman tentang Machine Learning, TensorFlow, dan Keras adalah keharusan untuk menjadi seorang Machine Learning Developer ataupun Data Scientist.
- Mengerjakan proyek-proyek Machine Learning sebagai portofolio merupakan keahlian yang harus dimiliki untuk mereka yang ingin memulai karier menjadi Machine Learning Developer.

## Domain Proyek --> Rekomendasi menggunakan Content Based Filtering dan Collaborative Filtering
---
Pada saat ini film telah menjadi salah satu media komunikasi untuk menyampaikan suatu pesan kepada sekelompok orang yang berkumpul di suatu tempat tertentu. Setiap tahun, ribuan film telah dikeluarkan oleh industri perfilman yang meliputi serial tv, episode tv, movie, video, video game, dan lainnya. Judul film yang telah banyak beredar dapat membuat masyarakat cenderung kesulitan untuk menemukan film yang mereka inginkan, maka dibutuhkanlah sebuah rekomendasi film. Rekomendasi film untuk pengguna sebaiknya memiliki kesamaan karakteristik antara film satu dengan yang lainnya. Pada website tertentu terdapat banyak data-data film yang dapat dimanfaatkan untuk merekomendasikan film kepada pengguna. Oleh karena itu, diperlukan suatu sistem yang dapat merekomendasikan film kepada pengguna[1].

Sistem rekomendasi adalah sebuah alat dan teknik perangkat lunak yang bisa memberikan saran-saran untuk item yang sekiranya bermanfaat bagi pengguna[2]. Sistem rekomendasi telah banyak digunakan dalam e-commerce, pencarian buku, aplikasi wisata, dan lain sebagainya. Sistem rekomendasi film merupakan suatu teknik yang diciptakan dengan tujuan untuk mempermudah pencarian film tertentu dari banyak kumpulan film sesuai dengan informasi yang diberikan pengguna. Informasi dari pengguna tersebut lalu diolah oleh sistem rekomendasi menjadi sebuah prediksi film. Terdapat beberapa metode yang dapat dipakai untuk membangun sistem rekomendasi yaitu Collaborative filtering, Content-based filtering, Hybrid, dan sebagainya.

Penerapan rekomendasi di dalam sebuah sistem biasanya melakukan prediksi suatu item, seperti rekomendasi film, music, buku, berita dan lain sebagainya yang menarik user. Sistem ini berjalan dengan mengumpulkan data dari user secara langsung maupun tidak[3]. Rekomendasi yang diberikan diharapkan dapat membantu pengguna dalam proses pengambilan keputusan, seperti barang apa yang akan dibeli, buku apa yang akan dibaca, atau musik apa yang akan didengar, dan lainnya [4].

Metode sistem rekomendasi yang akan digunakan kali ini adalah content-base filtering yang akan merekomendasikan film berdasarkan judul film. Serta melakukan rekomendasi menggunakan Collaborative Filtering yang berdasarkan user dan ranking yang diberikan.

##### _Referensi Didapatkan dari beberapa jurnal dibawah ini_
- [1] A. Y. Leonardo, "Sistem Rekomendasi Pemilihan Kerja untuk Mahasiswa Universitas Atmajaya Yogyakarta Menggunakan Metode Content-Based Filtering," Universitas Atma Jaya Yogyakarta, 2015.
- [2] F. Ricci, L. Rokach, B. Shapira and P. B. Kantor, Recommender Systems Handbook, Springer US, 2011.
- [3] J. Fadlil and W. F. Mahmudy, "Pembuatan sistem rekomendasi menggunakan decision tree dan clustering," Jurnal Informatika, vol. 3, no. 1, pp. 45-66, 2007.
- [4] M. J. Pazzani, "A Framework for Collaborative, Content-Based and Demographic
Filtering," Artificiall Intelligence, vol. 13, no. 5-6, pp. 393-408, 1999.


## Business Understanding
---

### Rumusan Masalah
Berlandaskan dari pemaparan latar belakang, Saya dapat merumuskan beberapa masalah yang ingin diselesaikan.
- Bagaimana membuat sistem rekomendasi film menggunakan metode content-based filtering?
- Bagaimana membuat sistem rekomendasi film menggunakan metode Collaborative Filtering?

### Tujuan
Tujuan dari proyek yang hendak diperoleh berdasarkan permasalahan yang dirumuskan di atas adalah sebagai berikut.
- Menghasilkan sebuah aplikasi web yang dapat menghasilkan sistem rekomendasi film menggunakan metode content-based filtering.
- Menghasilkan sebuah aplikasi web yang dapat menghasilkan sistem rekomendasi film menggunakan metode Collaborative filtering.

### Batasan Proyek
Untuk melakukan proyek ini makin intensif berdasarkan definisi masalah yang dipaparkan, peneliti menentukan batasan masalah sebagai berikut:
- Sistem yang dibuat hanya berfokus dalam memberikan rekomendasi film berdasarkan konten dari inputan judul film yang dimasukkan pengguna.
- Sistem akan memberikan output berupa judul, tahun pembuatan, dan durasi film yang merupakan rekomendasi yang dihasilkan..
- Algoritma rekomendasi yang digunakan adalah tf-idf dan cosine similarity..
- Dataset yang akan digunakan didapatkan dari open data kaggle.

### Solusi dari Permasalahan
- Menggunakan metode content-based filtering untuk mendapatkan rekomendasi berdasarkan judul film yang ada.
- Menggunakan metode Collaborative filtering untuk mendapatkan rekomendasi berdasarkan rating yang diberikan user film yang ada.
- Menggunakan metode Tfidf untuk melakukan rekomendasi daro content-based filtering.
- Menggunakan metode svd untuk melakukan rekomendasi dari collaborative filtering.


## Data Understanding
---

Data understanding adalah sebuah tahapan di dalam metodologi sains data dan pengembangan AI yang bertujuan untuk mendapatkan pemahaman awal mengenai data yang dibutuhkan untuk memecahkan permasalahan bisnis yang diberikan. Data understanding memberikan gambaran awal tentang:

- Kekuatan data.
- Kekurangan dan batasan penggunaan data.
- Tingkat kesesuaian data dengan masalah bisnis yang akan dipecahkan.
- Ketersediaan data (terbuka/tertutup, biaya akses, dsb).

### Dataset dan Library
Dataset yang digunakan untuk melakukan penelitian ini merupakan data movie yang didapatkan dari open data kaggle dengan judul The Movies Dataset dengan klik **[link Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)**. Dataset yang digunakan terdiri dari 4 data yaitu data movie, data keywords, data credis, dan data rating.Data Movie terdiri dari 45466 bari dan 24 kolom, Data keyword terdiri dari 46419 baris dan 2 kolom, Data credits terdiri dari 45476 dan 3 kolom, dan Data Rating terdiri dari 26024289 baris dan 4 kolom.. Library yang digunakan yaitu Pandas, Numpy, Seaborn, plotly, tfidf, dan svd. Serta Metode rekomendasi yang akan digunakan adalah Content Based Filtering dan Collaborative Filtering.
##### Variabel - Variabel yang terdapat di dataset Movie
1. adult : Apakah film tersebut untuk anak-anak atau bukan
2. belongs_to_collection : Pemilik koleksi film
3. budget : Anggaran untuk pembuatan film
4. genres : Genre dari film tersebut
5. homepage : Tautan ke beranda film
6. id : Pengidentifikasi unik untuk setiap film
7. imdb_id : Pengidentifikasi unik untuk setiap film yang ada di imdb
8. original_language : Bahasa original untuk film
9. original_title : Judul film
10. overview : Gambaran/Ringkasan dari film
11. popularity : Popular dari film
12. poster_path : Poster gambar dari film
13. production_companies : Asal Perusahaan pembuatan film
14. production_countries : Asal Negara pembuatan film
15. release_date : Tanggal peluncuran film
16. revenue : Pendapatan dari film
17. runtime : Waktu durasi film
18. spoken_languages : Bahasa lisan yang digunakan
19. status : "Dirilis" atau "Dikabarkan"
20. tagline : Tagline film
21. title : Judul film
22. video : film
23. vote_average : peringkat rata-rata yang diterima film
24. vote_count : penghitungan suara yang diterima.

##### Variabel - Variabel yang terdapat di dataset Rating
1.  UserId : Id dari user/penontong film
2.  movieId : Id dari film
3.  rating :Peringkat dari film
4.  timestamp : Waktu yang dibutuhkan

##### Variabel - Variabel yang terdapat di dataset Keywords
1.  id : Id dari film
2.  keywords : kata kunci dari film

##### Variabel - Variabel yang terdapat di dataset Credits
1.  cast : Nama Pemeran Utama dan Pemeran Pendukung
2.  crew : Nama Crew seperti Sutradara, Editor, Komposer, Penulis dll
3.  Id :Id dari film

### Visualisasi dan exploratory data analysis(EDA)
Menampilkan informasi yang ada pada dataset movie, karena dataset ini merupakan inti utama yang akan digunakan
```sh
movies.info()
```
Hasil yang informasi yang ditampilkan untuk dataset movie
![informasi dataset movie](https://user-images.githubusercontent.com/56246122/193377508-7a0ed053-0f07-453a-bcf9-4c346c70cc34.PNG) 

Melakukan pengecekan dari persebaran statistika menggunakan fungsi describe yang ada di python dari dataset movie
```sh
movies.describe()
```
Menampilkan genre film menggunakan library wordcloud.
![genre](https://user-images.githubusercontent.com/56246122/193377640-24e1f794-c439-49e1-9a2d-73667d345d36.png)

## Data Preparation
---
Data preparation adalah proses mengambil data mentah dan menyiapkannya untuk diserap dalam platform analitik. Untuk mencapai tahap akhir persiapan, data harus dibersihkan, diformat, dan diubah menjadi sesuatu yang dapat dicerna oleh alat analisis. Salah satu fungsi utama data preparation adalah memastikan keakuratan dan konsistensi data mentah yang disiapkan untuk pemrosesan dan analisis.

Ada beberapa tahapan yang dilakukan saat data preparation untuk menyelesaikan proyek ini, yaitu:
### Langkah Pertama
Mengambil beberapa variabel/kolom movie saja karena yang dibutuhkan untuk rekomendasi hanya sedikit kolom seperti id, original title, overview, genres, release date, runtim.
```sh
movies_md = movies_md[['id', 'original_title', 'overview', 'genres', 'release_date', 'runtime']]
```
Code tersebut untuk mengambil beberapa kolom saja dari dataset movie. Selain dataset movie, dataset lainnya juga dilakukan langkah yang sama yaitu hanya diambil kolom yang dibutuhkan saja.

### Penggabungan Dataset
Integrasi data adalah proses menggabungkan atau mengkombinasikan dua atau lebih set data yang berasal dari sumber yang berbeda. Salah satu manfaat yang didapatkan dengan melakukan integrasi data adalah terhindar dari duplikat data. Seperti kita ketahui, jika terdapat duplikat data maka akan mengganggu proses selanjutnya yang hendak dilakukan seperti analisis data karena nilai yang diperoleh bisa tidak konsisten. Penggabungan tabel ini akan sangat berguna bagi praktisi data jika data yang ingin dilihat tersedia di dalam beberapa tabel yang berbeda. Ada 3 jenis penggabungan data :
- Inner join merupakan penggabungan yang cukup umum digunakan, dimana pada join ini hanya akan mengambil data yang beririsan saja untuk masing-masing tabel. Sementara untuk data yang tidak sama untuk kedua tabel akan diabaikan.
- Left join yaitu menggabungkan dua tabel atau lebih, tetapi akan menampilkan semua isi dari tabel pertama kemudian untuk data di tabel kedua akan menyesuaikan dengan kolom yang ada di tabel kedua.
- Right join sebenarnya hampir mirip dengan konsep yang digunakan pada left join. Jika pada left join SQL akan menggabungkan data dengan mengikuti tabel pertama yang dianggap berada di kiri, maka pada right join data akan digabungkan sesuai dengan kolom yang ada pada tabel kedua.

Dari ketiga Macam join yang ada, pada rojek ini menggunakan left join dimana dataset utamanya yaitu data movie. Dataset yang akan digabungkan adalah data keyword dan data cast.
Setelah melakukan penggabungan dari ketiga dataset tersebut, selanjutnya dilakukan pengecekan data null lagi. Data null yang ada akan dihapus supaya data yang akan diproses bersih dan memudahkan sistem untuk melakukan rekomendasi.
Pada kolom cast dan keyword sebenernya masih berupa bentuk json, sehingga harus diambil datanya saja yang diperlukan yaitu hanya namanya saja. Berikut langkah untuk mengambil namanya saja di kolom cast dan keyword.

Selanjutnya adalah melakukan penghapusan data yang duplikat pada data menggunakan fungsi drop_duplicate. Setelah melakukan pengahpusan duplikat, selanjutnya melakukan pengecekan kolom dan baris yang akan digunakan untuk melakukan rekomendasi. Dataset yang akan digunakan untuk melakukan rekomendasi menggunakan metode Content Based Filtering sebesar 5769 baris dan 5 kolom. Setelah dirasa dataset yang akan digunakan rekomendasi sudah bersih dan layak diproses, selanjutnya langsung ke proses pembuatan sistemnya.


## Modeling
---
### Pengertian Metode yang digunakan
Sistem rekomendasi adalah suatu system yang digunakan oleh para user/customer/pelanggan untuk mendapatkan produk yang diinginkan. Ide awal dari sistem rekomendasi sendiri adalah untuk menggunakan beberapa sumber informasi, tujuan utama dari sistem rekomendasi adalah untuk meningkatkan penjualan produk. Terdapat beragam metode yang digunakan untuk membuat sistem rekomendasi.
##### Content Based Filtering
Sistem rekomendasi berbasis konten (Content-based Recommendation System) menggunakan ketersediaan konten (sering juga disebut dengan fitur, atribut atau karakteristik) sebuah item sebagai basis dalam pemberian rekomendasi [1]. Secara umum, metode content-based filtering mempunyai 2 teknik umum dalam membuat rekomendasi yaitu heuristic-based dan model-based. Cosine similarity, Boolean query, teknik TF-IDF (term frequency-invers document frequency) dan Clustering termasuk dalam golongan heuristic-based sedangkan yang masuk dalam golongan model-based adalah teknik Bayesian classifier & Clustering, Decision Tree dan Artificial Neural Network.
Metode content-based filtering, item direkomendasikan berdasarkan perbandingan antara profil item dan profil pengguna. Profil pengguna adalah konten yang ditemukan relevan dengan pengguna dalam bentuk kata kunci atau fitur. Profil pengguna dapat dilihat sebagai serangkaian kata kunci yang dikumpulkan oleh algoritma dari item yang relevan atau menarik oleh pengguna. Satu set kata kunci dari suatu item adalah profil item. Keuntungan dari metode content-based filtering adalah dapat merekomendasikan barang yang tidak diberikan peringkat, dapat dengan mudah menjelaskan cara kerja sistem rekomendasi dengan daftar fitur konten dari suatu item.
##### Collaborative Filtering
Collaborative filtering adalah teknik dalam sistem rekomendasi yang populer digunakan saat ini. Banyak penelitian yang membahas tentang teknik ini karena beberapa keunggulannya seperti: menghasilkan serendipity(tak terduga) item,  sesuai trend market, mudah diimplementasikan dan memumgkinkan diterapkan pada beberapa domain (book, movies, music, dll). Cara kerja teknik ini adalah dengan memanfaatkan data pada komunitas dengan cara mencari kemiripan atar pengguna, yaitu mengasumsikan bahwa pengguna yang memiliki preferensi serupa di masa lalu cenderung memiliki preferensi yang sama di masa depan. Pada dasarnya kita akan lebih percaya dengan rekomendasi dari orang yang memiliki preferensi sama dengan kita, inilah dasar yang digunakan oleh collaborative filtering dalam mengenerate item rekomendasi. Dalam menghasilkan rekomendasi, sistem perlu mengumpulkan data. Tujuan dari pengumpulan data ini adalah untuk mendapatkan ide preferensi pengguna, dimana nantinya akan dibuat suatu rekomendasi berdasarkan preferensi tersebut.
#### Term Frequency Inverse Document Frequency
TF-IDF membantu dalam mengevaluasi pentingnya sebuah kata dalam sebuah dokumen. Dalam kasus ini dokumen yang dimaksud adalah Metode TF-IDF merupakan metode untuk menghitung bobot setiap kata yang paling umum digunakan pada information retrieval. Metode ini juga terkenal efisien, mudah dan memiliki hasil yang akurat. Metode TF-IDF ini menggabungkan 2 konsep yaitu frekuensi kemunculan sebuah kata di dalam sebuah dokumen dan inverse frekuensi dokumen yang mengandung kata tersebut. Dalam menghitung bobot dengan metode ini, nilai TF per kata masing-masing berbobot 1. Sedangkan nilai IDF dihitung dengan rumus :

$$idf(w) = log({ N\over df_t})$$

Idf(w) adalah nilai IDF dari setiap kata yang akan dicari, N adalah jumlah keseluruhan dokumen yang ada, df adalah jumlah kemunculan kata t pada semua dokumen[5]. Banyak kata yang muncul pada dokumen dalam kasus ini adalah kata yang muncul pada sinopsis cerita. Lalu akan dihitung juga bobot kata untuk aktor, sutradara, rumah produksi, dan genre yang muncul pada data film.

#### Cosine Similarity
Cosine similarity berfungsi sebagai alat untuk membandingkan suatu kemiripan dari dokumen ke dokumen. Dalam hal ini yang bisa dibandingkan adalah sebuah query dengan dokumen latih. Dalam sebuah proses menghitung cosine similarity, pertama yang dilakukannya adalah melakukan sebuah perkalian skalar antara query dengan sebuah dokumen yang kemudian ditambahkan, lalu itu melakukan perkalian antara ukuran panjang dari dokumen dengan ukuran panjang query yang telah dikuadratkan. Kesamaan kosinus mengukur kesamaan antara dua vektor ruang hasil kali dalam. Ini diukur dengan cosinus sudut antara dua vektor dan menentukan apakah dua vektor menunjuk ke arah yang kira-kira sama. Ini sering digunakan untuk mengukur kesamaan dokumen dalam analisis teks.
Cosine Similarity dapat diimplementasikan untuk menghitung nilai kemiripan antar kalimat dan menjadi salah satu teknik untuk mengukur kemiripan teks yang popular. Dari hasil matriks algoritma TF-IDF akan dilakukan perhitungan untuk menghitung nilai kesamaan antar film. Sedangkan rumus Cosine Similarity sebagai berikut.

$$cos(A, B) = { A*B\over {|A|}{|B|}} = {\sum_{n=1}^j (nA * nB)\over {\sqrt{{\sum_{n=1}^j (nA)^2}}X {\sqrt{{\sum_{n=1}^j (nB)^2}}} }}$$

Keterangan:
Cos(A,B) = Nilai kemiripan antara item x dan item y.
A = vektor A
B = vektor B
nA = jumlah kemunculan kata indeks ke-n dari daftar kata pada kalimat A
nB = jumlah kemunculan kata indeks ke-n dari daftar kata pada kalimat B.
Disini A merupakan bobot setiap ciri pada vektor A, dan B merupakan bobot ciri pada vektor B, jika dikaitkan dengan information retrieval maka A adalah bobot istilah pada dokumen A, dan B merupakan bobot setiap istilah pada document B[7]

#### Single Value Decomposition
Salah satu cara untuk menangani masalah skalabilitas dan sparsity yang dibuat oleh CF adalah dengan memanfaatkan model faktor laten untuk menangkap kesamaan antara pengguna dan item. Pada dasarnya, kami ingin mengubah masalah rekomendasi menjadi masalah optimasi. Kita dapat melihatnya sebagai seberapa baik kita dalam memprediksi peringkat untuk item yang diberikan pengguna. Salah satu metrik umum adalah Root Mean Square Error (RMSE). Semakin rendah RMSE, semakin baik kinerjanya.Faktor laten adalah ide luas yang menggambarkan properti atau konsep yang dimiliki pengguna atau item. Misalnya, untuk musik, faktor laten dapat merujuk pada genre musik tersebut. SVD mengurangi dimensi matriks utilitas dengan mengekstrak faktor latennya. Pada dasarnya, kami memetakan setiap pengguna dan setiap item ke dalam ruang laten dengan dimensi r. Oleh karena itu, ini membantu kami lebih memahami hubungan antara pengguna dan item karena mereka dapat dibandingkan secara langsung.

### Pembuatan Content Based Filtering
Melakukan import library TFidf terlebih dahulu, TF-IDF digunakan untuk untuk menghitung bobot setiap kata yang paling umum digunakan pada information retrieval. Pada proyek ini menggunakan maksimal feature yang di pakai adalah 5000 kata, sehinggal dapat menampung banyak kata yang berbeda d dalam dataset film. Menampilkan vektorisasi yang telah dilakukan oleh library tfudf sebelumnya sehinggal akan menghasilkan data seperti berikut.
![vektorisasi](https://user-images.githubusercontent.com/56246122/193393793-d142ac66-e101-4198-9765-ce05c11de332.PNG)
Setelah dilakukan Tfidf, selanjutnya melakukan import svd telebih dahulu. SVD mengurangi dimensi matriks utilitas dengan mengekstrak faktor latennya karena dapat membantu kami lebih memahami hubungan antara pengguna dan item karena mereka dapat dibandingkan secara langsung. Diproyek ini menggunakan component sebesar 3000. Setelah memproses menggunakan svd, selanjutnya dilakukan proses persamaan cosine. Cosine similarity berfungsi sebagai alat untuk membandingkan suatu kemiripan dari dokumen ke dokumen. Selanjutnya membuat fungsi untuk melakukan rekomendasi film berdasarkan kolom dari original title.
Berikut hasil dari Rekomendasi metode Content Based Filtering dengan dengan 19 Teratas rekomendasi yang didapatkan.
```sh
recomendation_system('The Matrix')
```

Tabel 1.0 Hasil Rekomendasi dengan metode Content Based Filtering
| Title | Tahun Pembuatan | Durasi film |
| ------ | ------ | ------ |
| The Matrix Revolutions | 2003-11-05 | 129 Menit |
| The Matrix Reloaded | 2003-05-15 | 138 Menit |
| The Animatrix | 2003-05-09 | 102 Menit |
| Commando | 1985-10-03 | 90 Menit |
| GHOST IN THE SHELL | 1995-11-18 | 83 Menit |
| Terminator 3: Rise of the Machines | 2003-07-02 | 109 Menit |
| Hackers | 1995-09-14 | 107 Menit |
| Tron | 1982-07-09 | 96 Menit |
| Who Am I - Kein System ist sicher | 2014-09-25 | 105 Menit |
| The Zero Theorem | 2014-01-02 | 107 Menit |
| サマーウォーズ | 2009-08-01 | 114 Menit |
| Pompeii | 2014-02-18 | 105 Menit |
| Æon Flux | 2005-11-30 | 93 Menit |
| Terminator Salvation | 2009-05-20 | 115 Menit |
| Strange Days | 1995-10-13 | 145 Menit |
| 2001: A Space Odyssey | 1968-04-10 | 149 Menit |
| 攻殻機動隊 2.0 | 2008-07-12 | 85 Menit |
| The Colony | 2013-04-12 | 95 Menit |
| アップルシード | 2004-04-17 | 101 Menit |

### Pembuatan Collaborative Filtering
Pembuatan sistem rekomendasi yang menggunakan metode Collaborative Filtering langkah pertama adalah melakukan import library yang akan digunakan seperti svd, reader, dataset, dan cross_validate. Dataset yang akan digunakan pada metode Collaborative Filtering adalah dataset dari rating. Dataset rating yang akan digunakan adalah kolom userId, movieId, dan rating.
Melakukan pembuatan rekomendasi menggunakan metode SVD, proses pembuatan svd menggunakan keseluruhan data dari dataset rating, dengan menggunakan cross validation sebesar 5.
```sh
trainset = data.build_full_trainset()
Svd.fit(trainset)
```
Setelah melakukan training dengan menggunakan metode Collaborative Filtering, selanjutnya melakukan pengecekan hasil yang didapatkan. Sebagai contoh akan menampilkan user id 1 dan menghasilkan beberapa movieId rekomendasi yang didapatkan sebagai berikut.
![collaborative](https://user-images.githubusercontent.com/56246122/193399577-3fa85b1b-c097-4f9c-8a57-01c8b93f2145.PNG)
Selanjutnya melakukan pengecekan prediksi yang didapatkan dengan userID 1, movieId 302, dan rating 3.
![callaborative 2](https://user-images.githubusercontent.com/56246122/193399206-5272128a-f0b4-4831-b2f7-0788c7203d1b.PNG).
Mendapatkan hasil estimasi prediksi sebesar 4.128. Dari hasil yang didapatkan menunjukan banyaknya perkiraan rekomendasi yang didapatkan.

## Evaluasi
---

Evaluasi adalah kegiatan terencana untuk mengukur, menilai, dan keberhasilan suatu program. Evaluasi merupakan cara terbaik untuk menguji efektivitas dan produktivitas dari suatu program. Adanya evaluasi yang dilakukan tentu untuk mengetahui seberapa jauh kebutuhan, nilai, dan kesempatan yang telah dicapai. Dengan evaluasi bisa diketahui pencapaian suatu tujuan, sasaran, dan target tertentu.
Evaluasi yang digunakan pada metode rekomendasi Collaborative Filtering adalah evaluasi dengan menggunakan MAE dan RMSE
1. MAE atau Mean Absolute Error menunjukkan nilai kesalahan rata-rata yang error dari nilai sebenarnya dengan nilai prediksi. MAE sendiri secara umum digunakan untuk pengukuran prediksi error pada analisis time series.
2. Root Mean Squared Error (RMSE) merupakan salah satu cara untuk mengevaluasi model regresi linear dengan mengukur tingkat akurasi hasil perkiraan suatu model. RMSE dihitung dengan mengkuadratkan error (prediksi – observasi) dibagi dengan jumlah data (= rata-rata), lalu diakarkan. RMSE tidak memiliki satuan. RMSE adalah cara standar dan populer untuk mengukur kesalahan suatu model dalam memprediksi data kuantitatif yang menunjukkan seberapa tersebar data di sekitar garis yang paling cocok. Semakil nilai RMSE mendekati 0 maka sistem rekomendasi memiliki akurat yang sangat tinggi.

### Hasil Evaluasi metode Content Based Filtering
Pada metode Content Based Filtering hasil evaluasi langsung ditampilkan dalam bentuk rekomendasi dari film yang di masukan. Berikut contoh hasil evaluasi dengan input original title "The Matrix" 

Dari hasil tabel 1.0 yang ada di Bagian hasil Modeling, didapatkan 19 hasil top teratas metode Content Based Filtering. Hasil tersebut mendapatkan 3 Judul film yang berbahasa selain inggris. sehingga didapatkan dari 19 item yang direkomendasikan, 16 item memiliki kategori title bahasa inggris. Artinya, precision sistem kita sebesar 16/19 atau 84,2%. Dengan perhitungan sebagai berikut
Presisi = Hasil Sesuai / Total hasil
Presisi = 16 /19
Presisi = 0,8421 ==> 84,2%

### Hasil Evaluasi metode Collaborative Filtering
Hasil Evaluasi rekomendasi dengan menggunakan metode Collaborative Filtering yang menggunakan metrik MAE dan RMSE dengan cross_validation sebanyak 5 kali.

Tabel 2.0 Hasil Rekomendasi dengan metode Collaborative Filtering
| cross_validate | RMSE | MAE | fit_time | test_time |
| ------ | ------ | ------ | ------ | ------ |
| CV 1 | 0.79592337 | 0.60202421 | 1105.187 | 100.870 |
| CV 2 | 0.79580482 | 0.60205324 | 1211.009 | 81.124 |
| CV 3 | 0.79638364 | 0.60237745 | 1130.229 | 73.294 |
| CV 4 | 0.79607742 | 0.60208224 | 925.825 | 73.672 |
| CV 5 | 0.79646898 | 0.60236943 | 922.419 | 58.854 |
| **Mean** | 0.796131646 | 0.602181314 | 1058.9338 | 77.5628|
| **Std** | 0.000287415 | 0.000176605 | 129.1350783 | 15.32390202 |

Hasil evaluasi MAE yang didapatkan dengan menggunakan metode Collaborative Filtering adalah sebesar 60%, jadi rekomendasi yang dihasilkan tidak terlalu jelek karena masih bisa 60% rekomendasi kesamaan yang didapatkan.
Mencoba hasil hari rekomendasi Collaborative Filtering dengan contoh userID 1 dan id movie 302 serta rating 3.
```sh
Svd.predict(1, 302, 3)
```
Mendapatkan perkiraan prediksi sebesar 4,128. Salah satu kelebihan dari rekomendasi Collaborative Filtering tidak memperdulikan dari apa filmnya. Karena ini bekerja murni berdasarkan ID film yang ditetapkan dan mencoba memprediksi peringkat berdasarkan bagaimana pengguna lain memprediksi film.

## Kesimpulan
---
Dari hasil proyek yang teah dibuat menggunakan 2 metode yaitu Content Based Filtering dan Collaborative Filtering. Hasil dari metode Content Based Filtering menghasilkan presisi sebesar 84,2%, sehingga dari hasil tersebut mengartikan bahwa metode yang digunakan sudah cukup bagus untuk melakukan rekomendasi fil berdasarkan original titlenya. Sedangkan hasil dari metode Collaborative Filtering mengahasilkan rata-rata MAE dari 5 kali percobaan sebesar 60%, sehingga dari hasil tersebut menunjukan bahwa rekomendasi Collaborative Filtering belum cukup baik untuk menghasilkan rekomendasi yang diinginkan. Tetapi kelebihan dari rekomendasi Collaborative Filtering tidak memperdulikan dari apa filmnya. Karena ini bekerja murni berdasarkan ID film yang ditetapkan dan mencoba memprediksi peringkat berdasarkan bagaimana pengguna lain memprediksi film.

## Referensi
---
Projek ini dilakukan dengan beberapa referensi yang didapatkan:
- [1] A. Y. Leonardo, "Sistem Rekomendasi Pemilihan Kerja untuk Mahasiswa Universitas Atmajaya Yogyakarta Menggunakan Metode Content-Based Filtering," Universitas Atma Jaya Yogyakarta, 2015.
- [2] F. Ricci, L. Rokach, B. Shapira and P. B. Kantor, Recommender Systems Handbook, Springer US, 2011.
- [3] J. Fadlil and W. F. Mahmudy, "Pembuatan sistem rekomendasi menggunakan decision tree dan clustering," Jurnal Informatika, vol. 3, no. 1, pp. 45-66, 2007.
- [4] M. J. Pazzani, "A Framework for Collaborative, Content-Based and Demographic
Filtering," Artificiall Intelligence, vol. 13, no. 5-6, pp. 393-408, 1999.
- [5] M. Leben, "Applying Item-Based and User-Based Collaborative Filtering on the Netflix Data," Hasso-Plattner-Institut, Potsdam, 2008.
- [6] A. A. Ma'arif, "Penerapan Algoritma TF-IDF Untuk Pencarian Karya Ilmiah," Universitas Dian Nuswantoro, Semarang, 2015.
- [7] D. A. R. Ariantini, A. S. M. Lumenta and A. Jacobus, "Pengukuran Kemiripan Dokumen Teks Bahasa Indonesia Menggunakan Metode Cosine Similarity," Jurnal Teknik Informatika, vol. 9, no. 1, pp. 1-8, 2016.


Berikut profil saya:

- [Github](https://github.com/Asholihin1705) - Profil Github
- [Linkedin](https://www.linkedin.com/in/ahmadsholihin/) - Profil Linkedin
- [Medium](https://medium.com/@ahmadsholihin1705) - Profil Medium

