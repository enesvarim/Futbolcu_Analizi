Yapay Sinir Ağları Dersi Proje Ödevi

Bu proje, futbolcuların istatistiksel verilerini kullanarak oyuncuların oyun stillerini analiz etmeyi ve benzer özelliklere sahip oyuncuları gruplandırmayı amaçlamaktadır.

## 🎯 Projenin Amacı

Futbolcular genellikle gol ve asist gibi basit metriklerle değerlendirilir.
Bu projede ise oyuncuların:

* hücum
* pas
* defans
* top kullanımı

gibi farklı yönleri birlikte ele alınarak daha kapsamlı bir analiz yapılmaktadır.

Amaç, oyuncuları oyun stillerine göre otomatik olarak gruplandırmaktır.

---

## 📊 Veri Seti

Veri seti Kaggle üzerinden elde edilmiştir.
Ancak veri doğrudan kullanılmamış, proje ihtiyaçlarına göre yeniden düzenlenmiştir.

Yapılan işlemler:

* Gereksiz sütunlar çıkarıldı
* Kaleciler veri setinden kaldırıldı
* Az süre oynayan oyuncular filtrelendi
* Anlamlı metrikler seçildi

Kullanılan bazı özellikler:

* npxG, xAG
* KP, PrgP
* Tkl, Int, Blocks
* PrgC, Succ

---

## 🧠 Kullanılan Yöntemler

Projede iki temel yaklaşım kullanılmaktadır:

### 1. Autoencoder (Yapay Sinir Ağı)

Oyuncuların çok boyutlu verilerini daha anlamlı bir temsil haline getirmek için kullanılır.

### 2. K-Means Kümeleme

Benzer özelliklere sahip oyuncuları gruplamak için kullanılır.

---

## ⚙️ Proje Yapısı

Proje şu bileşenlerden oluşmaktadır:

* Veri ön işleme
* Model eğitimi
* Kümeleme analizi
* Görselleştirme
* (Opsiyonel) Frontend arayüz

---

## 🚀 Planlanan Özellikler

* Oyuncu arama
* Oyuncu hangi grupta?
* Benzer oyuncuların listelenmesi
* Grafiksel analizler

---

## 👥 Ekip

Proje 5 kişilik bir ekip tarafından geliştirilmektedir.
Görevler veri işleme, model geliştirme, analiz ve arayüz geliştirme olarak paylaşılmıştır.

---

## 📌 Not

## Bu proje eğitim amaçlı geliştirilmiştir ve gerçek futbol verileri üzerinde çalışmaktadır.

## 🏁

Projenin amacı, futbolun artık sadece skor değil, veri ile analiz edilen çok boyutlu bir oyun olduğunu göstermektir.
