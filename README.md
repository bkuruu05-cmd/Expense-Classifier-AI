# 📊 Harcama Kategorilendiricisi AI (v2.0)
### İstatistiksel Yakınlık ve Yapay Zeka ile Akıllı Finansal Analiz

Bu proje, banka ekstrelerini ve harcama metinlerini **Hibrit bir zeka** ile sınıflandıran bir web uygulamasıdır. 2. sınıf bir **İstatistik öğrencisi** tarafından geliştirilmiştir.

## 🧠 Nasıl Çalışır? (Teknik Altyapı)
Uygulama, veriyi analiz ederken 3 katmanlı bir karar mekanizması kullanır:

1. **Sözlük Katmanı (Regex):** Tanımlı anahtar kelimelerle %100 doğrulukla eşleşme sağlar.
2. **Semantik Katman (Vektör Uzayı):** `sentence-transformers` kullanarak harcama metnini 384 boyutlu bir vektöre çevirir. Kategoriler arasındaki **Kosinüs Benzerliği (Cosine Similarity)** hesaplanarak anlamsal olarak en yakın kategori seçilir.
3. **Bilinmeyen Katman:** İki katmanda da eşleşme bulunamazsa işlem "Diğer" olarak işaretlenir ve loglanır.

## 🛠️ Kullanılan Teknolojiler
- **Backend:** FastAPI, Uvicorn
- **AI/ML:** Sentence-Transformers (Multilingual MiniLM), NumPy, Scikit-learn
- **Veri İşleme:** Pandas (CSV/Excel desteği)
- **Frontend:** Vanilla JS & CSS3 (Pastel Temalı Arayüz)

## 🚀 Kurulum
1. `pip install -r requirements.txt`
2. `python app.py`
3. Tarayıcıda `index.html` dosyasını açın.
