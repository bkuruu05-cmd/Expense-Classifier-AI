from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from typing import List
from collections import Counter
import numpy as np
import uvicorn
import re
import pandas as pd
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MODEL ───────────────────────────────────────────────────────────────────
# Flan-T5 yerine sentence-transformers — Türkçe dahil 50+ dil destekler
print("--- Semantic model yükleniyor... ---")
semantic_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("--- Model hazır ---")

# ─── SÖZLÜK ──────────────────────────────────────────────────────────────────

KATEGORILER = {
    "Ulaşım": [
        "ARABA", "ARAÇ", "OTO", "SHELL", "BP", "OPET", "PETROL", "OFISI",
        "AKARYAKIT", "YAKIT", "OTOPARK", "PARK", "UBER", "BOLT", "TAKSI",
        "BİLET", "METRO", "METROBUS", "OTOBÜS", "UÇAK", "THY", "PEGASUS",
        "ANADOLUJET", "RENT", "KİRALIK",
    ],
    "Gıda/Market": [
        "MIGROS", "BİM", "A101", "ŞOK", "CARREFOUR", "MACRO", "BURGER",
        "MCDONALD", "KFC", "PIZZA", "DÖNER", "RESTORAN", "CAFE", "KAHVE",
        "STARBUCKS", "GLORIA", "SIMIT", "FIRIN", "MARKET", "MANAV", "KASAP",
        "YEMEK", "GETIR", "YEMEKSEPETI", "TRENDYOL YEMEK", "MIGROS SANAL"
    ],
    "Eğlence": [
        "NETFLIX", "SPOTIFY", "YOUTUBE", "AMAZON PRIME", "BLUTV", "GAIN",
        "EXXEN", "SİNEMA", "CİNEMAXIMUM", "CINEMAXIMUM", "OYUN", "STEAM",
        "PS", "XBOX", "BİLETİX", "KONSER", "TİYATRO"
    ],
    "Alışveriş": [
        "TRENDYOL", "HEPSİBURADA", "N11", "AMAZON", "ZARA", "H&M",
        "LC WAIKIKI", "LCW", "KOTON", "DEFACTO", "MANGO", "IPEKYOL",
        "VAKKO", "BOYNER", "TEKNOSA", "MEDIAMARKT", "VATAN"
    ],
    "Sağlık": [
        "ECZANE", "PHARMACY", "HASTANE", "KLİNİK", "DOKTOR", "DİŞ",
        "OPTİK", "GÖZLÜK", "MEDLINE", "ACIBADEM", "MEMORIAL", "FLORENCE"
    ],
    "Finans/Banka": [
        "ZIRAAT", "AKBANK", "GARANTİ", "İŞBANKASI", "ISBANK", "YAPI KREDİ",
        "HALKBANK", "VAKIFBANK", "DENİZBANK", "QNB", "ING", "FAİZ", "KREDİ",
        "BORÇ", "ÖDEME", "TRANSFER", "EFT", "HAVALE", "ATM"
    ],
    "Faturalar": [
        "TEDAŞ", "AYEDAŞ", "ENERJİSA", "IGDAŞ", "ISKI", "ELEKTRİK",
        "DOĞALGAZ", "TURKCELL", "VODAFONE", "TÜRK TELEKOM", "TTNET",
        "INTERNET", "FATURA", "ABONELİK"
    ],
    "Eğitim": [
        "UDEMY", "COURSERA", "OKUL", "ÜNİVERSİTE", "KURS", "KİTAP",
        "D&R", "PANDORA", "ROBINSON"
    ],
}

# ─── SEMANTİK ÖRNEKLER (2. Katman) ───────────────────────────────────────────
# Sözlükte olmayan işlemleri anlamsal olarak yakalamak için örnek cümleler
KATEGORI_ORNEKLERI = {
    "Ulaşım": [
        "benzin aldım", "akaryakıt ödemesi", "yakıt doldurdum",
        "otopark ücreti ödedim", "araç park ettim",
        "taksi bindim", "servis ücreti", "şoför ödemesi",
        "metro kartı yükledim", "otobüs bileti aldım", "toplu taşıma",
        "uçak bileti", "havayolu ödemesi", "seyahat bileti",
        "köprü geçiş ücreti", "otoyol geçişi", "HGS yükleme",
        "araç kiralama","otoban geçiş ücreti", "otoyol geçişi", "köprü geçişi",
        "benzin dolumu", "benzin aldım", "akaryakıt istasyonu ödemesi",
        "istasyon ödemesi", "yakıt istasyonu",
        "vapur bileti", "vapur geçişi", "deniz otobüsü bileti", "feribot",
        "araç kiralama", "rent a car ödemesi",
        "HGS geçiş", "OGS geçiş", "köprü HGS", "rent a car",
    ],
    "Gıda/Market": [
        "marketten alışveriş yaptım", "süpermarket ödemesi",
        "gıda alışverişi", "sebze meyve aldım", "kasaptan et aldım",
        "bakkaldan alışveriş", "manav ödemesi",
        "restoranda yedim", "yemek yedim", "lokanta ödemesi",
        "kafede oturdum", "kahve içtim", "çay içtim",
        "yemek siparişi verdim", "eve yemek söyledim", "paket servis",
        "fast food yedim", "burger yedim", "pizza söyledim",
        "tatlı aldım", "pastane ödemesi",
    ],
    "Eğlence": [
        "dizi film izleme aboneliği", "streaming servisi ödemesi",
        "müzik dinleme aboneliği", "müzik servisi",
        "oyun satın aldım", "video oyunu ödemesi", "oyun içi satın alma",
        "sinema bileti aldım", "film izledim",
        "konser bileti", "tiyatro bileti", "etkinlik bileti",
        "eğlence merkezi", "lunapark", "bowling",
    ],
    "Alışveriş": [
        "online alışveriş yaptım", "e-ticaret siparişi",
        "kıyafet aldım", "giysi satın aldım", "elbise aldım",
        "ayakkabı aldım", "çanta aldım", "aksesuar aldım",
        "elektronik ürün aldım", "telefon aksesuar", "bilgisayar malzemesi",
        "ev eşyası aldım", "mobilya ödemesi", "dekorasyon ürünü",
    ],
    "Sağlık": [
        "eczaneden ilaç aldım", "ilaç ödemesi", "reçete ödemesi",
        "doktora gittim", "muayene ücreti ödedim", "klinik ödemesi",
        "hastane masrafı", "tedavi ücreti",
        "diş hekimine gittim", "diş tedavisi",
        "gözlük yaptırdım", "lens aldım", "optisyen ödemesi",
        "spor salonu üyeliği", "fitness merkezi",
    ],
    "Finans/Banka": [
        "banka transferi yaptım", "EFT gönderdim", "havale yaptım",
        "kredi kartı ödemesi", "kredi taksiti ödedim",
        "faiz ödemesi", "borç ödemesi",
        "ATM'den para çektim", "nakit çekim",
        "yatırım hesabına para gönderdim", "fon aldım",
        "kripto para aldım", "borsa işlemi",
    ],
    "Faturalar": [
        "elektrik faturası ödedim", "elektrik ödemesi",
        "doğalgaz faturası ödedim", "gaz ödemesi",
        "su faturası ödedim", "su ödemesi",
        "internet faturası ödedim", "internet aboneliği",
        "telefon faturası ödedim", "cep telefonu faturası",
        "kira ödedim", "aidat ödedim", "site aidatı",
    ],
    "Eğitim": [
        "online kurs satın aldım", "eğitim platformu ödemesi",
        "okul taksiti ödedim", "üniversite harcı",
        "kitap satın aldım", "ders kitabı aldım",
        "kurs ücreti ödedim", "özel ders ödemesi",
        "sertifika programı ödemesi", "sınav ücreti",
    ],
}

# Uygulama başlarken vektörler bir kez hesaplanır, her istekte tekrar hesaplanmaz
print("--- Kategori vektörleri hesaplanıyor... ---")
KATEGORI_VEKTORLERI = {}
for kategori, ornekler in KATEGORI_ORNEKLERI.items():
    vektorler = semantic_model.encode(ornekler)
    KATEGORI_VEKTORLERI[kategori] = np.mean(vektorler, axis=0)
print("--- Vektörler hazır ---")

SEMANTIC_ESIK = 0.35  # Bu değerin altı "Diğer" sayılır


# ─── ANALİZ FONKSİYONLARI ────────────────────────────────────────────────────

def sozluk_analiz(metin: str):
    metin_ust = metin.upper()
    for kategori, anahtar_kelimeler in KATEGORILER.items():
        for kelime in anahtar_kelimeler:
            if re.search(r'\b' + re.escape(kelime) + r'\b', metin_ust):
                return {
                    "cevap": f"[SÖZLÜK] Kategori: {kategori} — '{kelime}' eşleşti.",
                    "guven": 100.0,
                    "tip": "Sözlük",
                    "kategori": kategori
                }
    return None


def semantic_analiz(metin: str):
    metin_kucuk = metin.lower()
    metin_vek = semantic_model.encode([metin_kucuk]) 
    
    # geri kalan her şey aynı kalıyor

    en_iyi_kategori = None
    en_yuksek_skor = 0.0

    for kategori, vek in KATEGORI_VEKTORLERI.items():
        skor = float(cosine_similarity(metin_vek, [vek])[0][0])
        if skor > en_yuksek_skor:
            en_yuksek_skor = skor
            en_iyi_kategori = kategori

    if en_yuksek_skor >= SEMANTIC_ESIK:
        return {
            "cevap": f"[SEMANTİK] Kategori: {en_iyi_kategori} (benzerlik: %{round(en_yuksek_skor * 100, 1)})",
            "guven": round(en_yuksek_skor * 100, 1),
            "tip": "Semantik",
            "kategori": en_iyi_kategori
        }
    return None


def ozet_rapor_olustur(sonuclar: list) -> dict:
    toplam = len(sonuclar)
    kategoriler = [s["kategori"] for s in sonuclar]
    sayac = Counter(kategoriler)

    en_yuksek = sayac.most_common(1)[0]

    yuzdelik = {
        kategori: round((adet / toplam) * 100, 1)
        for kategori, adet in sayac.items()
    }

    tip_sayac = Counter(s["tip"] for s in sonuclar)

    ozet_metin = (
        f"Toplam {toplam} harcama analiz edildi. "
        + ", ".join([f"%{yuzde} {kat}" for kat, yuzde in yuzdelik.items()])
        + f". En yüksek harcama kalemi: {en_yuksek[0]} ({en_yuksek[1]} işlem)."
    )

    return {
        "toplam_islem": toplam,
        "kategori_dagilimi": yuzdelik,
        "en_yuksek_kategori": en_yuksek[0],
        "katman_dagilimi": dict(tip_sayac),  # kaçı sözlük, kaçı semantik buldu
        "ozet_metin": ozet_metin
    }


# ─── MODELLER ────────────────────────────────────────────────────────────────

class HarcamaIstegi(BaseModel):
    metin: str

class TopluHarcamaIstegi(BaseModel):
    harcamalar: List[str]


# ─── ENDPOİNTLER ─────────────────────────────────────────────────────────────

@app.post("/analiz")
async def analiz_et(istek: HarcamaIstegi):
    # 1. Sözlük
    sonuc = sozluk_analiz(istek.metin)
    if sonuc:
        return sonuc

    # 2. Semantik
    sonuc = semantic_analiz(istek.metin)
    if sonuc:
        return sonuc

    # 3. Hiçbiri tutmadı
    return {
        "cevap": f"Bilinmeyen işlem: {istek.metin}",
        "guven": 0,
        "tip": "Bilinmeyen",
        "kategori": "Diğer"
    }


@app.post("/toplu-analiz")
async def toplu_analiz_et(istek: TopluHarcamaIstegi):
    temiz_liste = [h.strip() for h in istek.harcamalar if h.strip()]

    if not temiz_liste:
        return {"hata": "Harcama listesi boş."}

    sonuclar = []

    for harcama in temiz_liste:
        # 1. Sözlük
        sonuc = sozluk_analiz(harcama)

        # 2. Semantik
        if not sonuc:
            sonuc = semantic_analiz(harcama)

        # 3. Bilinmeyen
        if not sonuc:
            sonuc = {
                "cevap": "Analiz edilemedi.",
                "guven": 0,
                "tip": "Bilinmeyen",
                "kategori": "Diğer"
            }

        sonuclar.append({**sonuc, "harcama": harcama})

    return {
        "sonuclar": sonuclar,
        "ozet": ozet_rapor_olustur(sonuclar)
    }


@app.post("/csv-analiz")
async def csv_analiz_et(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    potential_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_val = str(df[col].iloc[0])
            if len(sample_val) < 40 and not any(char.isdigit() for char in sample_val[:5]):
                potential_columns.append(col)

    if "Track Name" in df.columns:
        hedef_sutun = "Track Name"
    elif potential_columns:
        hedef_sutun = potential_columns[0]
    else:
        hedef_sutun = df.columns[0]

    harcamalar = df[hedef_sutun].astype(str).tolist()[:10]

    sonuclar = []
    for h in harcamalar:
        sonuc = sozluk_analiz(h)

        if not sonuc:
            sonuc = semantic_analiz(h)

        if not sonuc:
            sonuc = {
                "cevap": "Analiz edilemedi.",
                "guven": 0,
                "tip": "Bilinmeyen",
                "kategori": "Diğer"
            }

        sonuclar.append({**sonuc, "harcama": h})

    return {
        "sonuclar": sonuclar,
        "ozet": ozet_rapor_olustur(sonuclar)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)