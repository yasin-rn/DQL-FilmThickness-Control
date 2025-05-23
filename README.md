# 💎 DQL ile Plastik Film Ekstrüzyon Makinesi Kontrolü

## 🎯 Proje Hedefi
Plastik blow film ekstrüzyon makinesinde, Derin Q-Öğrenme (DQL) tabanlı bir yapay zeka ajanı geliştirerek film kalınlık profilinin 360 derece boyunca homojen (eşit kalınlıkta) hale getirilmesi, üretim kalitesinin ve verimliliğinin optimize edilmesi.

---

## 🧩 Adım Adım Uygulama Talimatları

### Faz 1: Sistem Analizi ve DQL Temellerinin Tanımlanması

#### 🔍 Sistemi Anlamak
- Ekstrüzyon makinesi çıkışında, hava halkası ile soğutulan balon film kullanılır.
- Hava halkasında 48 servo motor bulunur.
- Kalınlık profili sensör ile 360° taranır.

#### 📊 DQL Ajanı için Durum (State)
- `actuator_positions`: 48 motor pozisyonu [48]
- `average_thickness_per_actuator`: Her motor bölgesi için ortalama kalınlık [48]
- `thickness_profile`: Tam 360 derece kalınlık vektörü [360]
- Nihai durum vektörü: [5, 456] (son 5 zaman adımı birleştirilmiş)

#### ⚙️ Aksiyon Uzayı
- Her aktüatör için 3 aksiyon: `stop`, `up`, `down`
- Toplam Q-değeri çıktısı: 48 x 3 = 144
- Aksiyon seçimi: Her aktüatör için argmax(Q-values)

#### 🏆 Ödül Fonksiyonu
- Ana hedef: Kalınlık profilini homojenleştirmek
- Ödül önerileri:
  - `reward = -std_dev`
  - `reward = 1 / (1 + std_dev)`
  - `2Sigma` kademeli ödül
  - Hibrit: yoğun + kademeli ödül
  - Ek: Ortalama sapma cezası, aksiyon maliyeti

---

### Faz 2: Q-Network Mimarisi ve Ön Eğitim

#### 🧠 Q-Network Mimarisi (Transformer Encoder)
- Giriş: [5, 456]
- Çıkış: 144 Q-değeri
- Pozisyonel encoding:
  - Zamansal: 5 zaman adımı için
  - Özellik içi: actuator, kalınlık, tam profil sıralamaları

#### 🎓 Ön Eğitim (Behavior Cloning)
- Mevcut kontrolcü ile veri toplanır (state, action)
- Action: 48 x [0,1,0] gibi one-hot temsil
- Kayıp fonksiyonu: Cross Entropy Loss
- Transformer model denetimli öğrenme ile eğitilir
- Gerçek sistemle doğrulama: 2sigma iyileşmesine bakılır

---

### Faz 3: DQL Eğitimi, Değerlendirme ve Dağıtım

#### 🧪 DQL Eğitimi
- Ön-eğitimli ağırlıklarla başlatılır
- DQL çevrimiçi öğrenir (örneğin simülasyonda)
- Exploration (epsilon-greedy), replay buffer vs.

#### 📈 Performans Değerlendirme
- Metrikler: 2sigma, ortalama kalınlık, sapmalar
- İyileştirmeler: Hiperparametreler, ödül fonksiyonu, ağ mimarisi

---

## 📝 Ek Notlar ve İpuçları
- 📄 Belgeleme: Tüm varsayımlar, parametreler, sonuçlar detaylandırılmalı
- 🧪 Simülasyon: Gerçek sisteme geçmeden önce güvenli test ortamı önerilir
- 🔒 Güvenlik: Gerçek makinede test öncesi tüm önlemler alınmalı
- 🔁 Iteratif geliştirme: Küçük adımlar, sık test, sürekli iyileştirme

---

## 📚 Kaynaklar
*Projenin detaylı teknik kaynakları ve literatür incelemeleri ayrı dokümantasyonlarda belirtilmelidir.*

---
