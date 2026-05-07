# Bitirme Projesi Haftalık İlerleme Raporu

## Proje Bilgileri

| Alan | Bilgi |
|------|-------|
| **Öğrenci Adı Soyadı** | Gamze Yarımkulak |
| **Öğrenci No** | 22360859036 |
| **Proje Başlığı** | Bağlamsal Farkındalıklı BT Destek Asistanı: Hibrit RAG ve Anomali Tespiti ile Güvenilir Yanıt Üretimi |
| **Danışman** | Prof. Dr. Turgay Tugay Bilgin |
| **Dönem** | 2025-2026 Bahar |

---

## İş Planı


| Hafta | Tarih Aralığı | Planlanan İş | Tahmini Tamamlanma (%) | Durum |
|-------|---------------|--------------|------------------------|-------|
| 1 | 01.04 - 05.04 | Bitirme projesi için public GitHub reposunun oluşturulması, proje kapsamının netleştirilmesi, haftalık planın hazırlanması ve temel klasör yapısının kurulması | %10 | ✅ Tamamlandı |
| 2 | 06.04 - 12.04 | Veri kaynaklarının düzenlenmesi, veri ön işleme adımlarının belirlenmesi ve anonimleştirme hattının oluşturulması | %20 | ✅ Tamamlandı |
| 3 | 13.04 - 19.04 | BM25 tabanlı kelime temelli arama yapısının eklenmesi ve örnek sorgularla ilk retrieval testlerinin yapılması | %30 | ✅ Tamamlandı |
| 4 | 27.04 - 03.05 | Embedding tabanlı semantik arama hattının geliştirilmesi ve vektör indeksleme altyapısının hazırlanması | %40 | ✅ Tamamlandı |
| 5 | 04.05 - 10.05 | Hibrit retrieval yapısının kurulması, kelime temelli ve semantik sonuçların birleştirilmesi, sıralama mantığının iyileştirilmesi | %52 | ⬜ Başlamadı |
| 6 | 11.05 - 17.05 | RAG/chat pipeline’ının geliştirilmesi, kaynaklı yanıt üretimi ve “kaynak yoksa cevap yok” mantığının eklenmesi | %65 | ⬜ Başlamadı |
| 7 | 18.05 - 24.05 | Anomali tespiti modülünün geliştirilmesi, semantik drift ve örnek anomali senaryolarının test edilmesi | %75 | ⬜ Başlamadı |
| 8 | 01.06 - 07.06 | API ve frontend entegrasyonunun yapılması, sohbet ekranı ve anomali panelinin birlikte çalışacak şekilde düzenlenmesi | %85 | ⬜ Başlamadı |
| 9 | 08.06 - 14.06 | Testlerin genişletilmesi, hata senaryolarının kontrol edilmesi, sistem performansının ve yanıt kalitesinin değerlendirilmesi | %93 | ⬜ Başlamadı |
| 10 | 15.06 - 21.06 | Dokümantasyonun tamamlanması, son hata düzeltmeleri, demo/sunum hazırlığı ve final teslim öncesi genel kontrol | %100 | ⬜ Başlamadı |

**Durum simgeleri:** ⬜ Başlamadı | 🔄 Devam Ediyor | ✅ Tamamlandı | ⚠️ Gecikti

---

## Haftalık İlerleme Kayıtları

### Hafta 4 *(Tarih: 27.04.2026 - 03.05.2026)*

**Plandaki hedef:**
- Embedding tabanlı semantik arama hattının geliştirilmesi
- Vektör indeksleme altyapısının hazırlanması

**Bu hafta yaptıklarım:**
- Embedding tabanlı semantik retrieval modülünü projeye ekledim
- SentenceTransformer kullanarak metinlerin vektör temsillerini oluşturdum
- FAISS tabanlı vektör indeksleme yapısını kurarak benzerlik aramasını gerçekleştirdim
- Retrieval performansını ölçmek için precision, recall, MAP ve nDCG metriklerini içeren değerlendirme modülünü ekledim
- Semantik arama ile kelime temelli arama arasındaki farkları analiz ederek sistem davranışını gözlemledim

**Plana göre durumum:**
- Hafta 4 hedefleri büyük ölçüde tamamlandı
- Semantik retrieval altyapısı oluşturuldu ve test edilebilir hale getirildi

**Karşılaştığım sorunlar / zorluklar:**
- Embedding boyutları ve indeksleme sürecinin performansa etkisi değerlendirildi
- Vektör arama sonuçlarının anlamlılığını test etmek için uygun örnek veri oluşturulması gerekti

**Gelecek hafta hedefim:**
- Hibrit retrieval yapısını kurmak (BM25 + embedding)
- Retrieval sonuçlarını birleştirme ve sıralama mantığını geliştirmek

**Kişisel değerlendirme:**
Bu hafta öğrendiğim yöntemler model performansını artırmak açısından oldukça faydalıydı.

### Hafta 3 *(Tarih: 13.04.2026 - 19.04.2026)*

**Plandaki hedef:**
- BM25 tabanlı kelime temelli arama yapısının eklenmesi
- Örnek sorgularla ilk retrieval testlerinin yapılması

**Bu hafta yaptıklarım:**
- BM25 tabanlı kelime temelli retrieval modülünü projeye ekledim
- Retrieval modülünün proje yapısına entegrasyonunu başlattım
- Retrieval performansını kontrol etmek için temel test dosyası ekledim
- BM25 arama mantığının örnek dokümanlar ve ticket dönüşümü üzerinden çalışmasını doğruladım

**Plana göre durumum:**
- Hafta 3 hedefleri büyük ölçüde tamamlandı
- Kelime temelli retrieval hattı oluşturuldu ve test aşaması başlatıldı

**Karşılaştığım sorunlar / zorluklar:**
- Sorgu ve doküman metinlerinin uygun şekilde işlenmesi için yapı gözden geçirildi
- Test senaryolarında örnek veri ile anlamlı sonuç üretimini dengelemek gerekti

**Gelecek hafta hedefim:**
- Embedding tabanlı semantik retrieval yapısını eklemek
- Vektör indeksleme altyapısını hazırlamak

---
### Hafta 2 *(Tarih: 06.04.2026 - 12.04.2026)*

**Plandaki hedef:**
- Veri kaynaklarının düzenlenmesi
- Veri ön işleme adımlarının belirlenmesi
- Anonimleştirme hattının oluşturulması

**Bu hafta yaptıklarım:**
- Örnek ticket veri seti oluşturdum
- Veri ön işleme sürecini planladım
- E-posta, telefon, IP ve isim bilgilerini maskeleyen anonimleştirme modülünü geliştirdim
- Veri pipeline yapısına ait dokümantasyon ekledim

**Plana göre durumum:**
- Hafta 2 hedeflerine ulaşıldı
- Retrieval aşamasına geçmek için veri hazırlama zemini oluşturuldu

**Karşılaştığım sorunlar / zorluklar:**
- Farklı veri tipleri için ortak anonimleştirme yaklaşımını sade tutmak gerekti
- Gerçek veri yerine örnek veri kullanımı planlandı

**Gelecek hafta hedefim:**
- BM25 tabanlı retrieval yapısı eklenecek
- İlk arama testleri yapılacak

---

### Hafta 1 *(Tarih: 01.04.2026 - 05.04.2026)*

**Plandaki hedef:**
- Bitirme projesi için public GitHub reposunun oluşturulması
- Proje kapsamının ve hedeflerinin netleştirilmesi
- 10 haftalık iş planının hazırlanması
- Temel klasör yapısının kurulması

**Bu hafta yaptıklarım:**
- Bitirme projesi için public GitHub reposu oluşturdum
- Proje başlığını, kapsamını ve genel hedeflerini netleştirdim
- Haftalık ilerleme takibi için `HAFTALIK_ILERLEME.md` dosyasını ekledim
- Projede kullanılacak temel klasör yapısını planladım ve oluşturmaya başladım
- Geliştirme sürecinde izlenecek 10 haftalık iş planını hazırladım

**Plana göre durumum:**
- İlk hafta için belirlenen hedefler tamamlandı
- Projenin geliştirme sürecini düzenli takip edebilmek için gerekli repo ve dokümantasyon altyapısı oluşturuldu

**Karşılaştığım sorunlar / zorluklar:**
- Projenin kapsamını haftalara dengeli biçimde dağıtmak için planlama yapılması gerekti
- Kullanılacak klasör yapısı ve geliştirme sırasını netleştirme aşamasında başlangıçta karar verilmesi gereken noktalar oldu

**Gelecek hafta hedefim:**
- Veri kaynaklarını düzenlemek
- Veri ön işleme adımlarını netleştirmek
- Anonimleştirme hattının ilk sürümünü oluşturmak

---

<!--
ŞABLON: Yeni hafta eklemek için aşağıdaki bloğu kopyalayıp üste yapıştırın.

### Hafta X *(Tarih: GG.AA.YYYY - GG.AA.YYYY)*

**Plandaki hedef:**
- 

**Bu hafta yaptıklarım:**
- 

**Plana göre durumum:**
- 

**Karşılaştığım sorunlar / zorluklar:**
- 

**Gelecek hafta hedefim:**
- 

---
-->
