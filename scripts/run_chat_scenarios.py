#!/usr/bin/env python3
"""
BT Support Assistant - End-to-End Senaryo Test Script'i

Bu script, BT Support Assistant'ın chat endpoint'ini gerçekçi IT destek sorularıyla test eder.
RAG pipeline'ın makul yanıtlar üretip üretmediğini ve kabul edilebilir güven skorlarına
sahip olup olmadığını doğrular.

Kullanım:
    1. Sunucuyu başlatın: python scripts/run_server.py
    2. Bu script'i çalıştırın: python scripts/run_chat_scenarios.py

Script şunları yapar:
- Önceden tanımlanmış soruları chat endpoint'ine gönderir
- Yanıtlarda beklenen anahtar kelimelerin bulunup bulunmadığını kontrol eder
- Minimum güven skoru eşiklerini doğrular
- Geçti/başarısız durumunu gösteren detaylı bir rapor yazdırır
"""

import sys
import requests
from typing import List, Dict, Any
from dataclasses import dataclass
from colorama import init, Fore, Style

# Colorama'yı renkli terminal çıktısı için başlat
# autoreset=True: Her satırdan sonra renkleri sıfırlar
init(autoreset=True)

# ============================================
# YAPILANDIRMA (CONFIGURATION)
# ============================================

# API sunucusunun temel URL'i
# Sunucu farklı bir portta çalışıyorsa burayı güncelleyin
API_BASE_URL = "http://localhost:8000"

# Chat endpoint'inin tam URL'i
CHAT_ENDPOINT = f"{API_BASE_URL}/api/v1/chat"

# İstek zaman aşımı (saniye cinsinden)
# Uzun sorgular için bu değeri artırabilirsiniz
REQUEST_TIMEOUT = 30  # saniye


# ============================================
# SCENARIO DEFINITIONS
# ============================================

@dataclass
class ChatScenario:
    """
    Chat endpoint'i için bir test senaryosunu temsil eder.
    
    Her senaryo şunları içerir:
    - name: Senaryonun açıklayıcı adı
    - question: Test edilecek kullanıcı sorusu
    - expected_keywords: Yanıtta bulunması beklenen anahtar kelimeler (büyük/küçük harf duyarsız)
    - min_confidence: Minimum kabul edilebilir güven skoru (0.0-1.0 arası)
    - language: Soru dili (varsayılan: "tr" - Türkçe)
    """
    name: str
    question: str
    expected_keywords: List[str]  # Yanıtta görünmesi beklenen anahtar kelimeler (büyük/küçük harf duyarsız)
    min_confidence: float  # Minimum güven skoru eşiği
    language: str = "tr"  # Soru dili
    
    def __str__(self):
        """Senaryoyu okunabilir string formatında döndürür."""
        return f"[{self.name}] {self.question}"


# ============================================
# TEST SENARYOLARI
# ============================================
# Gerçekçi IT destek senaryolarını tanımlar
# Her senaryo için beklenen anahtar kelimeler ve minimum güven skoru belirtilir

SCENARIOS = [
    ChatScenario(
        name="Outlook Şifre Sıfırlama",
        question="Outlook şifremi unuttum, nasıl sıfırlarım?",
        expected_keywords=["outlook", "parola", "şifre", "sıfırlama", "bağlantı"],
        min_confidence=0.4,
    ),
    
    ChatScenario(
        name="VPN Bağlantı Sorunu",
        question="VPN'e bağlanamıyorum, ne yapmalıyım?",
        expected_keywords=["vpn", "bağlantı", "ayar", "istemci", "kimlik"],
        min_confidence=0.4,
    ),
    
    ChatScenario(
        name="Yazıcı Yazdırmıyor",
        question="Yazıcı yazdırmıyor, nasıl düzeltebilirim?",
        expected_keywords=["yazıcı", "sürücü", "bağlantı", "ayar"],
        min_confidence=0.3,
    ),
    
    ChatScenario(
        name="Laptop Yavaş Çalışıyor",
        question="Laptop çok yavaş çalışıyor, ne yapmalıyım?",
        expected_keywords=["performans", "disk", "bellek", "güncelleme", "temizlik"],
        min_confidence=0.3,
    ),
    
    ChatScenario(
        name="Email Gönderemiyorum",
        question="Email gönderemiyorum, hata veriyor",
        expected_keywords=["email", "mail", "gönder", "ayar", "sunucu"],
        min_confidence=0.3,
    ),
    
    ChatScenario(
        name="Disk Dolu Hatası",
        question="Disk alanı doldu hatası alıyorum",
        expected_keywords=["disk", "alan", "temizlik", "dosya", "silme"],
        min_confidence=0.35,
    ),
]


# ============================================
# TEST EXECUTION
# ============================================

def check_server_health() -> bool:
    """
    Sunucunun çalışıp çalışmadığını ve sağlıklı olup olmadığını kontrol eder.
    
    Returns:
        True: Sunucu çalışıyor ve sağlıklı
        False: Sunucu çalışmıyor veya erişilemiyor
    """
    try:
        health_url = f"{API_BASE_URL}/api/v1/health"
        response = requests.get(health_url, timeout=5)
        return response.status_code == 200
    except Exception:
        # Herhangi bir hata durumunda (bağlantı hatası, zaman aşımı, vb.) False döndür
        return False


def send_chat_request(scenario: ChatScenario) -> Dict[str, Any]:
    """
    API endpoint'ine bir chat isteği gönderir.
    
    Args:
        scenario: Test edilecek chat senaryosu
        
    Returns:
        API'den dönen JSON yanıtı (answer, confidence, sources, vb. içerir)
        
    Raises:
        requests.exceptions.HTTPError: HTTP hatası durumunda (örn: 500, 404)
        requests.exceptions.Timeout: İstek zaman aşımına uğrarsa
        requests.exceptions.ConnectionError: Sunucuya bağlanılamazsa
    """
    # API'ye gönderilecek payload'ı hazırla
    payload = {
        "query": scenario.question,  # Kullanıcı sorusu
        "language": scenario.language,  # Soru dili
    }
    
    # POST isteği gönder
    response = requests.post(
        CHAT_ENDPOINT,
        json=payload,  # JSON formatında gönder
        timeout=REQUEST_TIMEOUT,  # Zaman aşımı süresi
    )
    
    # HTTP hata kodlarını kontrol et (4xx, 5xx)
    response.raise_for_status()
    
    # JSON yanıtı parse et ve döndür
    return response.json()


def check_keywords_in_text(text: str, keywords: List[str]) -> Dict[str, bool]:
    """
    Metinde hangi anahtar kelimelerin bulunduğunu kontrol eder (büyük/küçük harf duyarsız).
    
    Args:
        text: Kontrol edilecek metin
        keywords: Aranacak anahtar kelimeler listesi
        
    Returns:
        Her anahtar kelime için True/False değerleri içeren dictionary
        Örnek: {"outlook": True, "şifre": False, "sıfırlama": True}
    """
    # Metni küçük harfe çevir (büyük/küçük harf duyarsız arama için)
    text_lower = text.lower()
    
    # Her anahtar kelimeyi kontrol et ve sonuçları dictionary olarak döndür
    return {keyword: keyword.lower() in text_lower for keyword in keywords}


def evaluate_scenario(scenario: ChatScenario, response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    API yanıtının senaryo beklentilerini karşılayıp karşılamadığını değerlendirir.
    
    Değerlendirme kriterleri:
    1. Güven skoru minimum eşik değerinden yüksek olmalı
    2. Yanıtta beklenen anahtar kelimelerin en az %50'si bulunmalı
    
    Args:
        scenario: Test edilen senaryo
        response_data: API'den dönen yanıt verisi
        
    Returns:
        Değerlendirme sonuçlarını içeren dictionary:
        - passed: Genel geçti/başarısız durumu
        - confidence: Gerçek güven skoru
        - confidence_ok: Güven skoru eşiği kontrolü
        - keyword_results: Her anahtar kelime için bulundu/bulunamadı
        - keywords_found: Bulunan anahtar kelime sayısı
        - keywords_total: Toplam beklenen anahtar kelime sayısı
        - keyword_ratio: Bulunan anahtar kelime oranı (0.0-1.0)
        - keywords_ok: Anahtar kelime eşiği kontrolü
        - num_sources: Kaynak doküman sayısı
        - answer_length: Yanıt uzunluğu (karakter sayısı)
    """
    # API yanıtından gerekli alanları çıkar
    answer = response_data.get("answer", "")
    confidence = response_data.get("confidence", 0.0)
    sources = response_data.get("sources", [])
    
    # Anahtar kelime kontrolü: Yanıtta beklenen kelimelerin bulunup bulunmadığını kontrol et
    keyword_results = check_keywords_in_text(answer, scenario.expected_keywords)
    keywords_found = sum(keyword_results.values())  # Bulunan kelime sayısı
    keywords_total = len(scenario.expected_keywords)  # Toplam beklenen kelime sayısı
    keyword_ratio = keywords_found / keywords_total if keywords_total > 0 else 0  # Bulunma oranı
    
    # Güven skoru kontrolü: Minimum eşik değerinden yüksek olmalı
    confidence_ok = confidence >= scenario.min_confidence
    
    # Anahtar kelime eşiği kontrolü: En az %50'si bulunmalı
    keywords_ok = keyword_ratio >= 0.5
    
    # Genel geçti/başarısız durumu: Her iki kriter de sağlanmalı
    passed = confidence_ok and keywords_ok
    
    return {
        "passed": passed,
        "confidence": confidence,
        "confidence_ok": confidence_ok,
        "keyword_results": keyword_results,
        "keywords_found": keywords_found,
        "keywords_total": keywords_total,
        "keyword_ratio": keyword_ratio,
        "keywords_ok": keywords_ok,
        "num_sources": len(sources),
        "answer_length": len(answer),
    }


def print_scenario_result(scenario: ChatScenario, evaluation: Dict[str, Any]):
    """
    Tek bir senaryonun sonuçlarını renkli ve formatlanmış şekilde yazdırır.
    
    Çıktı formatı:
    - Senaryo adı ve durum ikonu (✅ veya ❌)
    - Kullanıcı sorusu
    - Güven skoru ve eşik değeri
    - Anahtar kelime bulunma durumu
    - Her anahtar kelime için ayrı ayrı bulundu/bulunamadı durumu
    - Kaynak doküman sayısı
    - Yanıt uzunluğu
    
    Args:
        scenario: Test edilen senaryo
        evaluation: Değerlendirme sonuçları dictionary'si
    """
    # Başlık: Senaryo adı ve durum ikonu
    status_icon = "✅" if evaluation["passed"] else "❌"
    status_color = Fore.GREEN if evaluation["passed"] else Fore.RED
    
    print(f"\n{status_color}{status_icon} {scenario.name}{Style.RESET_ALL}")
    print(f"   Question: {Fore.CYAN}{scenario.question}{Style.RESET_ALL}")
    
    # Güven skoru bilgisi
    conf = evaluation["confidence"]
    min_conf = scenario.min_confidence
    conf_status = "✓" if evaluation["confidence_ok"] else "✗"
    conf_color = Fore.GREEN if evaluation["confidence_ok"] else Fore.YELLOW
    print(f"   Confidence: {conf_color}{conf:.2f}{Style.RESET_ALL} (threshold: {min_conf:.2f}) {conf_status}")
    
    # Anahtar kelime özeti
    kw_found = evaluation["keywords_found"]
    kw_total = evaluation["keywords_total"]
    kw_ratio = evaluation["keyword_ratio"]
    kw_status = "✓" if evaluation["keywords_ok"] else "✗"
    kw_color = Fore.GREEN if evaluation["keywords_ok"] else Fore.YELLOW
    print(f"   Keywords: {kw_color}{kw_found}/{kw_total} ({kw_ratio:.0%}){Style.RESET_ALL} {kw_status}")
    
    # Her anahtar kelime için ayrı ayrı durum gösterimi
    keyword_results = evaluation["keyword_results"]
    keyword_strs = []
    for keyword, found in keyword_results.items():
        if found:
            keyword_strs.append(f"{Fore.GREEN}{keyword} ✓{Style.RESET_ALL}")
        else:
            keyword_strs.append(f"{Fore.RED}{keyword} ✗{Style.RESET_ALL}")
    print(f"             {', '.join(keyword_strs)}")
    
    # Ek bilgiler: Kaynak sayısı ve yanıt uzunluğu
    print(f"   Sources: {evaluation['num_sources']} documents")
    print(f"   Answer length: {evaluation['answer_length']} chars")


def print_summary(results: List[Dict[str, Any]]):
    """
    Tüm senaryoların genel özetini yazdırır.
    
    Özet şunları içerir:
    - Toplam senaryo sayısı
    - Geçen/başarısız senaryo sayıları
    - Geçme oranı (pass rate)
    - Genel durum değerlendirmesi
    
    Args:
        results: Tüm senaryoların değerlendirme sonuçları listesi
    """
    # İstatistikleri hesapla
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    pass_rate = passed / total if total > 0 else 0
    
    # Özet başlığı
    print("\n" + "=" * 70)
    print(f"{Fore.CYAN}SUMMARY{Style.RESET_ALL}")
    print("=" * 70)
    
    # İstatistikleri yazdır
    print(f"Total scenarios: {total}")
    print(f"{Fore.GREEN}Passed: {passed}{Style.RESET_ALL}")
    print(f"{Fore.RED}Failed: {failed}{Style.RESET_ALL}")
    print(f"Pass rate: {pass_rate:.0%}")
    
    # Genel durum değerlendirmesi
    if pass_rate >= 0.8:
        # %80 ve üzeri geçme oranı → İyi durum
        print(f"\n{Fore.GREEN}✅ Overall status: GOOD - Most scenarios passed{Style.RESET_ALL}")
    elif pass_rate >= 0.5:
        # %50-80 arası geçme oranı → Kabul edilebilir, bazı iyileştirmeler gerekebilir
        print(f"\n{Fore.YELLOW}⚠️  Overall status: ACCEPTABLE - Some scenarios need attention{Style.RESET_ALL}")
    else:
        # %50'nin altında geçme oranı → Kötü durum, ciddi iyileştirmeler gerekli
        print(f"\n{Fore.RED}❌ Overall status: POOR - Many scenarios failing{Style.RESET_ALL}")


def run_all_scenarios():
    """
    Tüm önceden tanımlanmış senaryoları çalıştırır ve sonuçları yazdırır.
    
    İşlem adımları:
    1. Sunucu sağlık kontrolü yapar
    2. Her senaryoyu sırayla test eder
    3. Her senaryo için detaylı sonuç yazdırır
    4. Genel özet raporu yazdırır
    5. Exit code ile çıkar (0: tüm testler geçti, 1: bazı testler başarısız)
    """
    # Başlık banner'ı
    print(f"{Fore.CYAN}╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  BT Support Assistant - End-to-End Scenario Tests               ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    
    # Sunucu sağlık kontrolü: Sunucunun çalışıp çalışmadığını kontrol et
    print(f"\n{Fore.YELLOW}🔍 Checking server health...{Style.RESET_ALL}")
    if not check_server_health():
        print(f"{Fore.RED}❌ ERROR: Server is not responding at {API_BASE_URL}{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Please start the server first:{Style.RESET_ALL}")
        print(f"   python scripts/run_server.py")
        sys.exit(1)  # Hata durumunda çık
    
    print(f"{Fore.GREEN}✅ Server is running{Style.RESET_ALL}")
    
    # Senaryoları çalıştır
    print(f"\n{Fore.YELLOW}🚀 Running {len(SCENARIOS)} scenarios...{Style.RESET_ALL}")
    
    results = []  # Tüm senaryoların sonuçlarını saklamak için liste
    
    # Her senaryoyu sırayla test et
    for i, scenario in enumerate(SCENARIOS, 1):
        print(f"\n{Fore.CYAN}[{i}/{len(SCENARIOS)}] Testing: {scenario.name}...{Style.RESET_ALL}")
        
        try:
            # API'ye istek gönder
            response_data = send_chat_request(scenario)
            
            # Senaryoyu değerlendir
            evaluation = evaluate_scenario(scenario, response_data)
            results.append(evaluation)
            
            # Sonuçları yazdır
            print_scenario_result(scenario, evaluation)
            
        except requests.exceptions.RequestException as e:
            # HTTP/network hataları (bağlantı hatası, zaman aşımı, vb.)
            print(f"{Fore.RED}❌ Request failed: {e}{Style.RESET_ALL}")
            results.append({
                "passed": False,
                "confidence": 0.0,
                "confidence_ok": False,
                "keywords_found": 0,
                "keywords_total": len(scenario.expected_keywords),
                "keyword_ratio": 0.0,
                "keywords_ok": False,
                "num_sources": 0,
                "answer_length": 0,
            })
        except Exception as e:
            # Beklenmeyen diğer hatalar
            print(f"{Fore.RED}❌ Unexpected error: {e}{Style.RESET_ALL}")
            results.append({
                "passed": False,
                "confidence": 0.0,
                "confidence_ok": False,
                "keywords_found": 0,
                "keywords_total": len(scenario.expected_keywords),
                "keyword_ratio": 0.0,
                "keywords_ok": False,
                "num_sources": 0,
                "answer_length": 0,
            })
    
    # Genel özeti yazdır
    print_summary(results)
    
    # Exit code belirle: Tüm testler geçtiyse 0, aksi halde 1
    failed_count = sum(1 for r in results if not r["passed"])
    sys.exit(0 if failed_count == 0 else 1)


# ============================================
# ANA PROGRAM (MAIN)
# ============================================

if __name__ == "__main__":
    """
    Script doğrudan çalıştırıldığında (import edilmediğinde) bu blok çalışır.
    """
    try:
        # Tüm senaryoları çalıştır
        run_all_scenarios()
    except KeyboardInterrupt:
        # Kullanıcı Ctrl+C ile iptal ederse
        print(f"\n\n{Fore.YELLOW}⚠️  Test interrupted by user{Style.RESET_ALL}")
        sys.exit(130)  # Keyboard interrupt için standart exit code



