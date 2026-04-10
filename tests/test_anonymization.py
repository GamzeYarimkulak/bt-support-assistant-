"""
Tests for data anonymization.
"""

import pytest
from datetime import datetime
from data_pipeline.anonymize import DataAnonymizer, anonymize_text, anonymize_ticket, anonymize_tickets
from data_pipeline.ingestion import ITSMTicket


class TestDataAnonymizer:
    """Tests for DataAnonymizer."""
    
    def test_anonymize_email(self):
        """Test email anonymization."""
        anonymizer = DataAnonymizer(anonymization_enabled=True)
        
        text = "Contact me at john.doe@example.com for help"
        anonymized = anonymizer.anonymize_text(text)
        
        assert "john.doe@example.com" not in anonymized
        assert "[EMAIL]" in anonymized
    
    def test_anonymize_phone(self):
        """Test phone number anonymization."""
        anonymizer = DataAnonymizer(anonymization_enabled=True)
        
        text = "Call support at 555-123-4567"
        anonymized = anonymizer.anonymize_text(text)
        
        assert "555-123-4567" not in anonymized
        assert "[PHONE]" in anonymized
    
    def test_anonymize_ip(self):
        """Test IP address anonymization."""
        anonymizer = DataAnonymizer(anonymization_enabled=True)
        
        text = "Server IP is 192.168.1.100"
        anonymized = anonymizer.anonymize_text(text)
        
        assert "192.168.1.100" not in anonymized
        assert "[IP_ADDRESS]" in anonymized
    
    def test_detect_pii(self):
        """Test PII detection."""
        anonymizer = DataAnonymizer()
        
        text = "Email: test@example.com, Phone: 555-1234, IP: 10.0.0.1"
        detected = anonymizer.detect_pii(text)
        
        assert "emails" in detected
        assert "phones" in detected
        assert "ips" in detected
    
    def test_anonymize_ticket(self):
        """Test ticket anonymization."""
        anonymizer = DataAnonymizer(anonymization_enabled=True)
        
        ticket = {
            "id": "T123",
            "title": "Password reset needed",
            "description": "User john.doe@example.com cannot login. Contact at 555-1234.",
            "email": "john.doe@example.com"
        }
        
        anonymized = anonymizer.anonymize_ticket(ticket)
        
        assert anonymized["id"] == "T123"  # ID preserved
        assert "[EMAIL]" in anonymized["description"]
        assert "[PHONE]" in anonymized["description"]
        assert "john.doe@example.com" not in str(anonymized)


class TestStandaloneFunctions:
    """Tests for standalone anonymization functions (PHASE 2)."""
    
    def test_anonymize_text_email(self):
        """Test that anonymize_text replaces email addresses."""
        text = "Contact user at ahmet@example.com for help"
        result = anonymize_text(text)
        
        assert "ahmet@example.com" not in result
        assert "[EMAIL]" in result
        assert "Contact user at [EMAIL] for help" == result
    
    def test_anonymize_text_phone(self):
        """Test that anonymize_text replaces phone numbers."""
        # Turkish phone number format
        text = "Telefon: 0555 123 4567"
        result = anonymize_text(text)
        
        assert "0555 123 4567" not in result
        assert "[PHONE]" in result
    
    def test_anonymize_text_turkish_phone(self):
        """Test Turkish phone number with +90 country code."""
        text = "İletişim: +90 555 123 4567"
        result = anonymize_text(text)
        
        assert "+90 555 123 4567" not in result
        assert "[PHONE]" in result
    
    def test_anonymize_text_ip_address(self):
        """Test that anonymize_text replaces IP addresses."""
        text = "Server IP: 192.168.1.100"
        result = anonymize_text(text)
        
        assert "192.168.1.100" not in result
        assert "[IP]" in result
    
    def test_anonymize_text_simple_name(self):
        """Test that anonymize_text replaces simple person names."""
        text = "Kullanıcı Ahmet Yılmaz şifresini unutmuş"
        result = anonymize_text(text)
        
        assert "Ahmet Yılmaz" not in result
        assert "[NAME]" in result
        assert "şifresini unutmuş" in result  # Turkish text preserved
    
    def test_anonymize_text_multiple_pii(self):
        """Test that anonymize_text handles multiple PII types."""
        text = "User Mehmet Öz (mehmet@example.com) from IP 10.0.0.5 called 0555-999-8877"
        result = anonymize_text(text)
        
        # All PII should be replaced
        assert "Mehmet Öz" not in result
        assert "mehmet@example.com" not in result
        assert "10.0.0.5" not in result
        assert "0555-999-8877" not in result
        
        # Tokens should be present
        assert "[NAME]" in result
        assert "[EMAIL]" in result
        assert "[IP]" in result
        assert "[PHONE]" in result
    
    def test_anonymize_text_preserves_turkish_chars(self):
        """Test that Turkish characters are preserved in non-PII text."""
        text = "Şifre sıfırlama işlemi başarıyla tamamlandı. Çok teşekkür ederim."
        result = anonymize_text(text)
        
        # Turkish characters should be intact
        assert "Şifre" in result
        assert "sıfırlama" in result
        assert "işlemi" in result
        assert "başarıyla" in result
        assert "tamamlandı" in result
        assert "Çok" in result
        assert "teşekkür" in result
    
    def test_anonymize_text_empty_string(self):
        """Test that empty string is handled correctly."""
        assert anonymize_text("") == ""
        assert anonymize_text(None) is None
    
    def test_anonymize_ticket_anonymizes_text_fields(self):
        """Test that anonymize_ticket anonymizes text fields."""
        ticket = ITSMTicket(
            ticket_id="TCK-001",
            created_at=datetime(2025, 1, 10, 9, 0, 0),
            category="Uygulama",
            subcategory="Email",
            short_description="Kullanıcı ahmet.yilmaz@example.com giriş yapamıyor",
            description="Ahmet Yılmaz adlı kullanıcı sistemde oturum açamıyor. IP: 192.168.1.50",
            resolution="Şifre sıfırlandı ve yeni.sifre@example.com adresine gönderildi.",
            channel="email",
            priority="High",
            status="Closed"
        )
        
        anonymized = anonymize_ticket(ticket)
        
        # Text fields should be anonymized
        assert "[EMAIL]" in anonymized.short_description
        assert "ahmet.yilmaz@example.com" not in anonymized.short_description
        
        assert "[NAME]" in anonymized.description
        assert "[IP]" in anonymized.description
        assert "Ahmet Yılmaz" not in anonymized.description
        assert "192.168.1.50" not in anonymized.description
        
        assert "[EMAIL]" in anonymized.resolution
        assert "yeni.sifre@example.com" not in anonymized.resolution
    
    def test_anonymize_ticket_preserves_metadata(self):
        """Test that anonymize_ticket preserves non-text fields."""
        original = ITSMTicket(
            ticket_id="TCK-999",
            created_at=datetime(2025, 1, 15, 14, 30, 0),
            category="Network",
            subcategory="VPN",
            short_description="VPN bağlantı sorunu",
            description="Kullanıcı VPN'e bağlanamıyor",
            resolution="Çözüldü",
            channel="phone",
            priority="Medium",
            status="Resolved"
        )
        
        anonymized = anonymize_ticket(original)
        
        # Metadata should be unchanged
        assert anonymized.ticket_id == "TCK-999"
        assert anonymized.created_at == datetime(2025, 1, 15, 14, 30, 0)
        assert anonymized.category == "Network"
        assert anonymized.subcategory == "VPN"
        assert anonymized.channel == "phone"
        assert anonymized.priority == "Medium"
        assert anonymized.status == "Resolved"
    
    def test_anonymize_ticket_returns_new_object(self):
        """Test that anonymize_ticket returns a new immutable object."""
        original = ITSMTicket(
            ticket_id="TCK-100",
            created_at=datetime(2025, 1, 1, 10, 0, 0),
            short_description="Test ticket with email@example.com",
            description="Description"
        )
        
        anonymized = anonymize_ticket(original)
        
        # Should be different objects
        assert original is not anonymized
        
        # Original should be unchanged
        assert "email@example.com" in original.short_description
        
        # Anonymized should have token
        assert "[EMAIL]" in anonymized.short_description
    
    def test_anonymize_tickets_batch(self):
        """Test that anonymize_tickets processes a list correctly."""
        tickets = [
            ITSMTicket(
                ticket_id=f"TCK-{i}",
                created_at=datetime(2025, 1, i, 10, 0, 0),
                short_description=f"User user{i}@example.com needs help",
                description="Details"
            )
            for i in range(1, 4)
        ]
        
        anonymized_list = anonymize_tickets(tickets)
        
        # Should have same length
        assert len(anonymized_list) == 3
        
        # All should be anonymized
        for anon_ticket in anonymized_list:
            assert "[EMAIL]" in anon_ticket.short_description
            assert "@example.com" not in anon_ticket.short_description

