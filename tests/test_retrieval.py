"""
Tests for retrieval module (BM25, embeddings, hybrid).
"""

import pytest
import numpy as np
from datetime import datetime
from core.retrieval.bm25_retriever import BM25Retriever
from core.retrieval.eval_metrics import precision_at_k, recall_at_k
from data_pipeline.ingestion import ITSMTicket
from data_pipeline.build_indexes import convert_ticket_to_document


class TestBM25Retriever:
    """Tests for BM25Retriever."""
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {"id": "1", "text": "How to reset password for email account"},
            {"id": "2", "text": "VPN connection troubleshooting guide"},
            {"id": "3", "text": "Password reset procedure for Windows"},
            {"id": "4", "text": "Email configuration on mobile devices"},
        ]
    
    def test_index_documents(self, sample_documents):
        """Test document indexing."""
        retriever = BM25Retriever()
        retriever.index_documents(sample_documents)
        
        assert retriever.bm25 is not None
        assert len(retriever.documents) == len(sample_documents)
    
    def test_search(self, sample_documents):
        """Test BM25 search."""
        retriever = BM25Retriever()
        retriever.index_documents(sample_documents)
        
        query = "password reset"
        results = retriever.search(query, top_k=2)
        
        assert len(results) <= 2
        assert all("score" in r for r in results)
        
        # Check that password-related docs are ranked higher
        if results:
            assert "password" in results[0]["text"].lower()


class TestEvalMetrics:
    """Tests for evaluation metrics."""
    
    def test_precision_at_k(self):
        """Test Precision@k calculation."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc2", "doc4", "doc6"}
        
        p_at_3 = precision_at_k(retrieved, relevant, k=3)
        assert p_at_3 == 1/3  # Only doc2 is relevant in top-3
        
        p_at_5 = precision_at_k(retrieved, relevant, k=5)
        assert p_at_5 == 2/5  # doc2 and doc4 are relevant in top-5
    
    def test_recall_at_k(self):
        """Test Recall@k calculation."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc2", "doc4", "doc6"}
        
        r_at_3 = recall_at_k(retrieved, relevant, k=3)
        assert r_at_3 == 1/3  # Found 1 out of 3 relevant docs
        
        r_at_5 = recall_at_k(retrieved, relevant, k=5)
        assert r_at_5 == 2/3  # Found 2 out of 3 relevant docs


class TestPhase3Integration:
    """Tests for PHASE 3: Ticket-to-Document conversion and retrieval integration."""
    
    def test_convert_ticket_to_document(self):
        """Test conversion of ITSMTicket to document format."""
        ticket = ITSMTicket(
            ticket_id="TCK-001",
            created_at=datetime(2025, 1, 10, 9, 0, 0),
            category="Uygulama",
            subcategory="Outlook",
            short_description="Outlook şifremi unuttum",
            description="Kullanıcı Outlook'a giriş yapamıyor",
            resolution="Şifre sıfırlandı",
            channel="email",
            priority="High",
            status="Closed"
        )
        
        doc = convert_ticket_to_document(ticket)
        
        # Check required fields for retrieval
        assert "text" in doc
        assert "id" in doc
        assert "ticket_id" in doc
        
        # Check that text combines all fields
        assert "Outlook şifremi unuttum" in doc["text"]
        assert "giriş yapamıyor" in doc["text"]
        assert "Çözüm: Şifre sıfırlandı" in doc["text"]
        
        # Check metadata preserved
        assert doc["ticket_id"] == "TCK-001"
        assert doc["id"] == "TCK-001"
        assert doc["category"] == "Uygulama"
        assert doc["subcategory"] == "Outlook"
        assert doc["priority"] == "High"
        assert doc["status"] == "Closed"
    
    def test_retrieval_with_converted_tickets(self):
        """Test that BM25Retriever works with converted tickets."""
        # Create sample tickets
        tickets = [
            ITSMTicket(
                ticket_id="TCK-001",
                created_at=datetime(2025, 1, 10, 9, 0, 0),
                short_description="Outlook şifremi unuttum",
                description="Kullanıcı Outlook'a giriş yapamıyor",
                resolution="Şifre sıfırlandı"
            ),
            ITSMTicket(
                ticket_id="TCK-002",
                created_at=datetime(2025, 1, 11, 10, 0, 0),
                short_description="VPN bağlantı sorunu",
                description="VPN'e bağlanamıyorum",
                resolution="VPN ayarları düzeltildi"
            ),
            ITSMTicket(
                ticket_id="TCK-003",
                created_at=datetime(2025, 1, 12, 11, 0, 0),
                short_description="Email gönderemiyorum",
                description="Outlook'tan email gönderilemedi",
                resolution="SMTP ayarları yapıldı"
            ),
        ]
        
        # Convert to documents
        documents = [convert_ticket_to_document(t) for t in tickets]
        
        # Index with BM25
        retriever = BM25Retriever()
        retriever.index_documents(documents, text_field="text")
        
        # Search for Outlook-related issues
        results = retriever.search("Outlook şifre", top_k=2)
        
        # Should find tickets with Outlook mentions
        assert len(results) > 0
        assert results[0]["ticket_id"] in ["TCK-001", "TCK-003"]
        
        # Search for VPN
        results = retriever.search("VPN bağlantı", top_k=1)
        assert len(results) > 0
        assert results[0]["ticket_id"] == "TCK-002"

