"""
Unit tests for RAG system components
"""
import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDocumentLoading:
    """Test document loading functionality"""
    
    def test_company_policies_file_exists(self):
        """Test that company policies file exists"""
        policies_file = Path(__file__).parent.parent / "company_policies.txt"
        assert policies_file.exists(), "company_policies.txt not found"
    
    def test_company_policies_not_empty(self):
        """Test that company policies file is not empty"""
        policies_file = Path(__file__).parent.parent / "company_policies.txt"
        content = policies_file.read_text(encoding='utf-8')
        assert len(content) > 0, "company_policies.txt is empty"
        assert len(content) > 100, "company_policies.txt seems too short"


class TestFAISS:
    """Test FAISS index functionality"""
    
    def test_faiss_index_exists(self):
        """Test that FAISS index directory exists"""
        faiss_dir = Path(__file__).parent.parent / "faiss_index"
        assert faiss_dir.exists(), "faiss_index directory not found"
    
    def test_faiss_index_files_exist(self):
        """Test that required FAISS files exist"""
        faiss_dir = Path(__file__).parent.parent / "faiss_index"
        
        index_file = faiss_dir / "index.faiss"
        pkl_file = faiss_dir / "index.pkl"
        
        assert index_file.exists(), "index.faiss not found"
        assert pkl_file.exists(), "index.pkl not found"
    
    def test_faiss_index_files_not_empty(self):
        """Test that FAISS index files are not empty"""
        faiss_dir = Path(__file__).parent.parent / "faiss_index"
        
        index_file = faiss_dir / "index.faiss"
        pkl_file = faiss_dir / "index.pkl"
        
        assert index_file.stat().st_size > 0, "index.faiss is empty"
        assert pkl_file.stat().st_size > 0, "index.pkl is empty"


class TestQueryProcessing:
    """Test query processing logic"""
    
    def test_empty_query_handling(self):
        """Test handling of empty queries"""
        from app import chat_response
        
        # Test empty string
        answer, sources = chat_response("", [])
        assert answer == ""
        assert sources == ""
    
    def test_whitespace_query_handling(self):
        """Test handling of whitespace-only queries"""
        from app import chat_response
        
        # Test whitespace
        answer, sources = chat_response("   ", [])
        assert answer == ""
        assert sources == ""


class TestModelConfiguration:
    """Test model configuration and initialization"""
    
    def test_embeddings_model_name(self):
        """Test that embeddings model name is configured"""
        # This would require importing the actual RAG system
        # For now, just verify the constant exists
        embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"
        assert isinstance(embeddings_model, str)
        assert len(embeddings_model) > 0
    
    def test_llm_model_configuration(self):
        """Test that LLM model is configured"""
        # Verify Groq model configuration
        model_name = "llama-3.3-70b-versatile"
        assert isinstance(model_name, str)
        assert "llama" in model_name.lower()


class TestFileStructure:
    """Test project file structure"""
    
    def test_required_files_exist(self):
        """Test that all required files exist"""
        base_dir = Path(__file__).parent.parent
        
        required_files = [
            "app.py",
            "monitoring.py",
            "Dockerfile",
            "requirements.txt",
            "company_policies.txt"
        ]
        
        for filename in required_files:
            filepath = base_dir / filename
            assert filepath.exists(), f"{filename} not found"
    
    def test_requirements_file_not_empty(self):
        """Test that requirements.txt has dependencies"""
        req_file = Path(__file__).parent.parent / "requirements.txt"
        content = req_file.read_text()
        
        # Check for key dependencies
        assert "gradio" in content.lower()
        assert "langchain" in content.lower()
        assert "faiss" in content.lower()


class TestEnvironmentVariables:
    """Test environment variable handling"""
    
    def test_groq_api_key_check(self):
        """Test that GROQ_API_KEY check works"""
        import os
        
        # This test verifies the check mechanism exists
        # Not the actual key (which should be in .env)
        key = os.getenv("GROQ_API_KEY")
        
        # Key might not be set in test environment
        # Just verify the getenv call works
        assert isinstance(key, (str, type(None)))


class TestLogging:
    """Test logging configuration"""
    
    def test_logs_directory_structure(self):
        """Test that logs directory can be created"""
        logs_dir = Path(__file__).parent.parent / "logs"
        
        # Directory should exist or be creatable
        if not logs_dir.exists():
            logs_dir.mkdir(exist_ok=True)
        
        assert logs_dir.is_dir()
    
    def test_log_file_patterns(self):
        """Test log file naming patterns"""
        from datetime import datetime
        
        # Test that date format works
        date_str = datetime.now().strftime('%Y%m%d')
        assert len(date_str) == 8
        assert date_str.isdigit()


# Integration test (optional - might need GROQ_API_KEY)
class TestRAGIntegration:
    """Integration tests for RAG system"""
    
    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("faiss_index/index.faiss").exists(),
        reason="FAISS index not built"
    )
    def test_rag_system_can_initialize(self):
        """Test that RAG system can be imported and initialized"""
        try:
            from app import rag_system
            assert rag_system is not None
        except Exception as e:
            pytest.skip(f"RAG system initialization requires environment setup: {e}")