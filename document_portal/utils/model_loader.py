import os 
import sys
from dotenv import load_dotenv
from .config_loader import load_config
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
log = CustomLogger().get_logger(__name__)

class ModelLoader:
    """
    Utility class to load embedding and LLM models
    """

    def __init__(self):
        load_dotenv()
        self._validate_env()
        self.config=load_config()
        log.info("config loaded successfully", config_keys = list(self.config.keys()))

    def _validate_env(self):
        """
        validate necessary environment variables
        Ensure API keys exists
        """
        required_vars = ["GOOGLE_API_KEY", "GROQ_API_KEY"]
        self.api_keys = {var: os.getenv(var) for var in required_vars}
        missing_vars = [k for k,v in self.api_keys.items() if not v]
        if missing_vars:
            log.error("missing environment variables", missing_vars = missing_vars)
            raise DocumentPortalException("missing env variables", sys)
        log.info("env variables validated successfully", available_keys={k for k in self.api_keys if self.api_keys[k]})


    def load_embeddings(self):
        """
        load & return embedding model
        """
        try:
            log.info("loading embedding model")
            model_name = self.config["embedding_model"]["model_name"]
            return GoogleGenerativeAIEmbeddings(model=model_name)
        except Exception as e:
            log.error("failed to load embedding model", error=str(e))
            raise DocumentPortalException("failed to load embedding model", sys)
        
    def load_llm(self):
        """
        load & return embedding model
        """
        try:
            log.info("loading LLM model")
            llm_block = self.config["llm"]
            log.info("loading llm...")

        except Exception as e:
            log.error("failed to load LLM model", error=str(e))
            raise DocumentPortalException("failed to load LLM model", sys)
        
        provider_key = os.getenv("provider", "groq")
        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider_key = provider_key)
            raise ValueError(f"LLM provider {provider_key} not found in config")
        
        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 1000)

        log.info("loading LLM model", provider = provider, model_name = model_name, temperature = temperature, max_tokens = max_tokens)

        if provider == "groq":
            llm=ChatGroq(
                model_name = model_name,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature = temperature,
            )
            return llm
            
        elif provider == "google":
            llm = ChatGoogleGenerativeAI(
                model_name = model_name,
                api_key=os.getenv("GOOGLE_API_KEY"),
                temperature = temperature,
            )
            return llm
        else:
            log.error("invalid LLM provider", provider = provider)
            raise ValueError(f"Invalid LLM provider: {provider}")
        
if __name__ == "__main__":
    loader = ModelLoader()

    # test embedding model
    embeddings = loader.load_embeddings()
    print(f"embedding model loaded successfully")

    # Test the ModelLoader
    result=embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result: {result}")
    
    # Test LLM loading based on YAML config
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    
    # Test the ModelLoader
    result=llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")