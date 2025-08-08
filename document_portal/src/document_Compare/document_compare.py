import sys
from dotenv import load_dotenv
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from prompts.prompt_library import PROMPT_REGISTRY
from model.models import SummaryResponse, PromptType

class DocumentComparatorLLM:
    def __init__(self):
        load_dotenv()
        self.log = CustomLogger().get_logger(__name__)
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
        self.fix_parser = OutputFixingParser.from_llm(llm=self.llm, parser=self.parser)
        self.prompt = PROMPT_REGISTRY[PromptType.DOCUMENT_COMPARISON.value]
        self.chain = self.prompt | self.llm | self.parser
        self.log.info("DocumentComparatorLLM initialized successfully", model = self.llm)

    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        try:
            inputs = {
                "combined_docs": combined_docs,
                "format_instructions": self.parser.get_format_instructions()
            }

            self.log.info("Invoking DocumentComparatorLLM chain")
            response = self.chain.invoke(inputs)
            self.log.info("chain invoked successfully", response_preview=str(response)[:100])
            return self._format_response(response)
        except Exception as e:
            self.log.error("Error in DocumentComparatorLLM chain", error=str(e))
            raise DocumentPortalException("error in DocumentComparatorLLM chain", sys)
        
    def _format_response(self, response_parsed: list[dict]) -> pd.DataFrame:
        try:
            df = pd.DataFrame(response_parsed)
            return df
        except Exception as e:
            self.log.error("Error in formatting response", error=str(e))
            raise DocumentPortalException("error in formatting response", sys)
        
    