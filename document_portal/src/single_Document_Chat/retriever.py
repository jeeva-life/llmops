import sys, os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retriever_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from prompts.prompt_library import PROMPT_REGISTRY
from model.models import PromptType

load_dotenv()

class ConversationalRAG:
    def __init__(self, retriever, session_id: str):
        self.log = CustomLogger().get_logger(__name__)
        self.session_id = session_id
        self.retriever = retriever

        try:
            self.llm = _load_llm()
            self.contextualize_prompt = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]

            self.history_aware_retriever = create_history_aware_retriever(
                self.llm, self.retriever, self.contextualize_prompt
            )
            self.log.info("History aware retriever created successfully", session_id=self.session_id)

            self.qa_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
            self.rag_chain = create_retriever_chain(self.history_aware_retriever, self.qa_chain)
            self.log.info("RAG chain created successfully", session_id=self.session_id)

            self.chain = RunnableWithMessageHistory(
                self.rag_chain,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            self.log.info("Wrapped chain with message history", session_id=self.session_id)

        except Exception as e:
            self.log.error(f"Error initializing conversational RAG", error=str(e), session_id=self.session_id)
            raise DocumentPortalException(f"Error initializing conversational RAG", sys)
        

    def _load_llm(self):
        try:
            llm = ModelLoader.load_llm()
            self.log.info("LLM loaded successfully", class_name=llm.__class__.__name__)
            return llm
        except Exception as e:
            self.log.error(f"Error loading LLM via ModelLoader", error=str(e), session_id=self.session_id)
            raise DocumentPortalException(f"Error loading LLM via ModelLoader", sys)
        
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        try:
            if store not in st.session_state:
                st.session_state.store = {}
            
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
                self.log.info("New session history created", session_id=session_id)

            return st.session_state.store[session_id]
        except Exception as e:
            self.log.error(f"Error getting session history", error=str(e), session_id=self.session_id)
            raise DocumentPortalException(f"Error getting session history", sys)
        
    def load_retriever_from_faiss(self, faiss_index_path: str):
        try:
            embeddings = ModelLoader.load_embeddings()
            if not os.path.isdir(faiss_index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {faiss_index_path}")
            
            vectorstore = FAISS.load_local(faiss_index_path, embeddings)
            self.log.info("Retriever loaded from FAISS successfully", index_path=faiss_index_path)
            return vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})
        
        except Exception as e:
            self.log.error(f"Error loading retriever from FAISS", error=str(e))
            raise DocumentPortalException(f"Error loading retriever from FAISS", sys)
        
    
    def invoke(self, user_input: str) -> str:
        try:
            response = self.chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": self.session_id}}
            )
            answer = response.get("answer", "No answer found")

            if not answer:
                self.log.warning("No answer found", session_id=self.session_id)

            self.log.info("Answer generated successfully", session_id=self.session_id, user_input=user_input, answer_preview=answer[:100])
            return answer
        
        except Exception as e:
            self.log.error(f"Error invoking conversational RAG", error=str(e), session_id=self.session_id)
            raise DocumentPortalException(f"Error invoking conversational RAG chain", sys)