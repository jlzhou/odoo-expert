from typing import List, Dict, Optional, Tuple
from openai import AsyncOpenAI
from src.core.services.embedding import EmbeddingService
from src.core.services.db_service import DatabaseService
from src.config.settings import settings
from src.utils.logging import logger
from llama_index.llms.ollama.base import Ollama
from llama_index.core.llms import ChatMessage, MessageRole


class ChatService:
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        db_service: DatabaseService,
        embedding_service: EmbeddingService,
        ollama_llm: Ollama
    ):
        self.openai_client = openai_client
        self.ollama_llm = ollama_llm
        self.db_service = db_service
        self.embedding_service = embedding_service

    async def retrieve_relevant_chunks(
        self,
        query: str,
        version: int,
        limit: int = 3
    ) -> List[Dict]:
        try:
            query_embedding = await self.embedding_service.get_embedding(query)
            chunks = await self.db_service.search_documents(
                query_embedding,
                version,
                limit
            )
            return chunks
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            raise

    def prepare_context(self, chunks: List[Dict]) -> Tuple[str, List[Dict[str, str]]]:
        """Prepare context and sources from retrieved chunks."""
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(chunks, 1):
            source_info = (
                f"Source {i}:\n"
                f"Document: {chunk['url']}\n"
                f"Title: {chunk['title']}\n"
                f"Content: {chunk['content']}"
            )
            context_parts.append(source_info)
            sources.append({
                "url": chunk["url"],
                "title": chunk["title"]
            })
        
        return "\n\n---\n\n".join(context_parts), sources

    async def generate_response(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None,
        stream: bool = False
    ):
        """Generate AI response based on query and context."""
        try:
            model_provider = settings.MODEL_PROVIDER
            match model_provider:
                case "openai":
                    messages = [
                        {
                            "role": "system",
                            "content": settings.SYSTEM_PROMPT
                        }
                    ]

                    if conversation_history:
                        history_text = "\n".join([
                            f"User: {msg['user']}\nAssistant: {msg['assistant']}"
                            for msg in conversation_history[-3:]
                        ])
                        messages.append({
                            "role": "user",
                            "content": f"Previous conversation:\n{history_text}"
                        })

                    messages.append({
                        "role": "user",
                        "content": f"Question: {query}\n\nRelevant documentation:\n{context}"
                    })

                    response = await self.openai_client.chat.completions.create(
                        model=settings.LLM_MODEL,
                        messages=messages,
                        stream=stream
                    )

                    if stream:
                        return response
                    return response.choices[0].message.content

                case "ollama":
                    messages = [ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=settings.SYSTEM_PROMPT,
                    )]

                    if conversation_history:
                        history_text = "\n".join([
                            f"User: {msg['user']}\nAssistant: {msg['assistant']}"
                            for msg in conversation_history[-3:]
                        ])
                        messages.append(ChatMessage(
                            role=MessageRole.USER,
                            content=f"Previous conversation:\n{history_text}",
                        ))

                    messages.append(ChatMessage(
                        role=MessageRole.USER,
                        content=f"Question: {query}\n\nRelevant documentation:\n{context}",
                    ))

                    final_response = ""
                    response_stream = self.ollama_llm.chat(messages)
                    for response in response_stream:
                        if response[0] == 'message':
                            final_response = response[1].content
                    return final_response

                case _:
                    raise ValueError(f"Invalid model provider: {model_provider}")
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def generate_translation(
        self,
        query: str,
    ):
        messages = [ChatMessage(
            role=MessageRole.USER,
            content=f"Translate the following sentence in English Only: {query}",
        )]
        try:
            model_provider = settings.MODEL_PROVIDER
            match model_provider:
                case "openai":
                    raise NotImplementedError("openai Not Implemented")
                case "ollama":
                    final_response = ""
                    response_stream = self.ollama_llm.chat(messages)
                    for response in response_stream:
                        if response[0] == 'message':
                            final_response = response[1].content
                    return final_response
                case _:
                    raise ValueError(f"Invalid model provider: {model_provider}")

        except Exception as e:
            logger.error(f"Error generating translation: {e}")
            raise
