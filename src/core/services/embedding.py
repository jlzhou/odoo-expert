from typing import List
from openai import AsyncOpenAI
from src.utils.logging import logger
from src.config.settings import settings
from llama_index.embeddings.ollama import OllamaEmbedding


class EmbeddingService:
    def __init__(self, client: AsyncOpenAI):
        model_provider = settings.MODEL_PROVIDER
        match model_provider:
            case "openai":
                self.client = client

            case "ollama":
                base_url = settings.OLLAMA_BASE_URL
                self.embed_model = OllamaEmbedding(
                    base_url=base_url,
                    model_name=settings.EMBEDDING_MODEL,
                )

    async def get_embedding(self, text: str) -> List[float]:
        try:
            text = text.replace("\n", " ")
            if len(text) > 8000:
                text = text[:8000] + "..."

            model_provider = settings.MODEL_PROVIDER
            match model_provider:
                case "openai":
                    response = await self.client.embeddings.create(
                        model="text-embedding-3-small",
                        input=text
                    )
                    return response.data[0].embedding
                case "ollama":
                    return await self.embed_model.aget_general_text_embedding(text)
                case _:
                    raise ValueError(f"Invalid model provider: {model_provider}")
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
