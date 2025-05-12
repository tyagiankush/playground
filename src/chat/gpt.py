import asyncio
import logging

from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletion

from src.chat.settings import settings

# Configure logging
logger = logging.getLogger(__name__)


class GPTClient:
	def __init__(self):
		self.client = AsyncOpenAI(api_key=settings.api_key)
		self.model = settings.model_name
		self.temperature = settings.temperature
		self.max_tokens = settings.max_tokens
		self.top_p = settings.top_p

	async def chat_with_llm(self, query: str) -> str | None:
		"""
		Asynchronously chat with the LLM.

		Args:
			query: The user's query string

		Returns:
			Optional[str]: The model's response or None if an error occurs
		"""
		try:
			completion: ChatCompletion = await self.client.chat.completions.create(
				model=self.model,
				messages=[{"role": "user", "content": query}],
				temperature=self.temperature,
				max_tokens=self.max_tokens,
				top_p=self.top_p,
			)

			response = completion.choices[0].message.content
			logger.info(f"Successfully generated response for query: {query[:50]}...")
			return response

		except OpenAIError as e:
			logger.error(f"OpenAI API error: {str(e)}")
			return None
		except Exception as e:
			logger.error(f"Unexpected error: {str(e)}")
			return None


async def main():
	client = GPTClient()
	response = await client.chat_with_llm("write a haiku about ai")
	if response:
		print(response)
	else:
		print("Failed to generate response")


if __name__ == "__main__":
	asyncio.run(main())
