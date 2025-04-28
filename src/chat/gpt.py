from openai import OpenAI

from src.chat.settings import settings


def chat_with_llm(query: str) -> None:
	client = OpenAI(api_key=settings.api_key)

	completion = client.chat.completions.create(
		model=settings.model_name, store=True, messages=[{"role": "user", "content": query}]
	)

	print(completion.choices[0].message)


if __name__ == "__main__":
	chat_with_llm("write a haiku about ai")
