import os
import logging
from dotenv import load_dotenv
from openai import OpenAI, APIError

# Load environment variables from .env file
load_dotenv()

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OpenAIClient:
    """
    An exportable client for interacting with OpenAI's LLM (e.g., gpt-4o-mini).
    """
    
    def __init__(self):
        """
        Initializes the OpenAI client, loading the API key.
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logging.error("OPENAI_API_KEY not found in .env file.")
            raise ValueError("OPENAI_API_KEY must be set in the .env file.")
            
        try:
            self.client = OpenAI(api_key=self.api_key)
            logging.info("OpenAIClient initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def chat_in_language(self, user_prompt: str, language_code: str, model: str = "gpt-4o-mini"):
        """
        Gets a chat completion from OpenAI, forcing it to respond in a specific language.
        
        :param user_prompt: The user's question or message.
        :param language_code: The language code (e.g., 'hi-IN', 'es-ES') for the response.
        :param model: The OpenAI model to use.
        :return: The string content of the LLM's reply, or None on failure.
        """
        logging.info(f"Sending prompt to OpenAI '{model}' with language constraint {language_code}")
        
        # This system prompt instructs the AI to ONLY respond in the target language
        system_prompt = (
            f"You are a helpful assistant. You MUST respond ONLY in the language "
            f"identified by the code '{language_code}'. Do not use English or any other "
            f"language."
        )
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                logging.info(f"OpenAI Response received: {content[:50]}...")
                return content.strip()
            else:
                logging.warning("OpenAI response was empty.")
                return None

        except APIError as e:
            logging.error(f"OpenAI API error: {e.status_code} - {e.body}")
        except Exception as e:
            logging.error(f"Unexpected error during OpenAI chat: {e}")
            
        return None

# This block allows you to run this file directly for testing
if __name__ == "__main__":
    print("Testing OpenAIClient...")
    
    try:
        client = OpenAIClient()
    except Exception as e:
        print(f"Failed to initialize client. Check .env file for OPENAI_API_KEY. Error: {e}")
        exit()

    print("\n--- Testing Chat (Hindi) ---")
    hindi_prompt = "भारत की राजधानी क्या है?"
    chat_response_hi = client.chat_in_language(user_prompt=hindi_prompt, language_code="hi-IN")
    if chat_response_hi:
        print(f"Prompt: {hindi_prompt}")
        print(f"Response: {chat_response_hi}")

    print("\n--- Testing Chat (Spanish) ---")
    spanish_prompt = "What is the capital of Spain?"
    chat_response_es = client.chat_in_language(user_prompt=spanish_prompt, language_code="es-ES")
    if chat_response_es:
        print(f"Prompt: {spanish_prompt}")
        print(f"Response: {chat_response_es}")