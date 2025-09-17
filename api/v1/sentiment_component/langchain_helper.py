import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class SentimentAnalysisLLM:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.3,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"
        )
 
    def generate_summary(self, sentiment_data: dict) -> str:
        """
        Generate a single, concise summary of Reddit sentiment analysis
        combining professional insight and user-friendly readability.
        """
        prompt_template = PromptTemplate(
            input_variables=["sentiment_data"],
            template="""
You are an expert financial analyst. Based on the following Reddit sentiment data,
create **one concise summary** suitable for non-experts, covering key insights and trends:

{sentiment_data}

Instructions:
1. Provide a brief overview of overall sentiment (positive, neutral, negative)
2. Highlight main economic concerns or trends
3. Explain implications for everyday people
4. Keep the summary professional yet easy to understand
5. Limit output to 200-250 words

Write it as a single coherent paragraph.
"""
        )

        chain = LLMChain(llm=self.llm, prompt=prompt_template)

        try:
            response = chain.run(sentiment_data=str(sentiment_data))
            return response
        except Exception as e:
            return f"Error generating LLM summary: {str(e)}"