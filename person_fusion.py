from dotenv import load_dotenv
import os
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from output_parser import summary_parser

from third_partie.linkedin import scrape_linkedin_profile
from agents.linkedin_agent import lookup as linkedin_lookup_agent

def ice_break_with(name: str) -> str:
    linkedin_username = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username)

    summary_template = """
    given the Linkedin information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template,
        partial_variables={"format_instructions":summary_parser.get_format_instructions()}
    )

    # Load the environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Initialize the OpenAI API with the loaded API key
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    chain = summary_prompt_template | llm | summary_parser

    res = chain.invoke(input={"information": linkedin_data})

    print(res)

if __name__ == "__main__":
    print("PersonFusion Started")
    ice_break_with(name="Dhanushka Madhushan Gamage")
