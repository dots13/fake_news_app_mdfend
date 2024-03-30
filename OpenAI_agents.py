from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
import json
import serpapi
import os


class FilterAgent:
    def __init__(self, client):
        self.client = client
        filter_agent_template = """ You are an agent with the task of counting in how many entries of a context 
                {context}, a text extract {headline} can be found identically and literally word by word.
                You will review each entry and see if the extract {headline} can be found exactly the same within each 
                entry, not just similar semantically but word by word.

                your job is to generate a JSON structure with the number of entries where this happens:

                      (
                        "times": number of entries where the headline is found exactly and literally word by word,
                      )
                """
        self.__prompt_template = PromptTemplate(template=filter_agent_template, input_variables=["headline", "context"])
        self.llm_chain = LLMChain(prompt=self.__prompt_template, llm=self.client)

    def run_filter_agent(self, headline, context):
        try:
            output = self.llm_chain.run({'headline': headline, 'context': context})
            return output
        except Exception as e:
            print(e)
            return "Error in filtering layer"


class ClassAgent:
    def __init__(self, client):
        self.client = client

        class_agent_template = """ You are an agent with the task of analysing a headline {headline} .
                you will identify the subject, the event, and the field the news belongs to either Politics, Economics, 
                or Social.
                you will provide a JSON Structure:
                  (
                  "subject": subject of the news,
                  "event": event described,
                  "topic": field the news belongs to Politics, Economics, or Social
                  )
                """
        self.prompt_template = PromptTemplate(template=class_agent_template, input_variables=["headline"])
        self.llm_chain = LLMChain(prompt=self.prompt_template, llm=self.client)

    def run_class_agent(self, headline):
        try:
            output = self.llm_chain.run({'headline': headline})
            return output
        except Exception as e:
            print(e)
            return "Error in classification layer"


class DecisionAgent:
    def __init__(self, client):
        self.client = client
        decision_agent_template = """you are information verification agent in 2024,
                You will be presented with a piece of news {news} and information gathered from the internet 
                {filtered_context}.
                Your task is to evaluate whether the news is real or fake, based solely on:

                - How the {news} corresponds to the information retrieved {filtered_context}, considering the 
                reliability of the sources.
                - Probability of the news {probability} being real.
                - Alignment of the headline and the news {alignment},Not aligment is a sign of fake news .
                - Number of times the exact headline is found in other media outlets {times} which could indicate a 
                misinformation campaign.

                Based on these criteria provided in order of importance,
                produced a reasoned argumentation whether the news is Fake or real.
                You answer strictly as a single JSON string. Don't include any other verbose texts and don't include 
                the markdown syntax anywhere.

                  (
                "category": Fake or Real,
                "reasoning": Your reasoning here.
                   )  
                provide your answers in Spanish
                """
        self.prompt_template = PromptTemplate(template=decision_agent_template,
                                              input_variables=["news",
                                                               "filtered_context",
                                                               "probability",
                                                               "alignment",
                                                               "times"])
        self.llm_chain = LLMChain(prompt=self.prompt_template, llm=self.client)

    def run_decision_agent(self, news, filtered_context, probability, alignment, times):
        try:
            output = self.llm_chain.run(
                {'news': news, 'filtered_context': filtered_context, 'probability': probability, 'alignment': alignment,
                 'times': times})
            return output
        except Exception as e:
            print(e)
            return "Error in decision layer"


class HeadlineAgent:
    def __init__(self, client):
        self.client = client
        headline_agent_template = """ You are an agent with the task of identifying the 
        whether the headline {headline} is aligned with
        the body of the news {news}.
        you will generate a Json output:

      (
        "label": Aligned or not Aligned,
      )

        """
        self.prompt_template = PromptTemplate(template=headline_agent_template, input_variables=["headline", "news"])
        self.llm_chain = LLMChain(prompt=self.prompt_template,
                                  llm=self.client)  # Assuming LLLMChain was a typo and should be LLMChain

    def analyze_alignment(self, headline, news):
        """
        Analyzes the alignment between a given headline and the body of the news.

        Parameters:
        - headline (str): The news headline.
        - news (str): The full text of the news article.

        Returns:
        - A dictionary with the analysis results, including whether the headline is aligned with the news body and any
        relevant analysis details.
        """
        try:
            output = self.llm_chain.run({'headline': headline, 'news': news})
            return output
        except Exception as e:
            print(e)
            return {"error": "Error in headline alignment analysis layer"}


# Example usage:
if __name__ == "__main__":
    open_ai_key = ""
    os.environ["OPENAI_API_KEY"] = open_ai_key
    client = OpenAI(temperature=0)
    class_agent = ClassAgent(client=client)
    headline = 'Head'
    class_result = class_agent.run_class_agent(headline=headline)
    data = json.loads(class_result)
    subject = data["subject"]
    event = data["event"]
    print(data)
