from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import DOCXSearchTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json
from typing import List


load_dotenv()

ollama_llm = LLM(
    model="ollama/mistral:latest",
    base_url="http://localhost:11434",
)

docx_search_tool = DOCXSearchTool("/Users/nick_tugarin/Documents/CrewAI/ai_pilot/src/ai_pilot/tools/Dokumentation_SmartPumpProjekt_ScherzingerPumpen.docx")

'''search_tool = DOCXSearchTool(
    docx="/Users/nick_tugarin/Documents/CrewAI/ai_pilot/src/ai_pilot/tools/Dokumentation_SmartPumpProjekt_ScherzingerPumpen.docx",
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama3.1:8b",
                temperature=0.1,
                                                        #temperature=0.1,
            ),
        ),
        embedder=dict(
            provider="ollama",
            config=dict(
                model="nomic-embed-text:latest",#
                
            ),
        )   
    )
)'''

'''class RetrieverState(BaseModel):
    retriever: str = Field(description="Only the whole ToolOutput without any additional text")'''
class RetrieverState(BaseModel):
    retriever: str = Field(description="The retrieved information from the Tool Output Provider")

class CrewState(BaseModel):
    hallucination_check_status: bool = Field(description="The hallucination check status")
    reasoning: str = Field(description="The reasoning for the hallucination check decision")
    #sources: List[str] = Field(description="The sources used to answer the question and provide the hallucination check decision", default_factory=list)
class ResponseState(BaseModel):
    response: str = Field(description="The response to the user's question")

class SummaryState(BaseModel):
    retriever: str = Field(description="The retrieved information from the Tool Output Provider")
    response: str = Field(description="The response to the user's question")
    hallucination_check_status: bool = Field(description="The hallucination check status")
    reasoning: str = Field(description="The reasoning for the hallucination check")

@CrewBase   
class DocumentRetrieverCrew:
    """Document Retriever Crew"""
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def document_retriever_agent(self) -> Agent:
        return Agent(
            max_iter=15,
            config=self.agents_config["document_retriever_agent"],
            tools=[DOCXSearchTool(docx="/Users/nick_tugarin/Documents/CrewAI/ai_pilot/src/ai_pilot/tools/Dokumentation_SmartPumpProjekt_ScherzingerPumpen.docx", result_as_answer=True)],
            verbose=True,
            memory=True,    
        )
    
    @agent
    def response_agent(self) -> Agent:
        return Agent(
            max_iter=15,
            config=self.agents_config["response_agent"],
            verbose=True,
            memory=True,
            task_output_strategy="append",
        )
    
    @agent
    def hallucination_check_agent(self) -> Agent:
        return Agent(
            #llm=ollama_llm,
            max_iter=15,
            config=self.agents_config["hallucination_check_agent"],
            verbose=True,
            memory=True,
        )
    
    @agent
    def summary_agent(self) -> Agent:
        return Agent(
            #llm=ollama_llm,
            max_iter=15,
            config=self.agents_config["summary_agent"],
            verbose=True,
            memory=True,
        )

    @task
    def document_retriever_task(self) -> Task:
        return Task(
            config=self.tasks_config["document_retriever_task"],
            verbose=True,
            tools=[docx_search_tool],
            memory=True,
            output_json=RetrieverState,
        )    
    
    @task
    def response_task(self) -> Task:
        return Task(
            config=self.tasks_config["response_task"],
            verbose=True,
            output_json=ResponseState,
            memory=True,
        )

    @task
    def hallucination_check_task(self) -> Task:
        return Task(
            config=self.tasks_config["hallucination_check_task"],
            memory=True,
            verbose=True,
            output_json=CrewState,
        )
    
    @task
    def summary_task(self) -> Task:
        return Task(
            config=self.tasks_config["summary_task"],
            verbose=True,
            output_json=SummaryState,
            memory=True,
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew"""
        return Crew(
            memory=True,
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )