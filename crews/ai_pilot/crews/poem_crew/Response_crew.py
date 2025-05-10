from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json
from typing import List


load_dotenv()

ollama_llm = LLM(
    model="ollama/gemma3:4b",
    base_url="http://localhost:11434",
)

llama_llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434",
)

    #sources: List[str] = Field(description="The sources used to answer the question and provide the hallucination check decision", default_factory=list)
class ResponseState(BaseModel):
    response: str = Field(description="The response to the user's question")

@CrewBase   
class ResponseCrew:
    """Response Crew"""
    agents_config = "config/agents_response.yaml"
    tasks_config = "config/tasks_response.yaml"
    
    #Antwort generieren
    @agent
    def response_agent(self) -> Agent:
        return Agent(
            max_iter=5,
            config=self.agents_config["response_agent"],
            verbose=True,
        )

    
    @task
    def response_task(self) -> Task:
        return Task(
            config=self.tasks_config["response_task"],
            verbose=True,
            output_json=ResponseState,
        )
    
    # @task
    # def response_correction_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config["response_correction_task"],
    #         verbose=True,
    #         #output_json=ResponseState,
    #     )
    
    # @task
    # def response_agent_2_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config["response_agent_2_task"],
    #         verbose=True,
    #         output_json=ResponseState,
    #     )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )