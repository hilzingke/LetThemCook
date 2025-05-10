from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json
from typing import List

load_dotenv()
ANTHROPIC_API_KEY=""
llama_llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434",
)

phi_llm=LLM(
    model="ollama/phi4:latest",
    base_url="http://localhost:11434"
)

granite_llm = LLM(
    model="ollama/granite3-dense:8b",
    base_url="http://localhost:11434",
)

gemma_llm = LLM(
    model="ollama/gemma3:12b",
    base_url="http://localhost:11434"
)

ollama_llm = LLM(
    model="ollama/mistral:latest",
    base_url="http://localhost:11434",
)

anthropic_llm= LLM(
    model="anthropic/claude-3-5-haiku-20241022",  # Update to the latest version
    api_key=ANTHROPIC_API_KEY,
    temperature=0.1)


class HallucinationState(BaseModel):
    hallucination_check_status: bool = Field(description="The hallucination check status")
    reasoning: str = Field(description="The reasoning for the hallucination check decision")
    #sources: List[str] = Field(description="The sources used to answer the question and provide the hallucination check decision", default_factory=list)

@CrewBase   
class HallucinationCrew:
    """Hallucination Crew"""
    agents_config = "config/agents_hallucination.yaml"
    tasks_config = "config/tasks_hallucination.yaml"
    
    @agent
    def hallucination_check_agent(self) -> Agent:
        return Agent(
            #llm=granite_llm,
            #llm=gemma_llm,
            #llm=ollama_llm,
            #llm=phi_llm,
            #llm=llama_llm,
            llm=anthropic_llm,
            max_iter=15,
            config=self.agents_config["hallucination_check_agent"],
            verbose=True,
        )
    
    @task
    def hallucination_check_task(self) -> Task:
        return Task(
            config=self.tasks_config["hallucination_check_task"],
            verbose=True,
            output_json=HallucinationState,
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Hallucination Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )