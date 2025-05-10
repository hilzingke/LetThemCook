from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json
from typing import List


load_dotenv()
ANTHROPIC_API_KEY=""
anthropic_llm= LLM(
    model="anthropic/claude-3-opus-20240229",  # Update to the latest version
    api_key=ANTHROPIC_API_KEY,
    temperature=0.1)

ollama_llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434",
)


class RephrazeState(BaseModel):
    rehpraze_question: str = Field(description="The provided new question or task from the agent")


@CrewBase   
class RephrazeCrew:
    """Rephraze Crew"""
    agents_config = "config/agents_rephraze.yaml"
    tasks_config = "config/tasks_rephraze.yaml"

    @agent
    def rephraze_agent(self) -> Agent:
        return Agent(
            #llm=anthropic_llm,
            max_iter=3,
            config=self.agents_config["rephraze_agent"],
            verbose=True,
            memory=True,    
        )

    @task
    def rephraze_task(self) -> Task:
        return Task(
            config=self.tasks_config["rephraze_task"],
            verbose=True,
            memory=True,
            output_json=RephrazeState,
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