from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import DOCXSearchTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json
from typing import List


load_dotenv()

ollama_llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434",
)

docx_search_tool = DOCXSearchTool("/Users/nick_tugarin/Documents/CrewAI/ai_pilot/src/ai_pilot/tools/Dokumentation_SmartPumpProjekt_ScherzingerPumpen.docx")

class HumanFeedbackState(BaseModel):
    satisfaction: bool = Field(description="Check if the user is satisfied with the provided answer or Not")
    feedback: str = Field(description="The feedback from the user")

@CrewBase   
class HumanFeedbackCrew:
    """Human Feedback Crew"""
    agents_config = "config/agents_human_feedback.yaml"
    tasks_config = "config/tasks_human_feedback.yaml"

    @agent
    def human_feedback_agent(self) -> Agent:
        return Agent(
            max_iter=15,
            config=self.agents_config["human_feedback_agent"],
            verbose=True,
            memory=True,
        )
    

    @task
    def human_feedback_task(self) -> Task:
        return Task(
            config=self.tasks_config["human_feedback_task"],
            verbose=True,
            tools=[docx_search_tool],
            memory=True,
            output_json=HumanFeedbackState,
            human_input=True,
        )    
    
    
    @crew
    def crew(self) -> Crew:
        """Creates the Human Feedback Crew"""
        return Crew(
            memory=True,
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )