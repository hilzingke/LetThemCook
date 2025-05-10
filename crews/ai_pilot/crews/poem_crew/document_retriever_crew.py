from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json
from typing import List, Dict, Any
from crewai_tools import RagTool
import os
load_dotenv()

ANTHROPIC_API_KEY=""
mistral_llm = LLM(
    model="ollama/mistral:latest",
    base_url="http://localhost:11434"
)
anthropic_llm= LLM(
    model="anthropic/claude-3-opus-20240229",  # Update to the latest version
    api_key=ANTHROPIC_API_KEY,
    temperature=0.1)
#docx_search_tool = DOCXSearchTool("/Users/nick_tugarin/Documents/CrewAI/ai_pilot/src/ai_pilot/tools/Dokumentation_SmartPumpProjekt_ScherzingerPumpen.docx")

# Define the expected input schema for the RagTool
class RagToolInputSchema(BaseModel):
    query: str = Field(..., description="The search query for the vector database")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Optional keyword arguments for the underlying tool execution")

class RetrieverState(BaseModel):
    document: str = Field(description="The exact Tooloutput from the RagTool")
    # metadata: dict = Field(description="The metadata associated with the retrieved information", default={})

@CrewBase
class DocumentRetrieverCrew:
    """Document Retriever Crew"""
    agents_config = "config/agents_document.yaml"
    tasks_config = "config/tasks_document.yaml"

    @agent
    def document_retriever_agent(self) -> Agent:
        # Initialize RagTool with the defined schema
        knowledge_base_tool = RagTool(
            args_schema=RagToolInputSchema,
            result_as_answer=True
        )
        return Agent(
            config=self.agents_config["document_retriever_agent"],
            llm=anthropic_llm,
            max_iter=3,
            tools=[knowledge_base_tool] # Use the initialized tool
        )
                    
    @task
    def document_retriever_task(self) -> Task:
        return Task(
            config=self.tasks_config["document_retriever_task"],
            verbose=True,
            output_json=RetrieverState
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew"""
        return Crew(
            agents=[self.document_retriever_agent()],  # Automatically created by the @agent decorator
            tasks=[self.document_retriever_task()],  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
    
