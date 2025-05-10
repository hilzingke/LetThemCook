#!/usr/bin/env python
from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start
from src.ai_pilot.crews.poem_crew.document_retriever_crew import DocumentRetrieverCrew
from src.ai_pilot.crews.poem_crew.Response_crew import ResponseCrew
from src.ai_pilot.crews.poem_crew.hallucination_crew import HallucinationCrew
from src.ai_pilot.crews.poem_crew.Rephraze_crew import RephrazeCrew
import datetime
import asyncio
import os
class DocumentRetrieverState(BaseModel):
    question_state: str = ""
    document_state: str = ""  # Für document_retriever_task Output
    response_state: str = "" 
    hallucination_check_state: bool=False
    hallucination_reasoning_state: str = ""
    
class DocumentRetrieverFlow(Flow[DocumentRetrieverState]):
    @start()
    def generate_question(self):
        print("Generating question")
        # Only set default question if no query is provided
        if not self.state.question_state:
            self.state.question_state = "wer waren die verfasser zur dokumentation des projekts smart pump?"
        print(f"Aktuelle Frage (mit Kontext): {self.state.question_state}")
    @listen(generate_question)
    def generate_relevant_information(self):
        print("Generating relevant information")
        try:
            # Ensure query is a string
            query_str = str(self.state.question_state)

            # Remove tool_kwargs creation and passing entirely
            result = (
                DocumentRetrieverCrew() # No kwargs in constructor
                .crew()
                .kickoff(inputs={
                    "question_state": query_str
                    # No kwargs in inputs either
                })
            )
            if result is None:
                print("Warning: DocumentRetrieverCrew returned None")
                self.state.document_state = "Error: No document retrieved"
                return
            if not hasattr(result, 'json_dict'):
                print(f"Warning: Result has no json_dict attribute. Result type: {type(result)}")
                print(f"Result content: {result}")
                self.state.document_state = "Error: Invalid document format"
                return
            document = result.json_dict.get("document", "")
            if not document:
                print("Warning: Empty document in json_dict")
                print(f"Full json_dict: {result.json_dict}")
            self.state.document_state = document
            print(f"Retrieved document: {document[:100]}...")  # Print first 100 chars for debugging
        except Exception as e:
            print(f"Error in generate_relevant_information: {str(e)}")
            self.state.document_state = f"Error retrieving document: {str(e)}"
    @listen(generate_relevant_information)
    def generate_response(self):
        print("Generating response")
        try:
            result = (
                ResponseCrew()
                .crew()
                .kickoff(inputs={"document_state": self.state.document_state,
                                 "question_state": self.state.question_state})
            )
            
            if result is None:
                print("Warning: ResponseCrew returned None")
                self.state.response_state = "Error: No response generated"
                return
                
            if not hasattr(result, 'json_dict'):
                print(f"Warning: Result has no json_dict attribute. Result type: {type(result)}")
                print(f"Result content: {result}")
                self.state.response_state = "Error: Invalid response format"
                return
                
            response = result.json_dict.get("response", "")
            if not response:
                print("Warning: Empty response in json_dict")
                print(f"Full json_dict: {result.json_dict}")
                
            self.state.response_state = response
            print(f"Generated response: {response[:100]}...")  # Print first 100 chars for debugging
            
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            self.state.response_state = f"Error generating response: {str(e)}"
    @listen(generate_response)
    def generate_hallucination_check(self):
        print("Generating hallucination check")
        result = (
            HallucinationCrew()
            .crew()
            .kickoff(inputs={"document_state": self.state.document_state,
                             "question_state": self.state.question_state,
                             "response_state": self.state.response_state})
        )
        hallucination_response = result.json_dict
        
        self.state.hallucination_check_state = hallucination_response.get("hallucination_check_status", None)
        self.state.hallucination_reasoning_state = hallucination_response.get("reasoning", "")
    @listen(generate_hallucination_check)
    def finalize(self):
        # Print final states
        print("Final response:")
        print(self.state.response_state)
        print(f"Halluzination-Status: {self.state.hallucination_check_state}")
        print(f"Begründung: {self.state.hallucination_reasoning_state}")
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Write state files with relative paths
        try:
            # Convert all states to strings
            question_state = str(self.state.question_state)
            document_state = str(self.state.document_state)
            response_state = str(self.state.response_state)
            hallucination_check_state = str(self.state.hallucination_check_state)
            hallucination_reasoning_state = str(self.state.hallucination_reasoning_state)
            # Write each file with proper string content
            files_to_write = {
                "Question.txt": question_state,
                "Document.txt": document_state,
                "Response.txt": response_state,
                "TrueOrFalse.txt": hallucination_check_state,
                "Reasoning.txt": hallucination_reasoning_state
            }
            
            for filename, content in files_to_write.items():
                file_path = os.path.join(project_root, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Successfully wrote {filename}")
                
        except Exception as e:
            print(f"Error writing files: {e}")
            raise  # Re-raise the exception to ensure we know if something went wrong
def kickoff(query: str = None, kwargs: dict = None):
    document_retriever_flow = DocumentRetrieverFlow()
    if query:
        document_retriever_flow.state.question_state = query
    
    # Handle default kwargs if None are provided
    if kwargs is None:
        kwargs = {
            "temperature": 0.0,
            "max_tokens": 2000 # Or another sensible default
        }
        
    # Pass kwargs to the flow kickoff
    try:
        document_retriever_flow.kickoff(inputs={
            "question_state": document_retriever_flow.state.question_state,
            "kwargs": kwargs # Pass kwargs here
        })
    except Exception as e:
        print(f"Error in kickoff: {str(e)}")
        # Reset state and try again, passing kwargs
        document_retriever_flow.state = DocumentRetrieverState()
        document_retriever_flow.state.question_state = query
        # Use the provided or default kwargs for the retry
        if kwargs is None: # Recalculate default if necessary (shouldn't happen here normally)
             kwargs = {"temperature": 0.0, "max_tokens": 2000}
        document_retriever_flow.kickoff(inputs={
            "question_state": document_retriever_flow.state.question_state,
            "kwargs": kwargs # Pass kwargs here too
        })

if __name__ == "__main__":
    # Call without kwargs, defaults will be applied inside kickoff
    kickoff()
