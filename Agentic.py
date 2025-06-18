import asyncio
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
import httpx
import llama_index
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent import ReActAgent
from Ollama_Text_To_Sql import TextToSQL
from voice import SpeechTranscriber
from text import text_to_audio_speaker
from mainTool import GoogleCalendarAPI

transcriber = SpeechTranscriber()
# -------------------------
# 1. Initialize AI Model & Embeddings
# -------------------------

text_to_sql = TextToSQL(connection_string="sqlite:///ai_recruitment.sqlite")
api = GoogleCalendarAPI()
tools = [
    FunctionTool.from_defaults(
        fn=text_to_sql.natural_language_to_sql, 
        name="database_query",
        description="""
        Use this tool to query the database using NATURAL LANGUAGE ONLY.
        
        INPUT REQUIREMENTS:
        - ALWAYS use plain, conversational English questions
        - NEVER input SQL code or SQL syntax
        - Format as a clear, simple question (e.g. "What job openings are available?")
        
        USAGE EXAMPLES:
        - "Show me all interviews scheduled for next week"
        - "List the applicants for the Frontend Developer position" 
        - "What are the qualifications for the Senior Developer role?"
        
        The system will automatically convert your natural language to the appropriate SQL query.
        """
    ),
    FunctionTool.from_defaults(api.find_free_slots),
    FunctionTool.from_defaults(api.create_calendar_event_for_interview),
]

agent = ReActAgent(
    llm=Ollama(
        model="granite3.3",
        temperature=0.4,
        request_timeout=120,  # Increased timeout to 120 seconds
    ), 
    max_iterations=5,
    tools=tools,
    verbose=True,
    memory=ChatMemoryBuffer.from_defaults(token_limit=8000)
)

async def main():
    try:
        print("Please enter your questions below (type 'exit' to quit):")
        while True:
            user_input = transcriber.record_and_transcribe()
            if not user_input:
                text_to_audio_speaker("🤐 No speech detected. Please try again.")
                continue
            print(f"\n🗣️ User : {user_input}")
            if user_input.strip().lower() in ("exit", "quit"):
                text_to_audio_speaker("Goodbye!")
                break
            # Fix: ReActAgent.chat() returns a response object, not an awaitable
            ret = agent.chat(user_input)
            print(f"\n🤖 AGENT : {ret}")
            text_to_audio_speaker(str(ret))
    except httpx.ReadTimeout as e:
        print(f"Request timed out: {e}. Please try again later.")
    except llama_index.core.workflow.errors.WorkflowRuntimeError as e:
        print(f"Workflow error occurred: {e}. Please try again.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())