import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from phase01.memory_for_llm import embedding_model


# Environmental varibale for hugging face moidels and other paths etc
load_dotenv()
HF_TOKEN=os.getenv("HF_TOKEN")
DB_PATH = "vectorstore/faiss_db"


# steps
# Setup llm (Mistral)
def setting_llm():
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature = 0.5,
        task="conversational",   # 👈 this is the important fix
        max_new_tokens = 512
        )
    return llm

# create a custom prdompt for the answer
CUSTOM_PROMPT="""
Usse the pieces of information provided in the context to answer the user's question.
If in any case you doo not know the exact answer, just say you do not know the answer.
Do not provide aanything out of the context

Context: {context}
Question: {question}

Start the answer directly and avoid the small taks please.
"""

# set the prompt template
def set_custom_prompt(custom_prompt):
    prompt = PromptTemplate(
        template=custom_prompt,
        input_variables=['context', 'question']
    )
    return prompt

# load the db
db = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)

# CREATE a qa chain
def qa_cahin():
    chain = RetrievalQA.from_chain_type(
        llm = setting_llm(),
        chain_type = "stuff",
        retriever = db.as_retriever(search_kwargs={"k":3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt":set_custom_prompt(CUSTOM_PROMPT)}
    )
    return chain
chain = qa_cahin()

# user query and response generation
user_query = input("Query: ")
resp = chain.invoke({"query":user_query})
print("Response: ", resp['result'])
print("docs: ", resp['source_documents'])