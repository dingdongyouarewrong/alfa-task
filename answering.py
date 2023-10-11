from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler
)
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from constants import PROMPT_TEMPLATE_STRING, EMBEDDER_MODEL_NAME

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

model = LlamaCpp(
    model_path="/Users/dmitry/PycharmProjects/alfa-task/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=2000,
    top_k=30,
    top_p=0.9,
    temperature=0.2,
    repeat_penalty=1.1,
    n_parts=1,
    lora_path="/finetuning/ggml-adapter-model.bin"
)

prompt = PromptTemplate(input_variables=["context", "question"], template=PROMPT_TEMPLATE_STRING)

llm_chain = LLMChain(prompt=prompt, llm=model)

loader = UnstructuredFileLoader(
    "/data/Альфа-Клиент_Руководство пользователя_2017.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# embedding engine
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDER_MODEL_NAME)

db = FAISS.from_documents(docs, embeddings)

# save embeddings in local directory
db.save_local("faiss_alfa_embeddings")

# load from local
db = FAISS.load_local("/Users/dmitry/PycharmProjects/alfa-task/faiss_alfa_embeddings/", embeddings=embeddings)

query = "Что может делать клиент банка?"
search = db.similarity_search(query, k=2)

final_prompt = prompt.format(question=query, context=search)

print(llm_chain.run({"question": query, "context": search}))
