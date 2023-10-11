from langchain import PromptTemplate, LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from constants import PROMPT_TEMPLATE_STRING, EMBEDDER_MODEL_NAME


class LangChainWrapper:
    def __init__(self, model):
        # Initialize the PromptTemplate with input variables and template string
        self.prompt = PromptTemplate(input_variables=["context", "question"], template=PROMPT_TEMPLATE_STRING)
        # Create an instance of LLMChain with the prompt and the language model
        self.llm_chain = LLMChain(prompt=self.prompt, llm=model)
        self.db = None

    def embed_file(self, file_path, embeddings_folder_path="faiss_alfa_embeddings", save_embeddings=True,
                   load_embeddings_from_file=False):
        # Load unstructured file (such as a PDF) and extract documents
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()
        # Split the text into smaller chunks for processing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        # Initialize the HuggingFaceEmbeddings model for generating document embeddings
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDER_MODEL_NAME)

        # Create a FAISS index and store the document embeddings
        self.db = FAISS.from_documents(docs, embeddings)

        if save_embeddings:
            # Save the document embeddings to a local folder
            self.db.save_local(embeddings_folder_path)
        if load_embeddings_from_file:
            # Load the document embeddings from a local folder
            self.db = FAISS.load_local(embeddings_folder_path, embeddings=embeddings)

    def get_answer(self, question):
        # Perform a similarity search using the question and the document embeddings
        search = self.db.similarity_search(question, k=2)

        # Run the LLMChain using the question and the search results as input variables
        return self.llm_chain.run({"question": question, "context": search})