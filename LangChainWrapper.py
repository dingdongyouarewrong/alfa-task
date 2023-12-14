from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

from constants import PROMPT_TEMPLATE_STRING, EMBEDDER_MODEL_NAME, CHROMA_PATH


class LangChainWrapper:
    def __init__(self, model):
        # Initialize the PromptTemplate with input variables and template string
        self.prompt = PromptTemplate(input_variables=["context", "question"], template=PROMPT_TEMPLATE_STRING)
        # Create an instance of LLMChain with the prompt and the language model
        self.llm_chain = LLMChain(prompt=self.prompt, llm=model)
        self.db = None

    def get_embeddings(self, file_path=None, embeddings_folder_path=CHROMA_PATH, save_embeddings=True,
                       load_embeddings_from_file=False):
        embedder = HuggingFaceEmbeddings(model_name=EMBEDDER_MODEL_NAME)

        if not load_embeddings_from_file:
            # Load unstructured file (such as a PDF) and extract documents
            loader = UnstructuredFileLoader(file_path)
            documents = loader.load()
            # Split the text into smaller chunks for processing
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            docs = text_splitter.split_documents(documents)

            # Initialize the HuggingFaceEmbeddings model for generating document embeddings

            # Create a FAISS index and store the document embeddings
            self.db = Chroma.from_documents(docs, embedder)

            if save_embeddings:
                # Save the document embeddings to a local folder
                self.db.persist()
        else:
            # Load the document embeddings from a local folder
            self.db = Chroma(persist_directory=embeddings_folder_path, embedding_function=embedder)

    def get_answer(self, question):
        # Perform a similarity search using the question and the document embeddings
        search = self.db.similarity_search(question, k=2)

        # Run the LLMChain using the question and the search results as input variables
        return self.llm_chain.run({"question": question, "context": search})