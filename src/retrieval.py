"""
Provides the `Retrieval` class which implements retrieval: fetching relevant documents.

This implementation combines this with modifying the user input to add context.
"""

from sentence_transformers import SentenceTransformer
import chromadb

# Text embeddings model to use
# - must be compatible with the sentence_transformers library
# - must be the embeddings model the vector collection was created with
EMBEDDINGS_MODEL = "sentence-transformers/all-distilroberta-v1"
# ChromaDB vector database directory
VECTOR_DB_DIR = "/opt/shared/data/chromadb" 
# ChromaDB collection name
VECTOR_DB_COLLECTION = "all-documents"

class Retrieval:
    def __init__(self, model: str = EMBEDDINGS_MODEL, dbdir: str = VECTOR_DB_DIR, collectionName = VECTOR_DB_COLLECTION, log: bool = False):
        """
        Loads the embeddings model and the documents from the vector database.
        """
        
        self.already_injected = set()

        if log:
            print(f"Loading the sentence transformer, {model} ...")

        self.embeddings_model = SentenceTransformer(model)
        
        if log:
            print(f"Loading ChromaDB, {dbdir} ...")
        
        self.chroma = chromadb.PersistentClient(path=dbdir)
        
        if log:
            print(f"Loading collection, {collectionName} ...")
        
        self.vector_collection = self.chroma.get_collection(
            name=collectionName,
            # We generate the embeddings using sentence-transformers directly for pedagogical reasons
            embedding_function=None,
        )

        if log:
            print("Initializing retrieval done")

    def reset(self):
        """
        Resets conversation history.
        """

        self.already_injected = set()

    def augment(self, userinput: str, maxdocs: int, log: bool = False) -> str:
        """
        Augments `userinput` to include up to `maxdocs` document chunks prepended as context.
        Some models may have a chat template for RAG, but notably Qwen does not. This approach works for any model.

        This can be passed into a chat template and given to an LLM to implement Retrieval-Augmented Generation.

        Note that this does not re-inject the same document more than once for a given instance of `Retrieval`.
        If it's determined that duplicate documents are most relevant to subsequent queries, these are ignored.
        Only novel document chunks are injected. 
        """
        
        query_embeddings = self.embeddings_model.encode(userinput)
    
        results = self.vector_collection.query(
            query_embeddings=query_embeddings,
            n_results=maxdocs,
            include=['documents', 'distances'], # returns 'ids', 'documents', 'distances' not 'metadatas'
        )
    
        docs = []
        for doc_id, doc, dist in zip(results['ids'][0], results['documents'][0], results['distances'][0]):
            if doc_id not in self.already_injected:
                # ChromaDB converts the cosine similarity result into something like "cosine distance"
                # by subtracting it from one. This means 0 is most relevant and 2 is least relevant.
                # This threshhold is arbitrary, chosen as it seems to work well with the dataset tested.
                if dist < 0.62:
                    self.already_injected.add(doc_id)
                    docs.append({ "title": doc_id, "dist": dist, "text": doc })
                    
                    if log:
                        print(f"[RAG] INJECTING ({dist}): {doc_id}")
                else:
                    if log:
                        print(f"[RAG] IRRELEVANT ({dist}): NOT INJECTING FILE: {doc_id}")
            else:
                if log:
                    print(f"[RAG] DUPLICATE ({dist}): NOT INJECTING FILE: {doc_id}")
                
    
        # No support in Qwen models for RAG, inject it manually
        if docs:
            context = "<addedcontext>\n" + "\n\n".join([
                    f"<document>\n{doc['text']}\n</document>" for doc in docs
                ]) + "\n</addedcontext>"
                
            return f"{context}\n\nQuestion: {userinput}"
        else:
            return userinput