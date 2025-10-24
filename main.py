import os
import subprocess
import shutil
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb

import split_text

embeddings_model_name = "sentence-transformers/all-distilroberta-v1"
chat_model_name = "Qwen/Qwen2.5-14B-Instruct"

PRJ_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(PRJ_DIR, "data", "raw")
DOC_DIR = os.path.join(PRJ_DIR, "data", "split")

def load_documents_and_generate_embeddings(embeddings_model, vector_collection):

    paths = []
    docs = []
    for entry in os.scandir(DOC_DIR):
        if entry.is_file():
            paths.append(entry.path)

            with open(entry.path, "r") as f:
                docs.append(f.read())

    embeddings = embeddings_model.encode(docs)

    fnames = [os.path.basename(path) for path in paths]

    vector_collection.upsert(
        ids=fnames,
        embeddings=embeddings,
        documents=docs,
        # metadatas=[{"chapter": 3, "verse": 16}, {"chapter": 3, "verse": 5}, {"chapter": 29, "verse": 11}, ...],
    )

class Chat:
    def __init__(self, chat_tokenizer, chat_model, embeddings_model, vector_collection):
        self.chat_tokenizer = chat_tokenizer
        self.chat_model = chat_model
        self.embeddings_model = embeddings_model
        self.vector_collection = vector_collection
    
        """The IDs of documents already injected into the chat context."""
        self.doc_ids = set()
        """The history of messages."""
        self.conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

    def get_response(self, userinput: str):
        query_embeddings = self.embeddings_model.encode(userinput)

        results = self.vector_collection.query(
            query_embeddings=query_embeddings,
            n_results=5,
            include=["documents"], # returns 'ids', 'documents' not 'distances', 'metadatas'
        )

        docs = []
        for doc_id, doc in zip(results['ids'][0], results['documents'][0]):
            if doc_id not in self.doc_ids:
                self.doc_ids.add(doc_id)
                docs.append({ "title": doc_id, "text": doc })

        # No support in these Qwen models for RAG, inject it manually
        if docs:
            context = "<addedcontext>\n" + "\n\n".join([
                    f"<document id='{doc['title']}'>\n{doc['text']}\n</document>" for doc in docs
                ]) + "\n</addedcontext>"
                
            user_message = f"{context}\n\nQuestion: {userinput}"
        else:
            user_message = userinput
    
        self.conversation.append({
            "role": "user",
            "content": user_message
        })
        
        input_ids = self.chat_tokenizer.apply_chat_template(
            conversation=self.conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.chat_model.device)

        attention_mask = (input_ids != self.chat_tokenizer.pad_token_id).long().to(self.chat_model.device)

        # Generate a response 
        generated_tokens = self.chat_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.3,
        )

        # Only decode the newly generated tokens, not the input
        generated_text = self.chat_tokenizer.decode(
            generated_tokens[0][input_ids.shape[1]:],  # Skip input tokens
            skip_special_tokens=True
        )

        self.conversation.append({
            "role": "assistant",
            "content": generated_text
        })

        return generated_text

def main():

    # Check GPU availability
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Checking for documents...")

    if not os.path.isdir(RAW_DIR):
        subprocess.run("./pull_data.sh")

    print("Splitting documents...")

    shutil.rmtree(DOC_DIR, ignore_errors=True)
    split_text.split_many(RAW_DIR, DOC_DIR)

    print(f"Loading the sentence transformer, {embeddings_model_name}...")

    embeddings_model = SentenceTransformer(embeddings_model_name)

    print("Loading ChromaDB and the documents...")

    chroma = chromadb.PersistentClient(path=os.path.join(PRJ_DIR, "data", "chroma"))
    
    vector_collection = chroma.get_or_create_collection(
        name="all-documents",
        # We generate the embeddings using sentence-transformers directly for pedagogical reasons
        embedding_function=None,
        metadata={
            "description": "my first Chroma collection",
            "created": str(datetime.now())
        },
        configuration={
            "hnsw": {
                # Cosine similarity is most useful for text embeddings I believe,
                # where scale is of little importance?
                "space": "cosine",
            }
        }
    )

    load_documents_and_generate_embeddings(embeddings_model, vector_collection)

    print(f"Loading chat model, {chat_model_name}...")

    chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_name)
    chat_model = AutoModelForCausalLM.from_pretrained(
        chat_model_name,
        dtype="auto",
        device_map="auto"
    )

    print("Preparing chat interface...")

    chat = Chat(chat_tokenizer, chat_model, embeddings_model, vector_collection)

    print("Ready! Enter [q]uit to exit.")

    while True:
        print("> ", end="")
        userinput = input()

        if not userinput:
            continue

        if userinput == "quit" or userinput == "q" or userinput == "exit":
            break

        response = chat.get_response(userinput)

        print(response)

if __name__ == "__main__":
    main()
