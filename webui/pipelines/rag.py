"""
title: RAG: ChromaDB + all-distilroberta-v1
author: Shaun Beautement
version: 0.1.0
required_open_webui_version: 0.3.9
requirements: chromadb>=1.3, sentence-transformers
"""

from typing import List, Optional
from pydantic import BaseModel, Field
import os
import sys
import chromadb

sys.path.append('/deps')
from retrieval import Retrieval


class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]

        enabled: bool = Field(
            default=True, description="Enable or disable this RAG filter"
        )
        chroma_path: str = Field(
            default="/data/chromadb",
            description="Path to ChromaDB persistent storage"
        )
        collection_name: str = Field(
            default="all-documents",
            description="ChromaDB collection name"
        )
        top_k: int = Field(
            default=3,
            description="Number of most relevant chunks to retrieve"
        )

    def __init__(self):
        self.type = "filter"
        # self.name = "RAG"
        # self.type = "manifold"
        # self.id = "rag"

        self.valves = self.Valves()
        self.retrieval = None


    def _initialize_retrieval(self):
        """Initialize ChromaDB client and collection"""
        try:
            chromadb.api.client.SharedSystemClient.clear_system_cache()
            self.retrieval = Retrieval(dbdir=self.valves.chroma_path, log=True)
            chromadb.api.client.SharedSystemClient.clear_system_cache()
            print(f"ChromaDB initialized: {self.valves.collection_name}")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            self.retrieval = None

    async def on_startup(self):
        print("on startup")
        self._initialize_retrieval()
        pass

    async def on_shutdown(self):
        print("on shutdown")
        self.retrieval = None
        pass

    async def on_valves_updated(self):
        self._initialize_retrieval()
        pass

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Intercept incoming requests and augment with RAG context
        """

        if not self.valves.enabled:
            print("RAG disabled")
            return body

        if not self.retrieval:
            print("ChromaDB not initialized, skipping RAG")
            return body

        if body["messages"][-1]["role"] != "user":
            return

        if "__user__" not in body:
            body["__user__"] = {}
        chat_id = body.get("chat_id")
        chat_id = str(chat_id)
        if chat_id not in body["__user__"]:
            body["__user__"][chat_id] = set()
        history = body["__user__"][chat_id]

        print(chat_id)
        print(history)

        try:
            user_query = body["messages"][-1]["content"]
            print("USER QUERY", user_query)
            augmented_content = self.retrieval.augment(user_query, self.valves.top_k, history=history, log=True)
            body["messages"][-1]["content"] = augmented_content
        except Exception as e:
            print(f"Error during RAG retrieval: {e}")
            # If RAG fails, continue with original query
            pass

        return body
