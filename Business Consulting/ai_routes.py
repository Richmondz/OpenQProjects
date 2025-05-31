# C:\Users\fishe\OneDrive\Documents\Domain Project\ai_routes.py
import os
import sqlite3
from datetime import datetime
import json
import pandas as pd 
import numpy as np
import logging # Import the logging module
from flask import Blueprint, request, jsonify, Response
from openai import OpenAI
from dotenv import load_dotenv

# Get a logger for this module
logger = logging.getLogger(__name__)

load_dotenv()

try:
    client = OpenAI()
    logger.info("OpenAI client initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {e}", exc_info=True)
    client = None 

ai_bp = Blueprint('ai_bp', __name__)

# --- Database Caching Setup ---
DATABASE_FILE = 'ai_cache.db'
def get_db_connection():
    conn = sqlite3.connect(DATABASE_FILE, check_same_thread=False) 
    conn.row_factory = sqlite3.Row
    return conn

def init_cache_db():
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_cache (
                prompt_text TEXT PRIMARY KEY,
                response_text TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        logger.info("Cache database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing cache database: {e}", exc_info=True)
    finally:
        if conn: conn.close()
init_cache_db()


# --- RAG: Knowledge Base Setup ---
knowledge_base_items = [] 
EMBEDDINGS_FILE_PATH = os.path.join('data', 'knowledge_base_with_embeddings.json') 

def get_embedding(text_to_embed, model="text-embedding-3-small"):
    if not client:
        logger.warning("get_embedding called but OpenAI client not initialized.")
        return None
    try:
        text_to_embed = str(text_to_embed).replace("\n", " ")
        if not text_to_embed.strip(): 
            logger.debug("get_embedding called with empty text.")
            return None
        # logger.debug(f"Getting embedding for text: {text_to_embed[:60]}...") # Optional: very verbose
        response = client.embeddings.create(input=[text_to_embed], model=model)
        # logger.debug("Embedding received from OpenAI.")
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding for text '{str(text_to_embed)[:50]}...': {e}", exc_info=True)
        return None

def load_precomputed_knowledge():
    global knowledge_base_items
    logger.info(f"Attempting to load pre-computed knowledge from: {EMBEDDINGS_FILE_PATH}")
    if not os.path.exists(EMBEDDINGS_FILE_PATH):
        logger.error(f"Embeddings file not found: {EMBEDDINGS_FILE_PATH}")
        logger.warning("Please run the preprocess_data.py script first in the 'scripts' folder to generate it.")
        knowledge_base_items = [] 
        return
    try:
        with open(EMBEDDINGS_FILE_PATH, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        temp_knowledge_base = []
        for item in loaded_data:
            if item.get("embedding") and isinstance(item["embedding"], list):
                item["embedding"] = np.array(item["embedding"], dtype=float)
                temp_knowledge_base.append(item)
            else:
                logger.warning(f"Item skipped due to missing or invalid embedding: {str(item.get('text', 'N/A'))[:50]}")
        knowledge_base_items = temp_knowledge_base
        logger.info(f"Successfully loaded {len(knowledge_base_items)} items with valid embeddings from pre-computed knowledge base: {EMBEDDINGS_FILE_PATH}")

    except Exception as e:
        logger.error(f"Error loading pre-computed knowledge base from {EMBEDDINGS_FILE_PATH}: {e}", exc_info=True)
        knowledge_base_items = [] 
load_precomputed_knowledge() 


def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None: return 0.0
    np_vec1 = np.array(vec1, dtype=float) if not isinstance(vec1, np.ndarray) else vec1
    np_vec2 = np.array(vec2, dtype=float) if not isinstance(vec2, np.ndarray) else vec2
    dot_product = np.dot(np_vec1, np_vec2)
    norm_vec1 = np.linalg.norm(np_vec1)
    norm_vec2 = np.linalg.norm(np_vec2)
    if norm_vec1 == 0 or norm_vec2 == 0: return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

# --- AI Endpoint ---
@ai_bp.route("/api/ask", methods=["POST"])
def ask_ai():
    logger.debug("/api/ask route entered.") 

    if not client:
        logger.error("OpenAI client not initialized in /api/ask. Returning 500.")
        return jsonify({"error": "OpenAI client not initialized. Check API key and server logs."}), 500

    data = request.get_json()
    logger.debug(f"Received data: {data}") 

    if not data or "prompt" not in data:
        logger.warning("No data or prompt in request. Returning 400.")
        return jsonify({"error": "No prompt provided"}), 400
    
    user_prompt = data["prompt"].strip()
    service_context = data.get("service_context") 
    logger.info(f"User prompt: '{user_prompt[:50]}...', Service context: {service_context}") 


    if not user_prompt:
        logger.warning("User prompt is empty. Returning 400.")
        return jsonify({"error": "Prompt cannot be empty"}), 400

    db_conn_cache_check = None
    try:
        logger.debug("Attempting cache check.") 
        db_conn_cache_check = get_db_connection()
        cursor = db_conn_cache_check.cursor()
        cursor.execute("SELECT response_text FROM ai_cache WHERE prompt_text = ?", (user_prompt,))
        cached_result = cursor.fetchone()

        if cached_result:
            if db_conn_cache_check: db_conn_cache_check.close()
            logger.info(f"Serving response from cache for '{user_prompt[:50]}...'.") 
            return jsonify({"response": cached_result["response_text"], "source": "cache"})

        logger.info(f"Prompt not found in cache. Proceeding with RAG for '{user_prompt[:50]}...'.") 

        user_prompt_embedding = get_embedding(user_prompt)
        logger.debug(f"User prompt embedding generated (is None? {user_prompt_embedding is None}).") 

        relevant_chunks_text = []
        if user_prompt_embedding and knowledge_base_items:
            logger.debug(f"Knowledge base has {len(knowledge_base_items)} items. Starting similarity search.") 
            chunk_similarities = []
            for i, item in enumerate(knowledge_base_items):
                if item.get("embedding") is not None:
                    similarity = cosine_similarity(user_prompt_embedding, item["embedding"])
                    chunk_similarities.append({"text": item["text"], "source": item.get("source"), "score": similarity, "item_index": i})
            
            chunk_similarities.sort(key=lambda x: x["score"], reverse=True)
            logger.debug(f"Top 3 similarities (before threshold): {chunk_similarities[:3]}") 
            
            top_n = 3 
            similarity_threshold = 0.6 
            for chunk_info in chunk_similarities[:top_n]:
                if chunk_info["score"] > similarity_threshold: 
                    relevant_chunks_text.append(f"- From {chunk_info['source']} (Item Index: {chunk_info.get('item_index', 'N/A')}, Relevance: {chunk_info['score']:.2f}):\n  {chunk_info['text']}")
            
            logger.debug(f"Found {len(relevant_chunks_text)} relevant chunks after threshold.") 
        
        elif not user_prompt_embedding:
            logger.warning("User prompt embedding is None, cannot perform RAG.")
        else: 
            logger.warning("Knowledge base is empty, cannot perform RAG.")


        retrieved_context_str = "\n\n".join(relevant_chunks_text) if relevant_chunks_text else "No highly relevant pre-existing knowledge found for this query in the loaded knowledge base."
        logger.info(f"Retrieved context for RAG ({len(relevant_chunks_text)} chunks):\n{retrieved_context_str[:300]}...") 
        
        system_message_content = "You are a general AI Business Advisor." 
        if service_context == "market_strategy": system_message_content = ("You are an expert in Market & Growth Strategy...")
        elif service_context == "operational_efficiency": system_message_content = ("You specialize in Operational Efficiency...")
        elif service_context == "digital_transformation": system_message_content = ("You are a Digital Transformation consultant...")
        elif service_context == "startup_advisory": system_message_content = ("You are a Startup Advisor...")
        
        rag_instructed_system_message = (
            f"{system_message_content}\n\n"
            "When answering the user's question, first consider the 'Retrieved Context' provided below. "
            "If this context is relevant and helps answer the question, integrate it into your response and you can mention that your information is based on provided data. "
            "If the context is not relevant or insufficient, rely on your general knowledge as a business consultant. "
            "Strive for comprehensive, actionable answers. Please format your response using Markdown, including lists and bolding for emphasis where appropriate."
        )
        messages_for_openai = [
            {"role": "system", "content": rag_instructed_system_message},
            {"role": "user", "content": f"User Question: {user_prompt}\n\nRetrieved Context for your consideration:\n{retrieved_context_str}"}
        ]
        
        logger.info(f"Streaming response for '{user_prompt[:50]}...' (context: {service_context}) with RAG from OpenAI.") 
        
        def generate_response_chunks():
            full_ai_response_for_caching = []
            try:
                stream = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages_for_openai, stream=True)
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content is not None:
                        full_ai_response_for_caching.append(content)
                        yield f"data: {json.dumps({'delta': content})}\n\n"
                final_response_text = "".join(full_ai_response_for_caching)
                if final_response_text: 
                    db_conn_caching = None
                    try:
                        db_conn_caching = get_db_connection()
                        cache_cursor = db_conn_caching.cursor()
                        cache_cursor.execute("INSERT INTO ai_cache (prompt_text, response_text, timestamp) VALUES (?, ?, ?)",
                                           (user_prompt, final_response_text, datetime.now()))
                        db_conn_caching.commit()
                        logger.info(f"Cached RAG response for '{user_prompt[:50]}...'.")
                    except sqlite3.IntegrityError: 
                        logger.warning(f"Cache insert failed (IntegrityError) for RAG response '{user_prompt[:50]}...'.")
                    except Exception as e_cache_insert:
                        logger.error(f"Cache insert error for RAG response '{user_prompt[:50]}...': {e_cache_insert}", exc_info=True)
                    finally: 
                        if db_conn_caching: db_conn_caching.close()
                yield f"data: {json.dumps({'event': 'done'})}\n\n"
            except Exception as e_stream:
                logger.error(f"Error during RAG OpenAI stream for '{user_prompt[:50]}...': {str(e_stream)}", exc_info=True)
                yield f"data: {json.dumps({'error': str(e_stream)})}\n\n"

        if db_conn_cache_check:
            db_conn_cache_check.close()
            db_conn_cache_check = None 
        
        return Response(generate_response_chunks(), mimetype='text/event-stream')

    except Exception as e_outer:
        logger.error(f"Outer error in /api/ask for '{user_prompt[:50]}...': {str(e_outer)}", exc_info=True)
        if db_conn_cache_check: 
            db_conn_cache_check.close()
        return jsonify({"error": str(e_outer)}), 500
