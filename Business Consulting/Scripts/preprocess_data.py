# scripts/preprocess_data.py
import os
import pandas as pd
import json
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Ensure your .env file is in the parent directory (project root)
# when running this script from the 'scripts' directory.
# (e.g., Domain Project/.env)
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

try:
    client = OpenAI() # Expects OPENAI_API_KEY in environment
except Exception as e:
    print(f"Error initializing OpenAI client: {e}. Make sure OPENAI_API_KEY is set in your .env file.")
    client = None
    sys.exit(1) # Exit if client can't be initialized

# --- DEFINE YOUR CSV FILES TO PREPROCESS ---
# This list should contain the CSVs you want to include in your knowledge base.
# Ensure these files are in your '../data/processed/' directory relative to this script's location.
# (i.e., Domain Project/data/processed/your_file.csv)
#
# IMPORTANT: Review this list and include ONLY the CSVs relevant to your business consultant AI.
# Processing many large files will take a long time and incur embedding costs.
csv_files_to_preprocess = [
    "startup data.csv", # Example, replace with your actual chosen files
    "stores_sales_forecasting.csv"
   
]
# --- ------------------------------------ ---

# Output file will be in 'Domain Project/data/knowledge_base_with_embeddings.json'
OUTPUT_EMBEDDINGS_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge_base_with_embeddings.json')
# Raw data directory is 'Domain Project/data/processed/'
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


def get_embedding_for_preprocessing(text_to_embed, model="text-embedding-3-small"):
    """Generates an embedding for the given text using OpenAI."""
    if not client:
        # This case should ideally be handled by the sys.exit(1) if client fails to initialize
        print("OpenAI client not available for embedding.")
        return None
    try:
        text_to_embed = str(text_to_embed).replace("\n", " ") # API recommendation
        if not text_to_embed.strip(): # Handle empty or whitespace-only strings
            return None
        response = client.embeddings.create(input=[text_to_embed], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding for text '{str(text_to_embed)[:50]}...': {e}")
        return None

def main():
    if not client:
        print("OpenAI client failed to initialize. Exiting preprocessing.")
        return

    processed_knowledge_base = []
    print("Starting preprocessing of CSV files to generate embeddings...")
    print(f"Looking for CSVs in: {RAW_DATA_DIR}")
    print(f"Output will be saved to: {OUTPUT_EMBEDDINGS_FILE}")

    for csv_filename in csv_files_to_preprocess:
        csv_file_path = os.path.join(RAW_DATA_DIR, csv_filename)

        if not os.path.exists(csv_file_path):
            print(f"SKIPPING: CSV file not found: {csv_file_path}")
            continue

        try:
            try:
                df = pd.read_csv(csv_file_path, low_memory=False, encoding='utf-8')
                print(f"\nSuccessfully read {csv_filename} with UTF-8.")
            except UnicodeDecodeError:
                print(f"UTF-8 decoding failed for {csv_filename}. Trying 'latin1' encoding...")
                df = pd.read_csv(csv_file_path, low_memory=False, encoding='latin1')
                print(f"Successfully read {csv_filename} with 'latin1'.")
            
            print(f"Processing {csv_filename} (Shape: {df.shape})...")
            file_items_embedded_count = 0
            for index, row in df.iterrows():
                row_texts = []
                for col_name in df.columns:
                    col_value = row[col_name]
                    if pd.notna(col_value) and str(col_value).strip() != "":
                        row_texts.append(f"{str(col_name).replace('_', ' ').title()}: {str(col_value).strip()}")
                
                combined_text_chunk = ", ".join(row_texts)
                if not combined_text_chunk:
                    continue
                
                embedding = get_embedding_for_preprocessing(combined_text_chunk)
                if embedding:
                    processed_knowledge_base.append({
                        "text": combined_text_chunk,
                        "embedding": embedding,
                        "source": csv_filename
                    })
                    file_items_embedded_count += 1
                    if file_items_embedded_count % 100 == 0 and file_items_embedded_count > 0:
                        print(f"  Embedded {file_items_embedded_count} rows from {csv_filename}...")
            
            if file_items_embedded_count > 0:
                print(f"Successfully processed and embedded {file_items_embedded_count} items from {csv_filename}.")
            else:
                print(f"No valid text content found or embedded from {csv_filename}.")


        except Exception as e:
            print(f"Error processing file {csv_file_path}: {e}")

    # Save the processed knowledge base with embeddings to a JSON file
    if processed_knowledge_base:
        try:
            output_dir = os.path.dirname(OUTPUT_EMBEDDINGS_FILE)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")

            with open(OUTPUT_EMBEDDINGS_FILE, 'w', encoding='utf-8') as f_out:
                json.dump(processed_knowledge_base, f_out, indent=4) # Added indent for readability
            print(f"\nSuccessfully saved knowledge base with {len(processed_knowledge_base)} items to: {OUTPUT_EMBEDDINGS_FILE}")
        except Exception as e:
            print(f"Error saving embeddings to JSON file: {e}")
    else:
        print("\nNo data was processed to save to the knowledge base file.")

if __name__ == "__main__":
    main()