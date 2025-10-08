#====================================
# Load AND Index stored embeddings 
#====================================
import json
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from spellchecker import SpellChecker
from keybert import KeyBERT
from google import genai


# Load your stored embeddings
with open(r"C:\Users\Shahe\my_server\RAG\rag_sys\political\paragraphs_with_embeddings.json") as f:
    paragraphs = json.load(f)  

with open(r"C:\Users\Shahe\my_server\RAG\rag_sys\political\sentences_with_embeddings.json") as f:
    sentences = json.load(f)

with open(r"C:\Users\Shahe\my_server\RAG\rag_sys\political\new_chunks_with_embeddings.json", "r", encoding="utf-8") as f:
    chunks = json.load(f) 

query_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

# Convert to numpy arrays
para_embeddings = np.array([p["paragraph_embedding"] for p in paragraphs]).astype("float32")
sent_embeddings = np.array([s["embedding"] for s in sentences]).astype("float32")
chunk_embeddings = np.array([c["embedding"] for c in chunks]).astype("float32")

# Build FAISS indexes
para_index = faiss.IndexFlatIP(para_embeddings.shape[1])  # cosine similarity (inner product)
sent_index = faiss.IndexFlatIP(sent_embeddings.shape[1])
chunk_index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
# Normalize for cosine similarity
faiss.normalize_L2(para_embeddings)
faiss.normalize_L2(sent_embeddings)
faiss.normalize_L2(chunk_embeddings) 

para_index.add(para_embeddings)
sent_index.add(sent_embeddings)
chunk_index.add(chunk_embeddings)
#====================================
# Retrieve Top-K Paragraphs
#====================================
def retrieve_paragraphs(query, top_k, threshold):
    q_emb = query_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    scores, idxs = para_index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if score >= threshold:
            results.append({
                "id": paragraphs[idx]["paragraph_id"],
                "sentences": paragraphs[idx]["sentences"],
                "text": paragraphs[idx]["text"],
                "score": float(score)
            })
    return results

#====================================
# Retrieve Top-K Chunks
#====================================
def retrieve_chunks(query, top_k, threshold):
    # embed & normalize query
    q_emb = query_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb) 

    scores, idxs = chunk_index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if score >= threshold:
            results.append({
                "topic": chunks[idx]["topic"], 
                "sentences": [s["sentence"] for s in chunks[idx]["sentences"]],
                "text": chunks[idx]["text"],
                "score": float(score)
            })
    return results

#====================================
# Collect Unique Sentences
#====================================
def collected_sentences(top_chunks,top_paragraphs):
    all_sentences = []
    # collect from chunks
    for ch in top_chunks:
        if "sentences" in ch:
            all_sentences.extend(ch["sentences"])

    # collect from paragraphs
    for para in top_paragraphs:
        if "sentences" in para:
            all_sentences.extend(para["sentences"])

    # deduplicate (preserve order)
    seen = set()
    unique_sentences = []
    for s in all_sentences:
        s_clean = s.strip()
        if s_clean not in seen:
            seen.add(s_clean)
            unique_sentences.append(s_clean)

    return unique_sentences

#====================================
# Re-rank with cross-encoder
#====================================
# Load a cross-encoder model (choose any you like)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_sentences(unique_sentences, query):
    # Prepare (query, sentence) pairs for scoring
    pairs = [(query, sent) for sent in unique_sentences]

    # Get scores
    scores = cross_encoder.predict(pairs)

    # Combine & sort by score
    ranked = sorted(
        zip(unique_sentences, scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_n = ranked[:10]

    for sent, score in top_n:
        print(f"Score: {score:.4f} | Sentence: {sent}")
    return top_n

#====================================
# Respose Generation
#====================================
def generate_response(top_n):
    # 1. Configure Gemini with your API key (export GEMINI_API_KEY first)
    client = genai.Client(api_key="AIzaSyBCEudgyUMmAVoTlx9fEXbaynRhVBoPPvg".strip())
    sentences = [s for s, _ in top_n]
    #sentences = [item["text"] for item in top_sentences]

    # 3. Build the prompt
    prompt = (
        "Act as a professional AI assistant. "
        "You are given the most relevant sentences from a document. "
        "Write a clear, cohesive, and human-readable answer to the user query "
        "based on these sentences.\n\n"
        "Relevant sentences:\n"
        + "\n".join(f"- {s}" for s in sentences) +
        "\n\nAnswer:"
    )
    # 4. Call Gemini (use gemini-1.5-flash or gemini-2.0-pro if available)
    response = client.models.generate_content( model="gemini-2.5-flash", contents=prompt)

    # 5. Output the generated text
    return response.text
#====================================
# Query Process & Keyword Extraction
#====================================
spell = SpellChecker()
def clean_text_with_spellcheck(text: str) -> str:
    text = text.lower()
    tokens = re.split(r"[\s\-]+", text)
    corrected_tokens = [spell.correction(t) or t for t in tokens]
    return " ".join(corrected_tokens)
    
kw_model = KeyBERT('all-MiniLM-L6-v2')
def extract_query_keywords(text: str, top_n: int = 5, ngram_range=(1, 3), stop_words='english') -> list:
    # 1. Extract keywords from a user query using KeyBERT embeddings.
        cleaned_text = clean_text_with_spellcheck(text)
        # Extract keywords with KeyBERT
        keywords = kw_model.extract_keywords(
                cleaned_text,
                keyphrase_ngram_range=ngram_range,
                stop_words=stop_words,
                top_n=top_n
            )
            
        # keywords is a list of tuples: (keyword, score)
        keyword_list = [kw for kw, score in keywords]
            
        return keyword_list
#====================================
# Full Pipeline
#====================================
if __name__ == "__main__":
    while True:
        user_query = input("\nEnter your query (or 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            print("Exiting...")
            break
        

        #user_query = normalize_query(keywords)
        keywords = extract_query_keywords(user_query)    
        print("Extracted Keywords:", keywords)
        normalized_keywords = [kw.lower().strip() for kw in keywords]
        normalized_query = " ".join(normalized_keywords)
        top_paragraphs = retrieve_paragraphs(normalized_query, top_k=7, threshold=0.5)
        top_chunks = retrieve_chunks(normalized_query, top_k=7, threshold=0.5)
                
        # Collect & rerank
        unique_sentences = collected_sentences(top_chunks, top_paragraphs)
        if len(unique_sentences) == 0:
            print( "⚠️ No relevant sentences found above the confidence threshold.")
            break
        print(f"\nUnique Sentences Collected: {len(unique_sentences)}")
        top_n = rerank_sentences(unique_sentences, user_query)
        

        # Generate answer for this query
        response = generate_response(top_n)
        print("\n=== Generated Response ===")
        print(response)
        print("\n" + "="*50 + "\n")