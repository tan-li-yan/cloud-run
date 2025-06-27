from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.sql.connector import Connector
import sqlalchemy
import pandas as pd
import os, json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import vertexai
from vertexai.language_models import TextEmbeddingModel
import time

app = Flask(__name__)

# Firebase
cred = credentials.Certificate("acme-project55-97277b000d98.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# SQL Connection Info
db_user = 'media-data'
db_pass = 'Cabbage@0506'
db_name = 'media_metadata'
project_id = 'acme-test-463507'
region = 'us-central1'
instance_name = 'media-data'

# Project information
PROJECT_ID = "acme-test-463507"
LOCATION = "us-central1"

# SQL Connector Setup
INSTANCE_CONNECTION_NAME = f"{project_id}:{region}:{instance_name}"
connector = Connector()

def getconn():
    return connector.connect(
        INSTANCE_CONNECTION_NAME, "pymysql",
        user=db_user, password=db_pass, db=db_name
    )

pool = sqlalchemy.create_engine("mysql+pymysql://", creator=getconn)

vertexai.init(project=PROJECT_ID, location=LOCATION)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

# Cleaning helper
def clean(text):
    if isinstance(text, list):
        text = " ".join(text)
    return text.lower().strip()

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    # Retrieve user data
    user_ref = db.collection("userID").document(user_id)
    user_data = user_ref.get().to_dict() or {}
    interest = clean(user_data.get("interest", ''))
    search = clean(user_data.get("search_history", []))

    # Watch history
    watch_query = user_ref.collection("watchHistory").order_by("watchedTime", direction=firestore.Query.DESCENDING).limit(5)
    docs = watch_query.stream()
    watched_medias = [doc.to_dict().get('mediaID') for doc in docs]
    if not watched_medias:
        return jsonify({"error": "No watch history found."}), 404

    # Retrieve keywords for watched videos
    media_ids_str = ",".join(f"'{mid}'" for mid in watched_medias)
    with pool.connect() as conn:
        df_watched = pd.read_sql(f"SELECT keywords FROM media WHERE media_id IN ({media_ids_str})", conn)

    keywords = ', '.join(df_watched['keywords'].dropna().tolist())
    unique_keywords = list(set(word.strip() for word in keywords.split(',')))
    watched_clean = " ".join(clean(k) for k in unique_keywords)

    user_text = f"{interest} {search} {watched_clean}"

    # Embed user text using the correct API
    try:
        # # Initialize the client (only once, move this outside if reused)
        # client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        
        # # Embed user input
        # user_embedded = client.models.embed_content(
        #     model="text-embedding-005",
        #     contents=user_text,
        #     config=genai.EmbedContentConfig(
        #         task_type="RETRIEVAL_QUERY",  
        #         output_dimensionality=768,   # Highest
        #     ),
        # )
        user_embedded = embedding_model.get_embeddings([user_text])[0]

        user_vec = np.array(user_embedded.values).reshape(1, -1)

    except Exception as e:
        return jsonify({"error": f"Embedding failed: {str(e)}"}), 500

    # Retrieve and compare all media embeddings
    with pool.connect() as conn:
        df_all = pd.read_sql("SELECT media_id, embedding FROM media WHERE embedding IS NOT NULL", conn)

    df_all["embedding"] = df_all["embedding"].apply(lambda x: json.loads(x))
    recommendations = []
    for _, row in df_all.iterrows():
        video_vec = np.array(row["embedding"]).reshape(1, -1)
        score = cosine_similarity(user_vec, video_vec)[0][0]
        recommendations.append((row["media_id"], score))

    top_6 = sorted(recommendations, key=lambda x: x[1], reverse=True)[:6]
    return jsonify({"user": user_id, "recommendations": [media_id for media_id, _ in top_6]})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200


print("Starting Flask app...")
print(f"PORT environment variable: {os.environ.get('PORT', 'Not set')}")

if __name__ == "__main__":
    # Add a delay to ensure all services are ready
    time.sleep(2)  # Wait 2 seconds before starting
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)

