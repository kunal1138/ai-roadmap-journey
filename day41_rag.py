# ============================================================
# DAY 41: RAG - Retrieval Augmented Generation 📚
# Build Your Own RAG System!
# ============================================================

# If needed, run this once in Google Colab:
# !pip install -q groq sentence-transformers faiss-cpu

import numpy as np
import time
import faiss

from groq import Groq
from sentence_transformers import SentenceTransformer
from google.colab import userdata


# ============================================================
# START
# ============================================================

print("RAG System! 📚")
print("Retrieval Augmented Generation!")
print("=" * 60)


# ============================================================
# PART 0: GROQ SETUP
# ============================================================

print("\n" + "=" * 60)
print("PART 0: GROQ SETUP")
print("=" * 60)

try:
    api_key = userdata.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY was not found."
        )

    client = Groq(api_key=api_key)

    print("✅ Groq API connected!")

except Exception as e:
    print("❌ Could not connect to Groq.")
    print(f"Error: {e}")
    raise


# ============================================================
# MODEL AUTO-DETECTION
# ============================================================

def get_model(client):

    preferred_models = [
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768"
    ]

    try:
        models = client.models.list()

        available_models = [
            model.id for model in models.data
        ]

        # Try preferred models first
        for model_name in preferred_models:
            if model_name in available_models:
                return model_name

        # Otherwise find a text-generation model
        if available_models:
            return available_models[0]

        raise ValueError("No Groq models available.")

    except Exception as e:
        print("⚠️ Could not automatically detect model.")
        print(f"Error: {e}")

        # Safe fallback
        return "openai/gpt-oss-120b"


MODEL = get_model(client)

print(f"🤖 Model: {MODEL}")


# ============================================================
# PART 1: KNOWLEDGE BASE
# ============================================================

print("\n" + "=" * 60)
print("PART 1: KNOWLEDGE BASE")
print("=" * 60)


documents = [

    """
    Machine Learning is a subset of Artificial Intelligence
    that enables computers to learn from data without being
    explicitly programmed.

    Main types of Machine Learning:
    1. Supervised Learning
    2. Unsupervised Learning
    3. Reinforcement Learning
    """,

    """
    Deep Learning uses artificial neural networks with many
    layers. It is especially powerful for image recognition,
    natural language processing, and speech processing.

    Important architectures include:
    CNN, RNN, and Transformer.
    """,

    """
    CNN (Convolutional Neural Network) is designed mainly
    for image processing.

    CNNs commonly use:
    - Conv2D layers to detect patterns
    - MaxPooling to reduce spatial dimensions
    - Dense layers for classification

    A CNN model achieved 99.32% accuracy on the MNIST dataset.
    """,

    """
    BERT stands for Bidirectional Encoder Representations
    from Transformers.

    BERT reads text in both directions and understands the
    relationship between words in a sentence.

    It was pre-trained on large text datasets and can be
    fine-tuned for specific Natural Language Processing tasks.
    """,

    """
    RAG stands for Retrieval Augmented Generation.

    RAG combines information retrieval with text generation.

    The system first retrieves relevant documents from a
    knowledge base and then provides those documents as
    context to an LLM.

    RAG helps an LLM answer questions using external or
    custom knowledge.
    """,

    """
    Kunal Choudhari is a final year B.Sc CS student from
    Nagpur, Maharashtra.

    He is learning AI/ML through a structured 15-month roadmap.

    He studies approximately 1-2 hours daily using an Android
    mobile phone and Google Colab.
    """,

    """
    Kunal completed:

    Phase 1:
    - Python basics
    - Object-Oriented Programming

    Phase 2:
    - KNN
    - Decision Trees
    - Random Forest
    - SVM
    - K-Means

    Phase 3:
    - Neural Networks
    - CNN

    His CNN achieved 99.32% accuracy on the MNIST dataset.
    """,

    """
    Kunal has 2 Anthropic certifications:

    1. AI Fluency for Students
    2. Claude with Amazon Bedrock

    He is currently in Phase 4, learning:

    - NLP
    - Transformers
    - BERT
    - Hugging Face
    - LLM APIs
    - RAG
    """,

    """
    Kunal's friend group includes:

    Sonu
    Glou
    Vedu
    Tantan
    Durgesh
    Om
    Sam
    Abhinav

    He builds creative coding projects using their names.

    His GitHub repository is:
    ai-roadmap-journey
    """,

    """
    Cricket is the most popular sport in India.

    Virat Kohli is known for aggressive batting.
    Rohit Sharma is famous for six-hitting.
    MS Dhoni is a legendary captain and finisher.

    IPL is one of the most popular cricket leagues.
    """,

    """
    Nagpur is famous for its oranges and cricket.

    It has a major cricket stadium called VCA Stadium.

    Nagpur is located in Maharashtra in Central India.

    The city is commonly known as the Orange City of India.
    """,

    """
    Python is one of the most popular programming languages
    for Artificial Intelligence and Data Science.

    Important libraries include:

    NumPy:
    Used for numerical arrays and mathematical operations.

    Pandas:
    Used for data manipulation and analysis.

    Matplotlib:
    Used for data visualization.

    Scikit-learn:
    Used for Machine Learning.

    TensorFlow and Keras:
    Used for Deep Learning.
    """
]


print(f"📚 Total documents: {len(documents)}")

print("\nSample document:")
print(documents[0][:150] + "...")


# ============================================================
# PART 2: CREATE EMBEDDINGS
# ============================================================

print("\n" + "=" * 60)
print("PART 2: CREATE EMBEDDINGS")
print("=" * 60)

print("\nLoading embedding model...")

try:

    embedder = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("✅ Embedding model loaded!")

except Exception as e:

    print("❌ Could not load embedding model.")
    print(f"Error: {e}")
    raise


print("\nCreating document embeddings...")

doc_embeddings = embedder.encode(
    documents,
    show_progress_bar=True,
    normalize_embeddings=True
)

doc_embeddings = np.asarray(
    doc_embeddings,
    dtype=np.float32
)

print(
    f"\nEmbedding shape: {doc_embeddings.shape}"
)

print(
    f"📊 {len(documents)} documents × "
    f"{doc_embeddings.shape[1]} dimensions"
)


# ============================================================
# PART 3: VECTOR DATABASE - FAISS
# ============================================================

print("\n" + "=" * 60)
print("PART 3: VECTOR DATABASE (FAISS)")
print("=" * 60)

dimension = doc_embeddings.shape[1]

# IndexFlatIP + normalized vectors
# = cosine similarity

index = faiss.IndexFlatIP(dimension)

index.add(doc_embeddings)

print("✅ FAISS index created!")
print(f"📦 Total vectors stored: {index.ntotal}")

print("""
FAISS = Facebook AI Similarity Search

It:
→ Stores embeddings as vectors
→ Searches similar vectors
→ Performs fast similarity search

We use:

IndexFlatIP
+
Normalized embeddings
=
Cosine similarity
""")


# ============================================================
# PART 4: RETRIEVAL
# ============================================================

print("\n" + "=" * 60)
print("PART 4: RETRIEVAL FUNCTION")
print("=" * 60)


def retrieve_relevant_docs(query, top_k=3):

    # Prevent top_k from being larger than
    # the number of documents
    top_k = min(top_k, len(documents))

    # Convert query into embedding
    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for i, idx in enumerate(indices[0]):

        # Safety check
        if idx < 0 or idx >= len(documents):
            continue

        results.append({
            "doc": documents[idx],
            "similarity": float(scores[0][i]),
            "index": int(idx)
        })

    return results


# ============================================================
# TEST RETRIEVAL
# ============================================================

test_queries = [

    "What is CNN?",

    "Tell me about Kunal",

    "What is Nagpur famous for?"
]


print("\n🔎 Testing retrieval:")

for query in test_queries:

    print("\n" + "-" * 50)
    print(f"Query: {query}")

    results = retrieve_relevant_docs(
        query,
        top_k=2
    )

    for i, result in enumerate(results):

        print(
            f"\n📄 Result {i + 1}"
        )

        print(
            f"Similarity: "
            f"{result['similarity']:.3f}"
        )

        print(
            f"Document index: "
            f"{result['index']}"
        )

        preview = " ".join(
            result["doc"].split()
        )

        print(
            f"Text: {preview[:120]}..."
        )


# ============================================================
# PART 5: COMPLETE RAG PIPELINE
# ============================================================

print("\n" + "=" * 60)
print("PART 5: COMPLETE RAG PIPELINE")
print("=" * 60)


def generate_rag_response(
    question,
    top_k=3,
    max_tokens=300
):

    # --------------------------------------------------------
    # STEP 1: RETRIEVAL
    # --------------------------------------------------------

    relevant_docs = retrieve_relevant_docs(
        question,
        top_k=top_k
    )

    if not relevant_docs:

        return {
            "answer": "I don't have information about that.",
            "sources": [],
            "num_sources": 0
        }


    # --------------------------------------------------------
    # STEP 2: BUILD CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for i, doc in enumerate(relevant_docs):

        context_parts.append(
            f"""
SOURCE {i + 1}
Document ID: {doc['index']}
Similarity: {doc['similarity']:.3f}

{doc['doc']}
"""
        )

    context = "\n".join(context_parts)


    # --------------------------------------------------------
    # STEP 3: RAG PROMPT
    # --------------------------------------------------------

    rag_prompt = f"""
You are a Retrieval Augmented Generation assistant.

Your job is to answer the user's question using ONLY
the information provided in the sources below.

IMPORTANT RULES:

1. Use only the provided sources.
2. Do not use outside knowledge.
3. Do not invent facts.
4. If the answer is not clearly available in the sources,
   say exactly:

"I don't have information about that."

5. Give a short and clear answer.
6. When useful, mention which source supports the answer.

SOURCES:
{context}

USER QUESTION:
{question}

ANSWER:
"""


    # --------------------------------------------------------
    # STEP 4: CALL GROQ
    # --------------------------------------------------------

    try:

        response = client.chat.completions.create(

            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a factual RAG assistant. "
                        "Only use the provided context."
                    )
                },
                {
                    "role": "user",
                    "content": rag_prompt
                }
            ],

            max_tokens=max_tokens,

            temperature=0
        )

        answer = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )


    except Exception as e:

        answer = (
            f"❌ Groq API error: {e}"
        )


    # --------------------------------------------------------
    # STEP 5: RETURN EVERYTHING
    # --------------------------------------------------------

    return {

        "answer": answer,

        "sources": relevant_docs,

        "num_sources": len(relevant_docs)
    }


# ============================================================
# PART 6: TEST RAG
# ============================================================

print("\n" + "=" * 60)
print("PART 6: TEST RAG SYSTEM")
print("=" * 60)


questions = [

    "What is Kunal learning?",

    "What certifications does Kunal have?",

    "What is CNN and its accuracy?",

    "What is Nagpur famous for?",

    "Who are Kunal's friends?",

    "What Python libraries are used for ML?",

    "What is RAG?",

    "How does BERT work?",

    "What is the capital of France?"
]


for question in questions:

    print("\n" + "=" * 55)

    print(
        f"❓ Question: {question}"
    )

    result = generate_rag_response(
        question,
        top_k=3
    )

    print(
        f"\n📚 Sources used: "
        f"{result['num_sources']}"
    )

    print(
        f"🤖 Answer:\n"
        f"{result['answer']}"
    )

    print("\n📌 Retrieved sources:")

    for i, source in enumerate(
        result["sources"]
    ):

        print(
            f"  Source {i + 1}: "
            f"Doc {source['index']} "
            f"→ similarity "
            f"{source['similarity']:.3f}"
        )


# ============================================================
# PART 7: RAG vs WITHOUT RAG
# ============================================================

print("\n" + "=" * 60)
print("PART 7: RAG vs WITHOUT RAG")
print("=" * 60)


def ask_without_rag(
    question,
    max_tokens=150
):

    try:

        response = client.chat.completions.create(

            model=MODEL,

            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ],

            max_tokens=max_tokens,

            temperature=0
        )

        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

    except Exception as e:

        return f"❌ Groq API error: {e}"


test_questions = [

    "What is Kunal's GitHub repository?",

    "What accuracy did Kunal achieve on MNIST?",

    "Who is in Kunal's friend group?"
]


print("\n🔬 Comparing RAG vs No RAG:")


for question in test_questions:

    print("\n" + "=" * 55)

    print(
        f"❓ Question: {question}"
    )


    # Without RAG
    print("\n❌ WITHOUT RAG:")

    no_rag = ask_without_rag(
        question
    )

    print(no_rag)


    # With RAG
    print("\n✅ WITH RAG:")

    with_rag = generate_rag_response(
        question,
        top_k=3
    )

    print(
        with_rag["answer"]
    )


# ============================================================
# PART 8: CHUNKING STRATEGIES
# ============================================================

print("\n" + "=" * 60)
print("PART 8: CHUNKING STRATEGIES")
print("=" * 60)


def chunk_text(
    text,
    chunk_size=50,
    overlap=10
):

    words = text.split()

    chunks = []

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )


    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunk = " ".join(
            words[start:end]
        )

        chunks.append(chunk)

        # Stop when we reach the end
        if end >= len(words):
            break

        start = end - overlap


    return chunks


long_doc = """
Kunal Choudhari is a final year B.Sc CS
student from Nagpur, Maharashtra, India.

He started his AI ML learning journey
in August 2026.

He studies 1-2 hours daily using only
an Android mobile phone.

He uses Google Colab for coding.

Kunal has completed Python basics,
OOP, Data Science tools like NumPy
and Pandas.

He also completed Machine Learning,
covering KNN, Decision Trees,
Random Forest, SVM, K-Means clustering,
and model evaluation.

Then he completed Deep Learning with
Neural Networks, Dropout, BatchNorm,
Early Stopping, and CNN.

His CNN achieved 99.32% on the
MNIST dataset.

Now he is in Phase 4 learning NLP,
BERT, Transformers, Hugging Face,
LLM APIs and RAG.

He has GitHub repo:
ai-roadmap-journey.

He has 2 Anthropic certifications.

His friends are Sonu, Glou, Vedu,
Tantan, Durgesh, Om, Sam and Abhinav.
"""


chunks = chunk_text(
    long_doc,
    chunk_size=50,
    overlap=10
)


print(
    f"Original document words: "
    f"{len(long_doc.split())}"
)

print(
    f"Number of chunks: "
    f"{len(chunks)}"
)


print("\n📦 Sample chunks:")

for i, chunk in enumerate(
    chunks[:3]
):

    print(
        f"\nChunk {i + 1}:"
    )

    print(chunk)


print("""
============================================================

CHUNKING STRATEGIES

1. Fixed Size
   → Split every N words/tokens

2. Sentence Based
   → Split by sentences

3. Paragraph Based
   → Split by paragraphs

4. Semantic Chunking
   → Split according to meaning

OVERLAP

→ Keeps some information between chunks
→ Reduces context loss at boundaries
→ Helps retrieval preserve meaning

============================================================
""")


# ============================================================
# PART 9: CONVERSATIONAL RAG
# ============================================================

print("\n" + "=" * 60)
print("PART 9: CONVERSATIONAL RAG")
print("=" * 60)


def conversational_rag(
    question,
    chat_history=None,
    top_k=2
):

    # --------------------------------------------------------
    # FIX:
    # Do NOT use chat_history=[] as a default argument.
    # --------------------------------------------------------

    if chat_history is None:
        chat_history = []


    # --------------------------------------------------------
    # RETRIEVE DOCUMENTS
    # --------------------------------------------------------

    relevant_docs = retrieve_relevant_docs(
        question,
        top_k=top_k
    )


    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for i, doc in enumerate(
        relevant_docs
    ):

        context_parts.append(
            f"""
SOURCE {i + 1}:

{doc['doc']}
"""
        )

    context = "\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------------

    system_prompt = f"""
You are a conversational RAG assistant.

Use ONLY the provided knowledge-base context
to answer questions.

If the answer cannot be found in the context,
say:

"I don't have information about that."

Do not invent facts.

Knowledge Base Context:
{context}
"""


    # --------------------------------------------------------
    # BUILD MESSAGES
    # --------------------------------------------------------

    messages = [

        {
            "role": "system",
            "content": system_prompt
        }

    ]


    # Add previous conversation
    messages.extend(
        chat_history
    )


    # Add current question
    messages.append({

        "role": "user",

        "content": question

    })


    # --------------------------------------------------------
    # CALL GROQ
    # --------------------------------------------------------

    try:

        response = client.chat.completions.create(

            model=MODEL,

            messages=messages,

            max_tokens=250,

            temperature=0
        )

        answer = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )


    except Exception as e:

        answer = (
            f"❌ Groq API error: {e}"
        )


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    updated_history = list(
        chat_history
    )

    updated_history.append({

        "role": "user",

        "content": question

    })

    updated_history.append({

        "role": "assistant",

        "content": answer

    })


    return answer, updated_history


# ============================================================
# TEST CONVERSATIONAL RAG
# ============================================================

print("\n💬 Multi-turn RAG conversation:")
print("-" * 50)


history = []


conversation = [

    "Tell me about Kunal",

    "What phase is he currently in?",

    "What are his certifications?"
]


for question in conversation:

    print(
        f"\n👤 User: {question}"
    )


    answer, history = conversational_rag(

        question,

        chat_history=history,

        top_k=2

    )


    print(
        f"🤖 RAG: {answer}"
    )


# ============================================================
# PART 10: INTERACTIVE RAG CHAT
# ============================================================

print("\n" + "=" * 60)
print("PART 10: INTERACTIVE RAG CHAT")
print("=" * 60)

print("""
You can now chat with your RAG system.

Type 'exit' or 'quit' to stop.
""")


def interactive_rag():

    history = []

    while True:

        question = input(
            "\n👤 You: "
        ).strip()


        if question.lower() in [
            "exit",
            "quit",
            "q"
        ]:

            print(
                "\n👋 RAG chat ended!"
            )

            break


        if not question:

            print(
                "⚠️ Please enter a question."
            )

            continue


        answer, history = conversational_rag(

            question,

            chat_history=history,

            top_k=3

        )


        print(
            f"\n🤖 RAG: {answer}"
        )


# Uncomment this to start interactive chat:
# interactive_rag()


# ============================================================
# PART 11: RAG SYSTEM INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("PART 11: RAG SYSTEM INFORMATION")
print("=" * 60)

print(f"""
📚 Knowledge Base
   Documents: {len(documents)}

🧠 Embedding Model
   all-MiniLM-L6-v2
   Dimensions: {dimension}

🔎 Vector Database
   FAISS
   Index: IndexFlatIP

📐 Similarity
   Cosine Similarity

🤖 LLM
   Groq
   Model: {MODEL}

🔗 Pipeline

   User Question
        ↓
   Query Embedding
        ↓
   FAISS Search
        ↓
   Relevant Documents
        ↓
   Context
        ↓
   Groq LLM
        ↓
   Final Answer
""")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DAY 41 SUMMARY")
print("=" * 60)

print("""
✅ RAG concept understood

✅ Knowledge base created

✅ SentenceTransformer embeddings

✅ FAISS vector database

✅ Cosine similarity search

✅ Document retrieval

✅ Complete RAG pipeline

✅ Strong context-based prompting

✅ RAG vs No RAG comparison

✅ Chunking strategies

✅ Conversational RAG

✅ Interactive RAG chat

✅ Error handling

✅ Safer conversation history

============================================================

📚 DAY 41 — RAG COMPLETE! 🎉

🚀 NEXT:
AI Agents — AI that can THINK and ACT!
============================================================
""")