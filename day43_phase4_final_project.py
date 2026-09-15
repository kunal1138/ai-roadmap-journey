# ============================================================
# DAY 43: PHASE 4 FINAL PROJECT 🚀
# COMPLETE AI ASSISTANT SYSTEM 🤖
# ============================================================

# Google Colab installation:
# !pip install -q groq sentence-transformers faiss-cpu

import json
import time
import numpy as np
import faiss
from datetime import datetime
from groq import Groq
from sentence_transformers import SentenceTransformer
from google.colab import userdata


print("=" * 60)
print("  COMPLETE AI ASSISTANT SYSTEM 🤖")
print("=" * 60)


# ============================================================
# SETUP
# ============================================================

API_KEY = userdata.get("GROQ_API_KEY")

client = Groq(api_key=API_KEY)

MODEL = "openai/gpt-oss-120b"

print(f"✅ Connected! Model: {MODEL}")


# ============================================================
# SECTION 1: KNOWLEDGE BASE (RAG)
# ============================================================

print("\n" + "=" * 60)
print("SECTION 1: KNOWLEDGE BASE")
print("=" * 60)


knowledge_docs = [

    # --------------------------------------------------------
    # PHASE 1
    # --------------------------------------------------------

    """Phase 1 Python Foundations Days 1-19.
    Kunal learned Python basics, variables,
    functions, operators, modules, loops,
    conditionals, data structures, strings,
    comprehensions, file handling, exception
    handling, OOP classes, inheritance,
    polymorphism, encapsulation, NumPy,
    Pandas, SQL, Git and Math for AI.""",


    # --------------------------------------------------------
    # PHASE 2
    # --------------------------------------------------------

    """Phase 2 Machine Learning Days 20-31.
    Kunal learned KNN K-Nearest Neighbors,
    Linear Regression, Logistic Regression,
    Decision Trees, Random Forest, SVM,
    Support Vector Machine, K-Means Clustering,
    Cross Validation and Model Evaluation.
    Projects included House Price Prediction,
    Spam Classifier and Student Performance
    Analyzer.""",


    # --------------------------------------------------------
    # PHASE 3
    # --------------------------------------------------------

    """Phase 3 Deep Learning Days 32-35.
    Kunal learned Neural Networks, TensorFlow,
    Keras, Dropout, BatchNormalization,
    Early Stopping and CNN Convolutional
    Neural Networks.
    Kunal built an MNIST digit recognition
    model with 99.32 percent accuracy.""",


    # --------------------------------------------------------
    # PHASE 4
    # --------------------------------------------------------

    """Phase 4 Generative AI Days 36-43.
    Kunal learned NLP Natural Language
    Processing, tokenization, stopwords,
    stemming, lemmatization, TF-IDF,
    sentiment analysis, Word2Vec embeddings,
    BERT, Transformers, Hugging Face,
    fine-tuning, LLM APIs, Groq, RAG,
    Retrieval Augmented Generation,
    FAISS vector database, AI Agents,
    ReAct pattern and tool calling.
    Day 43 is the Phase 4 final AI Assistant
    project combining these concepts.""",


    # --------------------------------------------------------
    # CERTIFICATIONS
    # --------------------------------------------------------

    """Kunal Choudhari completed Anthropic
    certifications including AI Fluency for
    Students and Claude with Amazon Bedrock
    in August 2026.
    GitHub repository:
    github.com/kunal1138/ai-roadmap-journey.""",


    # --------------------------------------------------------
    # PROJECTS
    # --------------------------------------------------------

    """Kunal built multiple projects during
    his AI learning journey.

    To-Do List App Python Day 7.
    Family Conversation OOP Day 12.
    Student Result Analysis Pandas Day 15.
    Gaming Analysis System Day 19.
    House Price Prediction Linear Regression
    Day 29.
    Spam Classifier NLP TF-IDF Day 30.
    Student Performance Analyzer ML Day 31.
    CNN MNIST 99.32 percent accuracy Day 34.
    RAG System FAISS Groq Day 41.
    AI Agents ReAct Day 42.
    Complete AI Assistant System Day 43.""",


    # --------------------------------------------------------
    # GENERAL PROFILE
    # --------------------------------------------------------

    # 🔧 CHANGED:
    # Removed unnecessary personal information
    # such as age and friends' names.

    """Kunal Choudhari is a final-year B.Sc.
    Computer Science student from Nagpur,
    Maharashtra, India.

    He is following a structured AI/ML
    learning roadmap with the goal of
    becoming a Generative AI / LLM Engineer.""",


    # --------------------------------------------------------
    # TECHNICAL SKILLS
    # --------------------------------------------------------

    """Technical skills include Python,
    NumPy, Pandas, Matplotlib, SQL, Git,
    Scikit-learn, TensorFlow, Keras, NLTK,
    Gensim, Hugging Face Transformers, BERT,
    Groq API, FAISS, SentenceTransformers,
    GitHub and Google Colab."""
]


print(f"Knowledge docs: {len(knowledge_docs)}")


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print("\nLoading embedding model...")


embedder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


print("✅ Embedding model loaded!")


print("Creating embeddings...")


doc_embeddings = embedder.encode(
    knowledge_docs,
    normalize_embeddings=True,
    show_progress_bar=False
)


# ============================================================
# FAISS VECTOR DATABASE
# ============================================================

dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(
    doc_embeddings.astype(np.float32)
)


print(
    f"✅ FAISS index: {index.ntotal} vectors"
)


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve_docs(query, top_k=3):

    q_emb = embedder.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        q_emb.astype(np.float32),
        top_k
    )

    results = []

    for j, i in enumerate(indices[0]):

        results.append({

            "doc": knowledge_docs[i],

            "score": float(scores[0][j])

        })

    return results


# ============================================================
# SECTION 2: AGENT TOOLS
# ============================================================

print("\n" + "=" * 60)
print("SECTION 2: AGENT TOOLS")
print("=" * 60)


# ============================================================
# TOOL 1: RAG SEARCH
# ============================================================

def rag_search(query, threshold=0.30):

    """
    Search knowledge base using RAG.
    """

    # 🔧 CHANGED:
    # Retrieve 3 documents instead of 2.

    docs = retrieve_docs(
        query,
        top_k=3
    )


    # 🔧 CHANGED:
    # Added relevance filtering so unrelated
    # documents are not blindly returned.

    relevant_docs = [

        d for d in docs

        if d["score"] >= threshold

    ]


    if not relevant_docs:

        return (
            "No sufficiently relevant information "
            "was found in the knowledge base."
        )


    # 🔧 CHANGED:
    # Return complete relevant documents
    # instead of cutting the context at 500 characters.

    context_parts = []


    for d in relevant_docs:

        context_parts.append(

            f"[Relevance: {d['score']:.3f}]\n"
            f"{d['doc']}"

        )


    return "\n\n".join(context_parts)


# ============================================================
# TOOL 2: CALCULATOR
# ============================================================

def calculate(expression):

    """
    Safe basic calculator.
    """

    try:

        import math

        allowed = {

            "abs": abs,

            "round": round,

            "sqrt": math.sqrt,

            "pi": math.pi,

            "e": math.e,

            "min": min,

            "max": max

        }


        result = eval(

            expression,

            {
                "__builtins__": {}
            },

            allowed

        )


        return str(result)


    except Exception as e:

        return f"Calculation error: {e}"


# ============================================================
# TOOL 3: GET TIME
# ============================================================

def get_time():

    """
    Get current date and time.
    """

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ============================================================
# TOOL 4: CHECK PROGRESS
# ============================================================

def check_progress(days):

    """
    Check AI roadmap progress.
    """

    try:

        days = int(days)

        total = 450

        percentage = round(
            days / total * 100,
            2
        )


        # 🔧 CHANGED:
        # Fixed phase boundaries.
        # Day 43 now correctly belongs to Phase 4.

        if days <= 19:

            phase = "Phase 1: Python Foundations"

        elif days <= 31:

            phase = "Phase 2: Machine Learning"

        elif days <= 35:

            phase = "Phase 3: Deep Learning"

        elif days <= 43:

            phase = "Phase 4: Generative AI"

        else:

            phase = "Phase 5: Deployment"


        return json.dumps({

            "days_done": days,

            "total_days": total,

            "percentage": percentage,

            "current_phase": phase,

            "days_remaining": max(
                total - days,
                0
            )

        }, indent=2)


    except Exception as e:

        return f"Progress error: {e}"


# ============================================================
# TOOL 5: STUDY PLAN
# ============================================================

def generate_study_plan(topic):

    """
    Generate a simple 3-day study plan.
    """

    response = client.chat.completions.create(

        model=MODEL,

        messages=[

            {

                "role": "system",

                "content":
                "Create a simple beginner-friendly "
                "3-day study plan."

            },

            {

                "role": "user",

                "content":
                f"""Create a simple 3-day
                study plan for {topic}.

                Format:

                Day 1: ...
                Day 2: ...
                Day 3: ...

                Keep it practical."""

            }

        ],

        max_tokens=200

    )


    return response.choices[0].message.content


# ============================================================
# TOOL 6: SENTIMENT
# ============================================================

def analyze_sentiment(text):

    """
    Lightweight rule-based sentiment analysis.
    """

    positive = [

        "love",
        "great",
        "amazing",
        "excellent",
        "good",
        "best",
        "brilliant",
        "fantastic",
        "completed",
        "achieved"

    ]


    negative = [

        "hate",
        "bad",
        "terrible",
        "worst",
        "boring",
        "failed",
        "disappointed",
        "awful"

    ]


    text_lower = text.lower()


    pos_count = sum(

        1

        for word in positive

        if word in text_lower

    )


    neg_count = sum(

        1

        for word in negative

        if word in text_lower

    )


    if pos_count > neg_count:

        sentiment = "POSITIVE 😊"

        score = pos_count / (
            pos_count +
            neg_count +
            1
        )


    elif neg_count > pos_count:

        sentiment = "NEGATIVE 😞"

        score = neg_count / (
            pos_count +
            neg_count +
            1
        )


    else:

        sentiment = "NEUTRAL 😐"

        score = 0.5


    return json.dumps({

        "sentiment": sentiment,

        "score": round(score, 2),

        "positive_words": pos_count,

        "negative_words": neg_count

    }, indent=2)


# ============================================================
# TOOL DEFINITIONS FOR GROQ
# ============================================================

TOOL_DEFS = [

    {

        "type": "function",

        "function": {

            "name": "rag_search",

            "description":
            "Search Kunal's AI learning journey "
            "knowledge base.",

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {

                        "type": "string",

                        "description":
                        "Search query."

                    }

                },

                "required": [
                    "query"
                ]

            }

        }

    },


    {

        "type": "function",

        "function": {

            "name": "calculate",

            "description":
            "Calculate a mathematical expression.",

            "parameters": {

                "type": "object",

                "properties": {

                    "expression": {

                        "type": "string",

                        "description":
                        "Math expression."

                    }

                },

                "required": [
                    "expression"
                ]

            }

        }

    },


    {

        "type": "function",

        "function": {

            "name": "get_time",

            "description":
            "Get current date and time.",

            "parameters": {

                "type": "object",

                "properties": {}

            }

        }

    },


    {

        "type": "function",

        "function": {

            "name": "check_progress",

            "description":
            "Check AI learning roadmap progress.",

            "parameters": {

                "type": "object",

                "properties": {

                    "days": {

                        "type": "integer",

                        "description":
                        "Number of completed days."

                    }

                },

                "required": [
                    "days"
                ]

            }

        }

    },


    {

        "type": "function",

        "function": {

            "name": "study_plan",

            "description":
            "Generate a 3-day study plan.",

            "parameters": {

                "type": "object",

                "properties": {

                    "topic": {

                        "type": "string",

                        "description":
                        "Topic to study."

                    }

                },

                "required": [
                    "topic"
                ]

            }

        }

    },


    {

        "type": "function",

        "function": {

            "name": "sentiment",

            "description":
            "Analyze text sentiment.",

            "parameters": {

                "type": "object",

                "properties": {

                    "text": {

                        "type": "string",

                        "description":
                        "Text to analyze."

                    }

                },

                "required": [
                    "text"
                ]

            }

        }

    }

]


print("Tools registered:")


for tool in TOOL_DEFS:

    print(
        f"  🔧 {tool['function']['name']}"
    )


# ============================================================
# TOOL EXECUTOR
# ============================================================

def execute_tool(name, args):

    """
    Execute the selected tool.
    """

    try:

        if name == "rag_search":

            return rag_search(
                args.get("query", "")
            )


        elif name == "calculate":

            return calculate(
                args.get("expression", "")
            )


        elif name == "get_time":

            return get_time()


        elif name == "check_progress":

            return check_progress(
                args.get("days", 0)
            )


        elif name == "study_plan":

            return generate_study_plan(
                args.get("topic", "")
            )


        elif name == "sentiment":

            return analyze_sentiment(
                args.get("text", "")
            )


        else:

            return f"Unknown tool: {name}"


    except Exception as e:

        return f"Tool execution error: {e}"


# ============================================================
# SECTION 3: AI ASSISTANT AGENT
# ============================================================

print("\n" + "=" * 60)
print("SECTION 3: AI ASSISTANT AGENT")
print("=" * 60)


# 🔧 CHANGED:
# Changed conversation_history=[] to None.
# This prevents multiple conversations from
# accidentally sharing the same list.

def ai_assistant(
    question,
    conversation_history=None,
    max_steps=5
):


    # 🔧 CHANGED:
    # Create a new list when history is not provided.

    if conversation_history is None:

        conversation_history = []


    messages = [

        {

            "role": "system",

            "content":
            """You are Kunal's personal
            AI assistant.

            You know about his AI/ML
            learning journey.

            Use tools when needed.

            rag_search:
            Use for questions about
            Kunal's learning, projects
            and skills.

            calculate:
            Use for mathematical
            calculations.

            check_progress:
            Use for roadmap progress.

            study_plan:
            Use for study planning.

            sentiment:
            Use for text sentiment.

            get_time:
            Use for current date/time.

            Do not invent facts about
            Kunal when the knowledge
            base does not contain them.

            Be helpful, encouraging
            and concise."""
        }

    ]


    # Add recent conversation memory

    messages.extend(
        conversation_history[-6:]
    )


    # Add current question

    messages.append({

        "role": "user",

        "content": question

    })


    # ========================================================
    # AGENT LOOP
    # ========================================================

    for step in range(max_steps):

        # 🔧 CHANGED:
        # Correct indentation.
        # This line belongs inside the for loop.

        print(
            f"  🧠 Agent step {step + 1}"
        )


        response = client.chat.completions.create(

            model=MODEL,

            messages=messages,

            tools=TOOL_DEFS,

            # 🔧 CHANGED:
            # "auto" allows the model to decide
            # whether a tool is needed.
            #
            # "none" previously caused:
            # Tool choice is none, but model called a tool.

            tool_choice="auto",

            max_tokens=500

        )


        msg = response.choices[0].message


        # ====================================================
        # TOOL CALL
        # ====================================================

        if msg.tool_calls:

            messages.append(msg)


            for tc in msg.tool_calls:

                name = tc.function.name


                try:

                    args = json.loads(
                        tc.function.arguments
                    )


                except Exception:

                    args = {}


                print(
                    f"  🔧 Using tool: {name}"
                )


                print(
                    f"  📥 Arguments: {args}"
                )


                result = execute_tool(
                    name,
                    args
                )


                print(
                    f"  📤 Result: "
                    f"{str(result)[:150]}..."
                )


                messages.append({

                    "role": "tool",

                    "tool_call_id": tc.id,

                    "content": str(result)

                })


        # ====================================================
        # FINAL ANSWER
        # ====================================================

        else:

            answer = msg.content


            # 🔧 CHANGED:
            # Added fallback if the model returns
            # an empty response.

            if not answer:

                answer = (
                    "I could not generate "
                    "a final response."
                )


            # Update conversation memory

            conversation_history.append({

                "role": "user",

                "content": question

            })


            conversation_history.append({

                "role": "assistant",

                "content": answer

            })


            return answer, conversation_history


    # Maximum steps reached

    return (

        "The agent reached the maximum "
        "number of reasoning steps.",

        conversation_history

    )


# ============================================================
# SECTION 4: TEST COMPLETE SYSTEM
# ============================================================

print("\n" + "=" * 60)
print("SECTION 4: TESTING AI ASSISTANT")
print("=" * 60)


test_questions = [

    "What phases has Kunal completed?",

    "What is Kunal's CNN accuracy on MNIST?",

    "Check Kunal's progress at day 43",

    "What projects did Kunal build?",

    "Calculate: how many hours if Kunal studies "
    "1.5 hours daily for 43 days?",

    "What are Kunal's certifications?",

    "Create a 3-day study plan for LangChain",

    "Analyze this text: I love learning AI "
    "and completed 43 days of coding!"

]


history = []


print("\n" + "=" * 60)


for q in test_questions:

    print(
        f"\n❓ {q}"
    )


    answer, history = ai_assistant(

        q,

        history

    )


    print(
        f"🤖 {answer}"
    )


    print("-" * 50)


    time.sleep(0.5)


# ============================================================
# SECTION 5: MULTI-TURN CONVERSATION
# ============================================================

print("\n" + "=" * 60)
print("SECTION 5: MULTI-TURN CONVERSATION")
print("=" * 60)


print(
    "\nTesting conversation memory..."
)


conv_history = []


conversations = [

    "My name is Kunal and I'm from Nagpur",

    "What is my current phase?",

    "How many days have I completed?",

    "What should I learn next after agents?",

    "Give me encouragement to keep going!"

]


print("\n" + "=" * 60)


for user_message in conversations:

    print(
        f"\nKunal: {user_message}"
    )


    response, conv_history = ai_assistant(

        user_message,

        conv_history

    )


    print(
        f"Assistant: {response}"
    )


    print("-" * 45)


    time.sleep(0.5)


print(
    f"\nConversation length: "
    f"{len(conv_history)} messages"
)


# ============================================================
# SECTION 6: PERFORMANCE SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SECTION 6: SYSTEM PERFORMANCE")
print("=" * 60)


progress_data = json.loads(
    check_progress(43)
)


print(f"""

╔══════════════════════════════════════════╗
║    PHASE 4 FINAL PROJECT REPORT 🤖      ║
╠══════════════════════════════════════════╣
║                                          ║
║  AI ASSISTANT COMPONENTS:                ║
║  ─────────────────────────────           ║
║  ✅ RAG Knowledge Base (8 docs)          ║
║  ✅ FAISS Vector Database                ║
║  ✅ SentenceTransformer Embeddings       ║
║  ✅ 6 Agent Tools                        ║
║  ✅ ReAct Agent Pattern                  ║
║  ✅ Conversation Memory                  ║
║  ✅ Groq LLM Integration                 ║
║                                          ║
║  KUNAL'S JOURNEY:                        ║
║  ─────────────────────────────           ║
║  Days Completed: {progress_data['days_done']:3d} / {progress_data['total_days']}         ║
║  Progress: {progress_data['percentage']:5.1f}%                    ║
║  Current: {progress_data['current_phase']:32s}║
║  Remaining: {progress_data['days_remaining']:3d} days                   ║
║                                          ║
║  PHASE 4 TOPICS:                         ║
║  ─────────────────────────────           ║
║  ✅ NLP + Text Processing                ║
║  ✅ Word2Vec + Embeddings                ║
║  ✅ BERT + Transformers                  ║
║  ✅ Hugging Face Fine-tuning             ║
║  ✅ LLM APIs + Groq                      ║
║  ✅ RAG + Vector Database                ║
║  ✅ AI Agents + ReAct                    ║
║  ✅ Phase 4 Final Project                ║
║                                          ║
║  PHASE 4 COMPLETE! 🎉                    ║
║  Ready for Phase 5: Deployment! 🚀       ║
╚══════════════════════════════════════════╝
""")


print(
    "Technologies used in this project:"
)


print("  ✅ Python")

print("  ✅ Groq API (LLM)")

print(
    "  ✅ SentenceTransformer (Embeddings)"
)

print(
    "  ✅ FAISS (Vector Search)"
)

print(
    "  ✅ RAG Pipeline"
)

print(
    "  ✅ AI Agents + Tool Calling"
)

print(
    "  ✅ Conversation Memory"
)


print(
    "\n🚀 Phase 4 GenAI — COMPLETE!"
)

print(
    "🎯 Next: Phase 5 Deployment!"
)