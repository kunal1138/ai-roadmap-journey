# ============================================================
# DAY 42: AI AGENTS 🤖
# AI that ACTS, not just answers!
# ============================================================

# Install Groq
!pip install -q groq

# ============================================================
# IMPORTS
# ============================================================

import json
import time
import math
import random
from datetime import datetime

from groq import Groq
from google.colab import userdata


print("AI Agents Day! 🤖")
print("AI that ACTS not just answers!")
print("=" * 60)


# ============================================================
# SETUP GROQ
# ============================================================

try:
    API_KEY = userdata.get("GROQ_API_KEY")

    if not API_KEY:
        raise ValueError("GROQ_API_KEY not found")

    client = Groq(api_key=API_KEY)

    MODEL = "openai/gpt-oss-120b"

    print("✅ Groq API connected!")
    print(f"🤖 Model: {MODEL}")

except Exception as e:
    print("❌ Could not connect to Groq.")
    print("Make sure you created a Colab Secret named:")
    print("GROQ_API_KEY")
    print()
    print("Error:", e)


# ============================================================
# PART 1: AGENT TOOLS
# ============================================================

print("\n" + "=" * 60)
print("PART 1: AGENT TOOLS")
print("=" * 60)


# ------------------------------------------------------------
# TOOL 1: CALCULATE
# ------------------------------------------------------------

def calculate(expression):
    """
    Calculate a basic mathematical expression.
    """

    try:
        # Safe mathematical environment
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "pow": pow,
            "sqrt": math.sqrt,
            "pi": math.pi,
            "e": math.e
        }

        result = eval(
            expression,
            {
                "__builtins__": {}
            },
            allowed_names
        )

        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"


# ------------------------------------------------------------
# TOOL 2: GET TIME
# ------------------------------------------------------------

def get_time():
    """
    Get current date and time.
    """

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ------------------------------------------------------------
# TOOL 3: SEARCH KNOWLEDGE BASE
# ------------------------------------------------------------

knowledge_base = [
    {
        "topic": "machine learning",
        "text": "Machine Learning is a subset of Artificial Intelligence that allows computers to learn patterns from data."
    },
    {
        "topic": "deep learning",
        "text": "Deep Learning uses neural networks with multiple layers to learn complex patterns from large amounts of data."
    },
    {
        "topic": "python",
        "text": "Python is a popular programming language used in AI, Machine Learning, data science, automation and web development."
    },
    {
        "topic": "artificial intelligence",
        "text": "Artificial Intelligence is the field of creating systems that can perform tasks that normally require human intelligence."
    },
    {
        "topic": "rag",
        "text": "RAG stands for Retrieval Augmented Generation. It combines information retrieval with language generation."
    },
    {
        "topic": "ai agents",
        "text": "AI Agents are AI systems that can reason about a goal, use tools, take actions and observe the results."
    },
    {
        "topic": "react",
        "text": "ReAct is an agent pattern where the model reasons about a task, takes an action using a tool, observes the result and continues."
    }
]


def search_knowledge_base(query):
    """
    Search the local knowledge base.
    """

    query_lower = query.lower()

    results = []

    for item in knowledge_base:

        if (
            query_lower in item["topic"].lower()
            or query_lower in item["text"].lower()
        ):
            results.append(item["text"])

    if results:
        return "\n".join(results)

    # Basic keyword matching
    words = query_lower.split()

    for item in knowledge_base:

        text = (
            item["topic"] + " " + item["text"]
        ).lower()

        if any(word in text for word in words):
            results.append(item["text"])

    if results:
        return "\n".join(results[:3])

    return "No relevant information found in the knowledge base."


# ------------------------------------------------------------
# TOOL 4: QUIZ
# ------------------------------------------------------------

def generate_quiz(topic):
    """
    Generate a simple quiz question.
    """

    quizzes = {
        "python": {
            "question": "Which keyword is used to define a function in Python?",
            "options": [
                "A) function",
                "B) def",
                "C) func",
                "D) define"
            ],
            "answer": "B) def"
        },

        "machine learning": {
            "question": "What does Machine Learning allow computers to do?",
            "options": [
                "A) Only perform calculations",
                "B) Learn patterns from data",
                "C) Only browse websites",
                "D) Only store files"
            ],
            "answer": "B) Learn patterns from data"
        },

        "ai agents": {
            "question": "What makes an AI Agent different from a normal chatbot?",
            "options": [
                "A) It can use tools and take actions",
                "B) It cannot answer questions",
                "C) It only works offline",
                "D) It cannot reason"
            ],
            "answer": "A) It can use tools and take actions"
        },

        "artificial intelligence": {
            "question": "What is AI?",
            "options": [
                "A) A programming language",
                "B) A database",
                "C) Systems that perform tasks requiring intelligence",
                "D) A computer game"
            ],
            "answer": "C) Systems that perform tasks requiring intelligence"
        }
    }

    topic_lower = topic.lower()

    selected = None

    for key in quizzes:
        if key in topic_lower or topic_lower in key:
            selected = quizzes[key]
            break

    if selected is None:
        selected = {
            "question": f"What is an important concept related to {topic}?",
            "options": [
                "A) Learning",
                "B) Data",
                "C) Algorithms",
                "D) All of the above"
            ],
            "answer": "D) All of the above"
        }

    return json.dumps(selected, indent=2)


# ------------------------------------------------------------
# TOOL 5: SUMMARIZE
# ------------------------------------------------------------

def summarize_text(text):
    """
    Simple text summarizer.
    """

    if not text:
        return "Nothing to summarize."

    # This tool intentionally performs a simple local summary.
    sentences = text.replace("!", ".").replace("?", ".").split(".")

    sentences = [
        s.strip()
        for s in sentences
        if s.strip()
    ]

    if len(sentences) <= 2:
        return text.strip()

    return ". ".join(sentences[:2]) + "."


# ------------------------------------------------------------
# TOOL 6: PROGRESS
# ------------------------------------------------------------

def check_progress(completed_days):
    """
    Check learning progress.
    """

    try:
        completed_days = int(completed_days)

        total_days = 100

        percentage = (
            completed_days / total_days
        ) * 100

        remaining = max(
            total_days - completed_days,
            0
        )

        return json.dumps({
            "completed_days": completed_days,
            "total_days": total_days,
            "progress_percentage": round(percentage, 2),
            "remaining_days": remaining
        }, indent=2)

    except Exception as e:
        return f"Progress error: {str(e)}"


# ============================================================
# TOOL DEFINITIONS FOR GROQ
# ============================================================

tools = [

    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate basic mathematical expressions. Use this for arithmetic calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression such as 15*30 or 42/450*100"
                    }
                },
                "required": ["expression"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current date and time. No input is required.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the local knowledge base for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The information to search for"
                    }
                },
                "required": ["query"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "quiz",
            "description": "Generate a quiz question about a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic for the quiz"
                    }
                },
                "required": ["topic"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "Summarize a piece of text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to summarize"
                    }
                },
                "required": ["text"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "progress",
            "description": "Check learning progress using the number of completed days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "completed_days": {
                        "type": "integer",
                        "description": "Number of completed learning days"
                    }
                },
                "required": ["completed_days"]
            }
        }
    }
]


print("\nAvailable tools:")

print("🔧 calculate: Calculate basic math expressions.")
print("🔧 get_time: Get current date and time.")
print("🔧 search: Search the knowledge base.")
print("🔧 quiz: Generate quiz questions.")
print("🔧 summarize: Summarize text.")
print("🔧 progress: Check learning progress.")


# ============================================================
# TOOL EXECUTOR
# ============================================================

def execute_tool(tool_name, arguments):

    print(f"🔧 Executing tool: {tool_name}")

    try:

        if tool_name == "calculate":

            return calculate(
                arguments.get("expression", "")
            )

        elif tool_name == "get_time":

            return get_time()

        elif tool_name == "search":

            return search_knowledge_base(
                arguments.get("query", "")
            )

        elif tool_name == "quiz":

            return generate_quiz(
                arguments.get("topic", "")
            )

        elif tool_name == "summarize":

            return summarize_text(
                arguments.get("text", "")
            )

        elif tool_name == "progress":

            return check_progress(
                arguments.get("completed_days", 0)
            )

        else:

            return f"Unknown tool: {tool_name}"

    except Exception as e:

        return f"Tool execution error: {str(e)}"


# ============================================================
# GROQ CALL
# ============================================================

def groq_call(
    messages,
    max_tokens=500,
    temperature=0.2
):

    response = client.chat.completions.create(

        model=MODEL,

        messages=messages,

        tools=tools,

        # IMPORTANT:
        # The old code used tool_choice="none".
        # That caused the error because the model tried
        # to call calculate while tools were disabled.
        tool_choice="auto",

        max_tokens=max_tokens,

        temperature=temperature
    )

    return response


# ============================================================
# PART 2: SIMPLE ReAct AGENT
# ============================================================

print("\n" + "=" * 60)
print("PART 2: SIMPLE ReAct AGENT")
print("=" * 60)


def simple_agent(user_goal, max_steps=6):

    """
    Simple ReAct-style agent.

    Loop:

    THINK
       ↓
    ACTION / TOOL
       ↓
    OBSERVATION
       ↓
    THINK
       ↓
    FINAL ANSWER
    """

    messages = [

        {
            "role": "system",
            "content": """
You are a helpful AI Agent.

Your job is to solve the user's goal.

You have access to tools.

Follow this process:

1. Understand the goal.
2. Decide whether a tool is needed.
3. Use the appropriate tool.
4. Look at the tool result.
5. Continue if another tool is needed.
6. Give a clear final answer.

IMPORTANT:
- Use calculate for mathematical calculations.
- Do not guess calculations when the calculate tool can do them.
- Use other tools when useful.
- Once you have enough information, provide the final answer.
- Keep the final answer simple and easy to understand.
"""
        },

        {
            "role": "user",
            "content": user_goal
        }
    ]


    for step in range(1, max_steps + 1):

        print(f"\n📍 Step {step}:")
        print("-" * 45)

        response = groq_call(
            messages,
            max_tokens=700,
            temperature=0.2
        )

        assistant_message = response.choices[0].message

        # ----------------------------------------------------
        # CHECK FOR TOOL CALL
        # ----------------------------------------------------

        if assistant_message.tool_calls:

            # Add assistant's tool-call message
            messages.append(
                assistant_message
            )

            for tool_call in assistant_message.tool_calls:

                tool_name = tool_call.function.name

                raw_arguments = (
                    tool_call.function.arguments
                )

                try:
                    arguments = json.loads(
                        raw_arguments
                    )
                except:
                    arguments = {}

                print(
                    f"🤔 Agent wants to use: {tool_name}"
                )

                print(
                    f"📥 Arguments: {arguments}"
                )

                # Execute tool
                tool_result = execute_tool(
                    tool_name,
                    arguments
                )

                print(
                    f"📤 Result: {tool_result}"
                )

                # Send observation back to model
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_result)
                    }
                )

            continue


        # ----------------------------------------------------
        # NO TOOL CALL = FINAL ANSWER
        # ----------------------------------------------------

        final_answer = assistant_message.content

        print("\n✅ Agent finished!")
        print("-" * 45)

        return final_answer


    return "Agent reached the maximum number of steps."


# ============================================================
# PART 3: TEST AGENT WITH GOALS
# ============================================================

print("\n" + "=" * 60)
print("PART 3: TEST AGENT WITH GOALS")
print("=" * 60)


goals = [

    "What is 15 months × 30 days? And what percentage is 42 days of that total?",

    "What is Machine Learning? Search the knowledge base and explain it simply.",

    "Generate a simple quiz about Python.",

    "What is my learning progress if I have completed 42 days?",

    "What is the current date and time?"
]


for goal in goals:

    print("\n")
    print("🎯 Goal:", goal)
    print("=" * 60)

    result = simple_agent(goal)

    print("\n🤖 FINAL ANSWER:")
    print(result)

    print("\n" + "=" * 60)

    time.sleep(1)


# ============================================================
# DAY 42 COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("🎉 DAY 42 COMPLETE!")
print("=" * 60)

print("""
You built a basic AI Agent!

The agent can:

✅ Understand goals
✅ Decide when to use tools
✅ Call tools
✅ Receive tool results
✅ Continue reasoning
✅ Produce a final answer

This is the basic ReAct pattern:

THINK → ACT → OBSERVE → THINK → ACT → FINAL ANSWER

🤖 AI that ACTS, not just answers!
""")