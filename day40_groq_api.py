# ============================================================
# DAY 40: LLM APIs - GROQ 🚀
# ============================================================

!pip install -q -U groq

from groq import Groq
import json
import time
from google.colab import userdata

print("LLM APIs with GROQ! 🚀")
print("Fast LLM inference! ⚡")
print("=" * 50)


# ============================================================
# SETUP
# ============================================================

try:
    api_key = userdata.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is empty")

    client = Groq(api_key=api_key)

    print("✅ Groq API connected!")

except Exception as e:
    print("❌ Groq setup failed.")
    print("Make sure GROQ_API_KEY is added to Colab Secrets.")
    print("Error:", e)


# ============================================================
# CURRENT GROQ MODELS
# ============================================================

MAIN_MODEL = "openai/gpt-oss-120b"
FAST_MODEL = "openai/gpt-oss-20b"

print("\nModels:")
print("🧠 Main:", MAIN_MODEL)
print("⚡ Fast:", FAST_MODEL)


# ============================================================
# PART 1: BASIC LLM MESSAGE
# ============================================================

print("\n" + "=" * 50)
print("PART 1: BASIC LLM MESSAGE")
print("=" * 50)


def ask_llm(
    prompt,
    max_tokens=500,
    model=MAIN_MODEL
):

    response = client.chat.completions.create(
        model=model,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        max_tokens=max_tokens,

        # Important for GPT-OSS
        include_reasoning=False
    )

    return response.choices[0].message.content


questions = [
    "What is machine learning in 2 sentences?",
    "What is the difference between AI and ML?",
    "Why is Python popular for data science?"
]


for q in questions:

    print(f"\nQuestion: {q}")

    response = ask_llm(
        q,
        max_tokens=100
    )

    print(f"LLM: {response}")

    print("-" * 40)


# ============================================================
# PART 2: SYSTEM PROMPTS
# ============================================================

print("\n" + "=" * 50)
print("PART 2: SYSTEM PROMPTS")
print("=" * 50)


def ask_with_system(
    system,
    user,
    max_tokens=300,
    model=MAIN_MODEL
):

    response = client.chat.completions.create(
        model=model,

        messages=[
            {
                "role": "system",
                "content": system
            },

            {
                "role": "user",
                "content": user
            }
        ],

        max_tokens=max_tokens,
        include_reasoning=False
    )

    return response.choices[0].message.content


personas = [

    {
        "system":
        "You are a cricket expert. "
        "Answer everything related to cricket.",

        "user":
        "What makes a good batsman?",

        "name":
        "Cricket Expert"
    },

    {
        "system":
        "You are a Python teacher for beginners. "
        "Use simple examples.",

        "user":
        "What is a list in Python?",

        "name":
        "Python Teacher"
    },

    {
        "system":
        "You are an ML engineer. "
        "Give technical detailed answers.",

        "user":
        "What is gradient descent?",

        "name":
        "ML Engineer"
    }
]


for persona in personas:

    print(
        f"\n🎭 Persona: {persona['name']}"
    )

    print(
        f"Question: {persona['user']}"
    )

    response = ask_with_system(
        persona["system"],
        persona["user"],
        max_tokens=150
    )

    print(
        f"Response: {response}"
    )

    print("-" * 40)


# ============================================================
# PART 3: PROMPT ENGINEERING
# ============================================================

print("\n" + "=" * 50)
print("PART 3: PROMPT ENGINEERING")
print("=" * 50)


bad_prompt = "Tell me about neural networks"


good_prompt = """
Explain neural networks using:

1. Simple definition (1 sentence)
2. Real life analogy
3. Key components (3 bullet points)
4. One practical example

Keep it beginner friendly.
"""


print("\nBad Prompt:")

bad_response = ask_llm(
    bad_prompt,
    max_tokens=80
)

print(
    f"Response: {bad_response}"
)


print("\nGood Prompt:")

good_response = ask_llm(
    good_prompt,
    max_tokens=250
)

print(
    f"Response: {good_response}"
)


# ============================================================
# PART 4: MULTI-TURN CHAT
# ============================================================

print("\n" + "=" * 50)
print("PART 4: MULTI-TURN CONVERSATION")
print("=" * 50)


def chat_with_llm(
    history,
    new_message,
    system=None,
    max_tokens=300,
    model=MAIN_MODEL
):

    messages = []

    if system:

        messages.append({
            "role": "system",
            "content": system
        })

    messages.extend(history)

    messages.append({
        "role": "user",
        "content": new_message
    })

    response = client.chat.completions.create(
        model=model,

        messages=messages,

        max_tokens=max_tokens,

        include_reasoning=False
    )

    assistant_msg = (
        response
        .choices[0]
        .message
        .content
    )

    history.append({
        "role": "user",
        "content": new_message
    })

    history.append({
        "role": "assistant",
        "content": assistant_msg
    })

    return assistant_msg, history


print("\nKunal's AI Tutor Conversation:")
print("-" * 40)


history = []


system = """
You are Kunal's AI tutor.

Kunal is a B.Sc CS student in Nagpur
learning ML/AI on mobile phone.

Give short, clear, encouraging answers.
"""


conversation = [

    "I just finished learning CNN today!",

    "What should I learn after CNN?",

    "How long to master deep learning?"
]


for message in conversation:

    print(
        f"\nKunal: {message}"
    )

    response, history = chat_with_llm(
        history,
        message,
        system,
        max_tokens=120
    )

    print(
        f"Tutor: {response}"
    )


print(
    f"\nTotal messages: {len(history)}"
)


# ============================================================
# PART 5: STRUCTURED JSON OUTPUT
# ============================================================

print("\n" + "=" * 50)
print("PART 5: STRUCTURED JSON OUTPUT")
print("=" * 50)


def get_json_output(
    prompt,
    model=MAIN_MODEL
):

    system = """
You are a data assistant.

Respond ONLY with valid JSON.

No explanation.
No markdown.
No code fences.
Just JSON.
"""

    response = ask_with_system(
        system,
        prompt,
        max_tokens=400,
        model=model
    )

    try:

        response = response.strip()

        if response.startswith("```"):

            response = response.replace(
                "```json",
                ""
            )

            response = response.replace(
                "```",
                ""
            )

            response = response.strip()

        return json.loads(response)

    except Exception:

        return {
            "raw": response
        }


# ------------------------------------------------------------
# STUDENT INFORMATION
# ------------------------------------------------------------

prompt1 = """
Extract this information as JSON:

"Kunal is 20 from Nagpur.
He studies B.Sc CS and loves cricket.
He knows Python and is learning ML."

Use this format:

{
  "name": "",
  "age": 0,
  "city": "",
  "degree": "",
  "hobbies": [],
  "skills": []
}
"""


print("\nExtracting student info:")


result1 = get_json_output(
    prompt1
)


print(
    json.dumps(
        result1,
        indent=2
    )
)


# ------------------------------------------------------------
# ML ALGORITHMS
# ------------------------------------------------------------

prompt2 = """
Create JSON for these 3 ML algorithms:

KNN
Random Forest
SVM

Use this format:

{
  "algorithms": [
    {
      "name": "",
      "type": "",
      "use_case": "",
      "accuracy": ""
    }
  ]
}

Do not claim an exact guaranteed accuracy.
Use "depends on dataset" if appropriate.
"""


print("\nML Algorithms JSON:")


result2 = get_json_output(
    prompt2
)


print(
    json.dumps(
        result2,
        indent=2
    )
)


# ============================================================
# PART 6: TEMPERATURE
# ============================================================

print("\n" + "=" * 50)
print("PART 6: TEMPERATURE PARAMETER")
print("=" * 50)


def ask_with_temp(
    prompt,
    temperature,
    max_tokens=80
):

    response = client.chat.completions.create(
        model=MAIN_MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        max_tokens=max_tokens,

        temperature=temperature,

        include_reasoning=False
    )

    return response.choices[0].message.content


prompt = (
    "Write a one-line tagline "
    "for an AI learning app"
)


print(
    f"\nPrompt: {prompt}\n"
)


for temp in [0.0, 0.5, 1.0]:

    response = ask_with_temp(
        prompt,
        temp,
        max_tokens=50
    )

    print(
        f"Temp={temp}: {response}"
    )


print("""
Temperature guide:

0.0 → More deterministic
0.5 → Balanced
1.0 → More creative and varied
""")


# ============================================================
# PART 7: PRACTICAL APPS
# ============================================================

print("\n" + "=" * 50)
print("PART 7: PRACTICAL APPS")
print("=" * 50)


# ------------------------------------------------------------
# APP 1: CODE REVIEWER
# ------------------------------------------------------------

print("\n--- App 1: Code Reviewer ---")


code = """
def find_average(nums):
    total = 0

    for n in nums:
        total = total + n

    return total / len(nums)

print(find_average([1,2,3,4,5]))
"""


review_prompt = f"""
Review this Python code:

```python
{code}