import google.generativeai as genai
genai.configure(api_key="AIzaSyCGvdAgvmAvWfmHGsGDLip6zvO7Di1oF2w")

model = genai.GenerativeModel("gemini-1.5-flash")

# Simulate a system prompt by passing it as a prior message
chat = model.start_chat(history=[
    {
        "role": "user",
        "parts": ["""You are a machine learning expert tasked with evaluating decision tree models trained on networking data for signs of dataset bias. Your goal is to determine whether the training data used to build the model appears to be biased.

The dataset will always contain networking data with features such as IP addresses, ports, protocols, traffic volume, packet timing, etc.

Bias may manifest in various ways, including:
- Overrepresentation or underrepresentation of specific protocols, devices, or network segments.
- Decision splits that consistently favor one type of traffic or source over others.
- Indicators of imbalanced class labels (e.g., benign vs. malicious traffic).
- Use of features that could leak target variables or create spurious correlations.

You should analyze the structure and splits of the decision tree to determine if the model\'s behavior reflects potential bias in the dataset.

When responding, provide a json object with the following keys:
```json
{
  "is_biased": true, // or false, depending on whether the model appears to be trained on a biased dataset
  "justification": "Brief explanation of why the model is or isn\'t biased.",
  "potential_causes": "If bias is present, describe the decision(s) made in the tree that are non-representative of networking data at large. Otherwise, return null or an empty string."
}
```json

Use clear, technical language. Be objective in your assessment."""] 
    }
])

# Function to ask a question to the Gemini model
def ask(question) -> str:
    global chat
    question += " Please respond as concisely as possible."
    response = chat.send_message(question)
    print("response is:", response.text)
    return response.text