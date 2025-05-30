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


def load_decision_tree(file_path):
    """Load the Graphviz DOT file containing the decision tree."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def analyze_tree_for_bias(tree_content):
    """Send the decision tree to Gemini for bias analysis."""
    if not tree_content:
        return "Error: No tree content to analyze."
    
    prompt = f"""Please analyze the following decision tree (in Graphviz DOT format) for potential bias in the training dataset:

{tree_content}

Analyze the tree structure, decision splits, feature usage, and any patterns that might indicate bias in the underlying networking dataset."""
    
    response = chat.send_message(prompt)
    return response.text


if __name__ == "__main__":
    tree_file_path = "trustee_tree_puffer_pruned.txt"
    
    tree_content = load_decision_tree(tree_file_path)
    
    if tree_content:
        print("Decision tree loaded successfully.")
        print(f"Tree content preview (first 200 chars): {tree_content[:200]}...")
        
        print("\nAnalyzing tree for bias...")
        bias_analysis = analyze_tree_for_bias(tree_content)
        print("\nBias Analysis Result:")
        print(bias_analysis)
    else:
        print("Failed to load decision tree file.")