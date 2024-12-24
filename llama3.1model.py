from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
import torch
import pandas as pd

HUGGINGFACE_TOKEN = "" # Enter your HUGGING FACE Token here

config = LlamaConfig(
    rope_scaling={
        "type": "dynamic",
        "factor": 8.0
    }
)

print("Loading model and tokenizer...")

# Load pre-trained model and tokenizer with correct configuration

''' Here I have used Meta Llama 3.1 with 70 Billion Parameters. 
    One of the latest models. 
    You can use other models as well.'''

model_name = "meta-llama/Meta-Llama-3.1-70B"
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, use_auth_token=HUGGINGFACE_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGINGFACE_TOKEN)
print("Model and tokenizer loaded successfully.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

file_path = '' # Add your file path to save the results of your legacy data
df = pd.read_excel(file_path)

def generate_text(seed_text, max_length=100):
    input_ids = tokenizer.encode(seed_text, return_tensors='pt').to(device)
    print("Generating text...")
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.95,
        top_k=50,
        num_beams=5,
        temperature=0.85,
        do_sample=True
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

questions = [
    {
        # Sample Question
        # Add more if you need to integrate with your legacy data as well
        
        "question": "1. Where is the field?",
        "fields": [
            {"name": "District", "column": "District"},
            {"name": "Block / Municipality / NAC", "column": "Block"},
            {"name": "Gram Panchayat (GP) / Ward", "column": "Gram Panchayat"},
            {"name": "Village", "column": "Village"}
        ],
        "choices": [],
        "if_statements": []
    }
]

def check_conditions(answers, question):
    for condition in question.get("if_statements", []):
        if condition.startswith("if answer to question "):
            q_num = int(condition.split()[4])
            expected_answer = condition.split("'")[1]
            if len(answers) >= q_num and answers[q_num - 1].lower() != expected_answer.lower():
                return False
    return True

def ask_questions():
    answers = []
    response_data = {}
    for question in questions:
        if check_conditions(answers, question):
            print(question["question"])
            if question["choices"]:
                print("Choices:", ", ".join(question["choices"]))
            if question["fields"]:
                for field in question["fields"]:
                    answer = input(f"Enter {field['name']}: ")
                    answers.append(answer)
                    response_data[field['column']] = answer
            else:
                answer = input("Your answer: ")
                answers.append(answer)
    return response_data

def chatbot_response(user_input):
    if user_input.lower() == "rcm app":
        print("Starting RCM app question sequence...")
        response_data = ask_questions()
        df_new = pd.DataFrame([response_data])
        global df
        df = pd.concat([df, df_new], ignore_index=True)
        df.to_excel(file_path, index=False)
        print("Thank you for your responses!")
        return None
    else:
        print("Generating response...")
        response = generate_text(user_input, max_length=50)
        print("Response generated.")
        return response

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = chatbot_response(user_input)
    if response:
        print(f"Chatbot: {response}")
