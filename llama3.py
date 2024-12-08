import transformers
import torch

# Load the model pipeline
model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
# model_id = "meta-llama/Llama-3.3-70B-Instruct"


# For downloading models from Hugging Face Hub
# from huggingface_hub import login
# with open('token.txt', 'r') as f:
#     token = f.read()
# login(token = token)


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Initial system message
messages = [
    {"role": "system", "content": "You are a helpful assitant."},
]

# Chat loop
while True:
    # User input
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat.")
        break

    # Add user input to the conversation
    messages.append({"role": "user", "content": user_input})
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # Generate model response
    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        pad_token_id = pipeline.tokenizer.eos_token_id
    )
    response = outputs[0]["generated_text"][-1]['content']
    print(f"Bot: {response}")
    print('--------------------------------------')

    # Add response to messages and print it
    messages.append({"role": "assistant", "content": response})
