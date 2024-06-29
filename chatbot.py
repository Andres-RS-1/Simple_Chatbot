# Import the necessary classes from the transformers library
# AutoTokenizer is used for tokenizing text
# AutoModelForSeq2SeqLM is a class for sequence-to-sequence models
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Define the name of the pre-trained model to be used
# In this case, we are using the "facebook/blenderbot-400M-distill" model
model_name = "facebook/blenderbot-400M-distill"

# Load the pre-trained sequence-to-sequence model from the model name specified
# This model will be responsible for generating responses in the conversation
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load the tokenizer corresponding to the pre-trained model
# The tokenizer is responsible for converting text into tokens that the model can understand
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize an empty list to store the conversation history
# This will keep track of both the user's inputs and the model's responses
conversation_history = []

# Start an infinite loop to keep the conversation going until manually interrupted
while True:
    # Convert the conversation history list into a single string separated by newline characters
    # This combined string will be used as part of the context for generating the model's response
    history_string = "\n".join(conversation_history)

    # Prompt the user for input using the input() function
    # The user's input will be the next part of the conversation
    input_text = input("> ")

    # Encode the conversation history and the user's input into tensors
    # The encode_plus() method prepares the input for the model, creating tensors from the text
    # return_tensors="pt" specifies that the output should be PyTorch tensors
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    # Generate the model's response based on the encoded inputs
    # The generate() method produces a sequence of tokens as the model's response
    outputs = model.generate(**inputs)

    # Decode the model's response from tokens back into a readable string
    # The decode() method converts the token IDs back into text
    # skip_special_tokens=True removes any special tokens used during tokenization
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Print the model's response to the console so the user can see it
    print(response)

    # Append the user's input to the conversation history list
    # This ensures that the input is included in the context for future responses
    conversation_history.append(input_text)
    
    # Append the model's response to the conversation history list
    # This ensures that the response is included in the context for future responses
    conversation_history.append(response)
