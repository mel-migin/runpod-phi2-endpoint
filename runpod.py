import torch
import runpod
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Model Initialization ---
# Load the model and tokenizer once when the worker starts.
model_name = "microsoft/phi-2"
torch.set_default_device("cuda") # Set the default device to GPU

# Load the model and tokenizer, trusting remote code for phi-2
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


# --- Handler Function ---
# This function will be called for every API request.
def handler(job):
    """
    Takes a job object as input, which contains the request payload.
    """
    job_input = job['input']
    prompt = job_input.get('prompt')

    if not prompt:
        return {"error": "No prompt provided in the input."}

    # Run inference
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs, max_length=200)
    result_text = tokenizer.batch_decode(outputs)[0]

    # Return the generated text
    return {
        "generated_text": result_text
    }


# --- Start the Serverless Worker ---
runpod.serverless.start({"handler": handler})