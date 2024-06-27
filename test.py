# inference code
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

# Model and Tokenizer upload
model = AutoPeftModelForCausalLM.from_pretrained(
    "your_path/trained_phi3",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation='flash_attention_2',
).cuda()
tokenizer = AutoTokenizer.from_pretrained("your_path/trained_phi3_2")

# system prompt function
system_prompt = lambda x: f'You should adaptively use proper words and sentences to make {x} understand. Please include contexts familiar with {x}'

# user question
questions = "Why is a balanced diet important?"
users='Elderly'

# Text Generation
_inputs = tokenizer(f'<|user|>{system_prompt(users)}\n{questions}<|end|>\n<|assistant|>\n', return_tensors="pt", padding=True)
output = model.generate(input_ids=_inputs.input_ids.cuda(),
                        attention_mask=_inputs.attention_mask.cuda(),
                        max_new_tokens=1024,
                        use_cache=True)
responses = tokenizer.batch_decode(output)
print(responses[0].split('<|assistant|>')[-1].split('<|end|>')[0])
