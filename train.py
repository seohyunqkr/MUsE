import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model Upload
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation='flash_attention_2',
).cuda()
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
model.train()

# PEFT
from peft import LoraConfig, get_peft_model
lora_config =  LoraConfig(
        r=64,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["qkv_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none"
)
model = get_peft_model(model, lora_config)

# Training Data
import pickle
with open(f'your_path/dataset_merge.pkl', 'rb') as f:
    train_data = pickle.load(f)


# batch size
batch_size = 3
batch_question_list = []
batch_question_answer_list = []
batch_label_list = []

# optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_data['questions'])//batch_size, eta_min=1e-6)

# training
progress_bar = tqdm(zip(train_data['questions'], train_data['answers'], train_data['users']), total=len(train_data['questions']))
for iter, (questions, answers, users) in enumerate(progress_bar):
    try:
      question = questions
      question_answer = question + f'{answers}<|end|>'

      if iter % batch_size != 0 or iter == 0:
          batch_question_list.append(question)
          batch_question_answer_list.append(question_answer)
      else:
          for q, q_a in zip(batch_question_list, batch_question_answer_list):
              length = tokenizer(q, return_tensors="pt").input_ids.shape[1]
              label = tokenizer(q_a, return_tensors="pt").input_ids[0]
              label[:length] = -100
              batch_label_list.append(label.flip(dims=(0,)))
          # label_token
          label_token = torch.nn.utils.rnn.pad_sequence(batch_label_list, batch_first=True, padding_value=-100).flip(dims=(1,))

          # model propagation
          phi3_input = tokenizer(batch_question_answer_list, return_tensors="pt", padding=True)
          output = model(input_ids=phi3_input.input_ids.cuda(), attention_mask=phi3_input.attention_mask.cuda(), labels=label_token)

          output.loss.backward()
          optimizer.step()
          scheduler.step()
          optimizer.zero_grad()
          progress_bar.set_description(f"Loss: {output.loss.item()} | Lr: {scheduler.get_last_lr()[0]:.6f}", refresh=True)

          # Refresh
          batch_question_list = []
          batch_question_answer_list = []
          batch_label_list = []
    except:
      pass

# Save
model.save_pretrained("your_path/trained_phi3")
tokenizer.save_pretrained("your_path/trained_phi3")
