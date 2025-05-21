import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler



# Initialize distributed training
def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size

# Custom dataset class for GSM8K
class GSM8KDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        text = example["text"]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()
        labels = input_ids.clone()
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Format GSM8K dataset
def format_gsm8k_example(example, tokenizer):
    question = example["question"]
    answer = example["answer"].split("####")[-1].strip()
    reasoning = example["answer"].split("####")[0].strip()
    messages = [
        {"role": "system", "content": "You are a helpful assistant skilled in solving math problems."},
        {"role": "user", "content": f"{question}\nPlease provide a step-by-step solution."},
        {"role": "assistant", "content": f"{reasoning}\n\nFinal Answer: {answer}"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

# Training step
def train_step(model, batch, optimizer, device):
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    with torch.cuda.amp.autocast():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
    
    return loss.item()

# Main training function
def main():
    # Setup distributed environment
    rank, world_size = setup_distributed()
    
    # Load tokenizer and model
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": rank}
    )

    model = FSDP(
        model
    )
    
    # Load and prepare dataset
    dataset = load_dataset("openai/gsm8k", "main")["train"]
    dataset = dataset.map(lambda x: format_gsm8k_example(x, tokenizer))
    train_dataset = GSM8KDataset(dataset.shuffle(seed=42).select(range(1000)), tokenizer)
    
    # Distributed data loader
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Training loop
    num_epochs = 3
    total_steps = len(dataloader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    device = torch.device(f"cuda:{rank}")
    
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            # Learning rate warmup
            lr = 5e-5 * min(1.0, (step + 1 + epoch * len(dataloader)) / warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            
            loss = train_step(model, batch, optimizer, device)
            total_loss += loss
            
            if rank == 0 and (step + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / len(dataloader)
        if rank == 0:
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    # Save model (only rank 0)
    if rank == 0:
        output_dir = "./qwen2.5-0.5b-sft-gsm8k"
        os.makedirs(output_dir, exist_ok=True)
        model.module.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()