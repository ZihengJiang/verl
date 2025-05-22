# torchrun --nproc_per_node=4 tests/engine/test_sft.py
import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from verl.engine.fsdp.engine_impl import FSDPConfig, FSDPEngine

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


# Main training function
def main():                        
    # Setup distributed environment
    rank, world_size = setup_distributed()

    # Download model from Hugging Face
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    if rank == 0:
        file_path = snapshot_download(repo_id=model_name)
    dist.barrier()
    
    # Setup traininig engine
    config = FSDPConfig(
        model_path = model_name,
    )
    engine = FSDPEngine(config)
    engine.init_model_and_optimizer()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    engine.set_loss_fn(loss_fn)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    
    # Load and prepare dataset
    dataset = load_dataset("openai/gsm8k", "main")["train"]
    dataset = dataset.map(lambda x: format_gsm8k_example(x, tokenizer))
    train_dataset = GSM8KDataset(dataset.shuffle(seed=42).select(range(1000)), tokenizer)
    
    # Distributed data loader
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    
    # Training loop
    num_epochs = 3
    total_steps = len(train_loader) * num_epochs
    
    for epoch in range(num_epochs):
        total_loss = 0
        for step, batch in enumerate(train_loader):
            engine.optimizer_zero_grad()
            outputs = engine.forward_backward_step(batch)
            loss = outputs.loss
            engine.optimizer_step()
            total_loss += loss.item()
            
            if rank == 0 and step % 10 == 0:
                step = step + epoch * len(train_loader)
                print(f"Epoch {epoch+1}/{num_epochs}, Step {step}/{total_steps}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()