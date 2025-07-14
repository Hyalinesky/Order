from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer
import json
import torch

# Load tokenizer
model_name = "../models/squad_1k/checkpoints"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def load_order_fair_data(file_path, num_orders=8, max_prompt_tokens=2048, max_total_tokens=3072):
    """
    Load data from jsonl file and format for order-fair training.
    
    Args:
        file_path: Path to the jsonl file
        num_orders: Number of orders per group (should be 8)
        max_prompt_tokens: Maximum tokens allowed for a single prompt (default: 2048)
        max_total_tokens: Maximum total tokens for the conversation (default: 3072)
    
    Returns:
        List of formatted data with prompt_variants and answers
    """
    data_items = []
    
    # Read jsonl file
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                data_items.append(json.loads(line))
    
    print(f"Loaded {len(data_items)} items from {file_path}")
    
    # Check if the number of items is divisible by num_orders
    if len(data_items) % num_orders != 0:
        print(f"Warning: Total items ({len(data_items)}) is not divisible by num_orders ({num_orders})")
        # Trim to make it divisible
        data_items = data_items[:len(data_items) // num_orders * num_orders]
        print(f"Trimmed to {len(data_items)} items")
    
    formatted_data = []
    skipped_groups = 0
    
    # Process data in groups of num_orders
    for i in range(0, len(data_items), num_orders):
        group = data_items[i:i + num_orders]
        
        # Check token lengths for all items in the group
        skip_group = False
        for item in group:
            # Create messages for token counting
            messages = [
                {"role": "system", "content": item.get("system", "You are a helpful assistant.")},
                {"role": "user", "content": item["prompt"]}
            ]
            
            # Count tokens for the prompt only (user message)
            prompt_tokens = len(tokenizer.encode(item["prompt"], add_special_tokens=False))
            
            # Count tokens for the entire conversation
            conversation_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            total_tokens = len(tokenizer.encode(conversation_text, add_special_tokens=True))
            
            # Check if prompt exceeds max_prompt_tokens or total exceeds max_total_tokens
            if prompt_tokens > max_prompt_tokens:
                print(f"Skipping group {i//num_orders + 1}: Prompt too long ({prompt_tokens} > {max_prompt_tokens} tokens)")
                skip_group = True
                break
            elif total_tokens > max_total_tokens:
                print(f"Skipping group {i//num_orders + 1}: Total conversation too long ({total_tokens} > {max_total_tokens} tokens)")
                skip_group = True
                break
        
        if skip_group:
            skipped_groups += 1
            continue
        
        # Check if all responses in the group are consistent
        responses = [item["response"] for item in group]
        if not all(resp == responses[0] for resp in responses):
            print(f"Warning: Inconsistent responses in group starting at index {i}")
            print(f"Responses: {responses}")
            # You can choose to skip this group or continue with warning
            # continue  # Uncomment this line to skip inconsistent groups
        
        # Extract the common answer (response)
        answer = responses[0]
        
        # Construct prompt variants for this group
        prompt_variants = []
        for item in group:
            # Format as conversational messages
            messages = [
                {"role": "system", "content": item.get("system", "You are a helpful assistant.")},
                {"role": "user", "content": item["prompt"]}
            ]
            prompt_variants.append(messages)
        
        # Add to formatted data
        formatted_data.append({
            "prompt_variants": prompt_variants,
            "answer": answer
        })
    
    print(f"Created {len(formatted_data)} groups with {num_orders} variants each")
    print(f"Skipped {skipped_groups} groups due to token length constraints")
    print(f"Total groups processed: {len(data_items) // num_orders}")
    print(f"Success rate: {len(formatted_data)}/{len(data_items) // num_orders} ({len(formatted_data)/(len(data_items) // num_orders)*100:.1f}%)")
    
    return formatted_data

def validate_data_consistency(data_list, num_orders=8):
    """
    Validate that each group has the expected number of variants
    and print some statistics
    """
    print("\n=== Data Validation ===")
    print(f"Total groups: {len(data_list)}")
    print(f"Expected variants per group: {num_orders}")
    
    variant_counts = [len(item["prompt_variants"]) for item in data_list]
    print(f"Actual variants per group: {set(variant_counts)}")
    
    if len(set(variant_counts)) == 1 and variant_counts[0] == num_orders:
        print("✓ All groups have the correct number of variants")
    else:
        print("✗ Some groups have incorrect number of variants")
    
    # Show a sample with token counts
    if data_list:
        print("\n=== Sample Data with Token Counts ===")
        sample = data_list[0]
        print(f"Answer: {sample['answer']}")
        print(f"Number of prompt variants: {len(sample['prompt_variants'])}")
        
        # Check token counts for the first variant
        first_variant = sample['prompt_variants'][0]
        prompt_text = first_variant[1]["content"]  # User message
        prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        
        conversation_text = tokenizer.apply_chat_template(
            first_variant, 
            tokenize=False, 
            add_generation_prompt=True
        )
        total_tokens = len(tokenizer.encode(conversation_text, add_special_tokens=True))
        
        print(f"Sample prompt tokens: {prompt_tokens}")
        print(f"Sample total conversation tokens: {total_tokens}")
        
        print("First variant:")
        for msg in sample['prompt_variants'][0]:
            print(f"  {msg['role']}: {msg['content'][:100]}...")
        print("Second variant:")
        for msg in sample['prompt_variants'][1]:
            print(f"  {msg['role']}: {msg['content'][:100]}...")

# Load the dataset with token length constraints
file_path = "data/squad_v2/train_8_1000.jsonl"
num_orders = 8
max_prompt_tokens = 2048
max_total_tokens = 4096

print("Loading and processing data...")
data_list = load_order_fair_data(
    file_path, 
    num_orders=num_orders,
    max_prompt_tokens=max_prompt_tokens,
    max_total_tokens=max_total_tokens
)

# Validate the data
validate_data_consistency(data_list, num_orders)

# Create a Hugging Face dataset
dataset = Dataset.from_list(data_list)

# Training configuration
training_args = GRPOConfig(
    output_dir="results/OrderFair-GRPO-squad",
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=8, 
    bf16=True,
    logging_steps=10,
    num_train_epochs=3.0,
    beta=0.05,
    num_orders=8,
    save_steps=250,
    temperature=0.5,
    top_p=0.5,
    gradient_checkpointing=False,
    steps_per_generation=1,  # 设置为1，每步都生成新的completions
    num_iterations=1,
    max_completion_length=128,
    max_prompt_length=None,
)

# Create trainer with alpha parameter for advantage mixing
trainer = GRPOTrainer(
    model=model_name,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,  # Use processing_class instead of tokenizer
    alpha=0.5,  # Weight for combining group and individual advantages
    task='squad',

)

print("Starting training...")
trainer.train()

print("Training completed!")