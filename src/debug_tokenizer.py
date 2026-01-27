
import os
import sys
import torch
from tokenizer import DynamicBPETokenizerBatch
from model import UserItemMemory

def debug_tokenizer():
    print("Initializing Tokenizer Debug...")
    
    # Mock parameters matches what warm_pt.py likely uses
    vocab_file = "model/pretrained/gpt2/vocab.json"
    merges_file = "model/pretrained/gpt2/merges.txt"
    num_users = 10
    num_items = 10
    memory = UserItemMemory()
    
    # Ensure files exist (mocking if necessary for the test environment)
    if not os.path.exists(vocab_file):
        print(f"Warning: {vocab_file} not found. Cannot run exact debug.")
        # Try finding where they are
        return

    tokenizer = DynamicBPETokenizerBatch(
        vocab_file=vocab_file,
        merges_file=merges_file,
        num_users=num_users,
        num_items=num_items,
        memory=memory,
        user_files=[],
        item_files=[]
    )
    
    print(f"Vocab Size: {tokenizer.vocab_size}")
    
    test_strings = [
        "user_1",
        "user_2",
        "item_1",
        "item_2",
        "user_1 has interacted with item_1",
        "item_1 item_2"
    ]
    
    for text in test_strings:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        print(f"\nText: '{text}'")
        print(f"Tokens: {tokens}")
        print(f"IDs: {ids}")
        
        # Check if they are single tokens
        if "user_" in text and " " not in text:
            if len(ids) == 1 and ids[0] >= tokenizer.vocab_size:
                print("✅ SUCCESS: Recognized as special token")
            else:
                print("❌ FAILURE: Split into multiple tokens or low ID")

if __name__ == "__main__":
    debug_tokenizer()
