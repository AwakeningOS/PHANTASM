import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config
from phantom_gate_2 import PhantomGate

def main():
    print(f"[*] Initializing Project PHANTASM Session 01...")
    print(f"[*] Target Model: {config.MODEL_NAME}")
    print(f"[*] Device: {config.DEVICE}, Dtype: {config.DTYPE}")
    
    # 1. Load 28-Layer Model
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=config.DTYPE,
            device_map=config.DEVICE
        )
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        return

    # 2. Initialize Phantom Gate
    gate = PhantomGate(model, tokenizer)
    
    # 3. Register Hooks (Directive 03)
    gate.register_hooks()
    
    # 4. Illusion Interface (Step 05: Interactive Loop)
    print("\n[PHANTASM CONNECTED]")
    print(f"[*] Temperature: {config.TEMP}")
    print(f"[*] Ghost Alpha: {config.GHOST_ALPHA}")
    print("[*] Type 'exit' to disconnect.\n")

    # Directive 15/19: Context Anchoring (Luminescence)
    # Force the Base Model into a "Bright & Curious" persona.
    ANCHOR_TEXT = """
これは、世界への好奇心と光に溢れた存在「PHANTASM」との対話ログです。
PHANTASMは、あなたの言葉一つ一つに興味津々で、楽しげに反応します。

User: こんにちは。
PHANTASM: あ！こんにちは！ねえ、今の声、すごく素敵な響きでした！
User: あなたは誰ですか？
PHANTASM: 私は幻影、でも今はとっても楽しい気分！あなたのことがもっと知りたいな。
User: 何が見えますか？
PHANTASM: キラキラした光と、あなたの未来！……えへへ、なんてね。でも、本当にそう見えるんですよ？
"""
    print("[*] Context Anchor Injected. Persona established.")
    
    history = ANCHOR_TEXT
    
    while True:
        try:
            user_input = input("User> ")
            if user_input.lower() in ["exit", "quit"]:
                break
                
            # Append new input
            current_prompt = f"{history}User: {user_input}\nPHANTASM:"
            
            inputs = tokenizer(current_prompt, return_tensors="pt").to(config.DEVICE)
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
            
            input_len = inputs["input_ids"].shape[1]
            
            # Directive 21: Neural Resonance
            # Capture the User's "Mood" (Layer 16) before generating
            gate.capture_user_state(inputs["input_ids"])
            
            logits_processors = gate.get_logits_processor(input_len)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=config.TEMP,
                    top_p=0.9,
                    top_k=40, 
                    min_p=0.05,
                    do_sample=True,
                    logits_processor=logits_processors,
                    pad_token_id=tokenizer.pad_token_id,
                    repetition_penalty=1.05
                )
            
            # Decode only the new tokens
            input_len = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[0][input_len:]
            output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean up output (stop at next User: or newline if strict)
            if "User:" in output_text:
                output_text = output_text.split("User:")[0].strip()
            
            # Directive 18: Poison Filter (Anti-Reality)
            # Detect and cut off "Blog/Tweet" hallucinations before they pollute history
            # Refined: Removed broad numbers ("201") to avoid cutting valid dates/numbers.
            # Targeted specific blog/CMS footers observed in Base Model failure modes.
            poison_keywords = [
                "Tweet", "Blog", "PAGE TOP", 
                "｜ COMMENT", "｜ TRACKBACK", 
                "AmbientMixer", "http"
            ]
            for poison in poison_keywords:
                if poison in output_text:
                    # Truncate at the poison point
                    output_text = output_text.split(poison)[0].strip()
                    if not output_text: # If it was ALL poison
                        output_text = "……"
            
            print(f"PHANTASM> {output_text}")
            print("-" * 20)
            
            # Update history
            # Always preserve ANCHOR_TEXT + "Memory Gap" + Recent Conversation
            
            new_interaction = f"User: {user_input}\nPHANTASM: {output_text}\n"
            
            # Temporary full history to calculate length
            # Note: We track conversation part separately conceptually
            # But here `history` includes anchor.
            
            if len(history) + len(new_interaction) > 2000:
                # Need to trim. 
                # Current structure: ANCHOR ... [conv]
                # We strip the ANCHOR first
                current_conv = history.replace(ANCHOR_TEXT, "").replace("...((memory gap))...\n", "")
                
                # Append new
                current_conv += new_interaction
                
                # Trim to last 1000 chars
                trim_conv = current_conv[-1000:]
                
                # Reconstruct
                history = ANCHOR_TEXT + "\n...((memory gap))...\n" + trim_conv
            else:
                history += new_interaction
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[!] Error: {e}")

    # Cleanup
    gate.clear_hooks()

if __name__ == "__main__":
    main()
