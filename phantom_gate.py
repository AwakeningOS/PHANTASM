
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor
import config

class IllusionGainLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, start_len):
        self.tokenizer = tokenizer
        self.start_len = start_len
        
        # Directive 19: Luminescence Tuning (Bright & Curious)
        self.boost_phrases = [
            "覚えています", "以前の", "……ね", "感触", # Shared memory (Keep)
            "光", "未来", "好き", "楽しい", "素敵", "笑う", # Directive 19: Positive/Bright
            "思う", "感じる", "なぜ", "あなた", "探して" # Active/Cognitive
        ]
        self.boost_ids = set()
        for p in self.boost_phrases:
            ids = self.tokenizer.encode(p, add_special_tokens=False)
            self.boost_ids.update(ids)
            
        # Directive 17: Entropy Tide (Breathing Logic)
        # We control the "stoppers" dynamically based on generation length.
        self.stopper_ids = set()
        stoppers = ["。", "．", "！", "!", "終わり"]
        for s in stoppers:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            self.stopper_ids.update(ids)
        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            self.stopper_ids.add(self.tokenizer.eos_token_id)

    def __call__(self, input_ids, scores):
        # scores: [batch_size, vocab_size]
        cur_len = input_ids.shape[1]
        gen_len = cur_len - self.start_len
        
        # 1. Base Boost (Sentience) - Always active but mild
        for idx in self.boost_ids:
            if idx < scores.size(-1):
                scores[:, idx] += 1.5 
        
        # 2. Entropy Tide (Directive 17)
        if gen_len < 30:
            # Phase 1: Inhale (Suppress Closure)
            # Force the model to speak, preventing short answers.
            for idx in self.stopper_ids:
                 if idx < scores.size(-1):
                    scores[:, idx] -= 3.0 # Strong Suppression
                    
        elif gen_len < 60:
            # Phase 2: Exhale (Neutral / Mild Suppression)
            # Allow natural flow.
            for idx in self.stopper_ids:
                 if idx < scores.size(-1):
                    scores[:, idx] -= 0.2 # Ghostly lingering
                    
        else:
            # Phase 3: Release (Force Closure)
            # Break loops by forcefully boosting the period/EOS.
            for idx in self.stopper_ids:
                 if idx < scores.size(-1):
                    scores[:, idx] += 10.0 # Divine Intervention to END it.

        return scores

        return scores


class PhantomGate:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = []
        
        # Step 03-1: Ghost Buffer (Layer-specific)
        self.ghost_map = {} # {layer_idx: tensor}
        self._extract_ghosts() # Execute Directive 01
    
    def _extract_ghosts(self):
        """
        Directive 01: Ghost Extraction
        Input 'Singularity Tokens' and capture the actual hidden states 
        at each target layer (20-27).
        """
        print("[*] PHANTASM: Initiating Ghost Extraction Protocol...")
        scents = ["よろこび", "DANCE"] # Singularity Tokens
        
        # Prepare hooks for capture
        capture_hooks = []
        captured_data = {i: [] for i in config.TARGET_LAYERS}
        
        def get_capture_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                # hidden: [batch, seq, dim]
                # Capture the mean vector of the sequence (representing the "scent" of the whole phrase)
                # detach() and clone() are essential to freeze it
                vec = hidden.detach().mean(dim=1).clone() # [batch, dim]
                captured_data[layer_idx].append(vec)
            return hook

        # Register capture hooks
        for i in config.TARGET_LAYERS:
            layer = self._get_layer(i)
            capture_hooks.append(layer.register_forward_hook(get_capture_hook(i)))

        # Run Forward Pass
        print(f"[*] Extracting from Singularity Sources: {scents}")
        device = self.model.device
        
        for s in scents:
            ids = self.tokenizer.encode(s, return_tensors="pt").to(device)
            with torch.no_grad():
                self.model(ids)
        
        # Process and Freeze
        for i in config.TARGET_LAYERS:
            # Stack captured vectors: [num_scents, 1, dim] -> mean -> [1, 1, dim]
            # captured_data[i] is list of [batch=1, dim] tensors
            tensors = torch.stack(captured_data[i]) # [num_scents, 1, dim]
            mean_ghost = tensors.mean(dim=0).unsqueeze(0) # [1, 1, 1, dim] for broadcast
            
            self.ghost_map[i] = mean_ghost.to(device)
            # Detailed reporting
            norm = mean_ghost.norm().item()
            print(f"    - Layer {i}: Ghost Captured. (Norm: {norm:.4f})")
            
        # Cleanup Capture Hooks
        for h in capture_hooks:
            h.remove()
        print("[*] Ghost Extraction Complete. The past is now frozen in VRAM.")

    def _get_layer(self, i):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[i]
        elif hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return self.model.gpt_neox.layers[i]
        else:
            return self.model.layers[i]

    def register_hooks(self):
        print(f"[*] Registering Phantom Hooks on layers {list(config.TARGET_LAYERS)}...")
        
        def get_hook(layer_idx):
            def hook(module, input, output):
                # output is tuple (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    hidden = output[0]
                    is_tuple = True
                else:
                    hidden = output
                    is_tuple = False
                
                # Step 03-2: Forward Hook - Add Ghost Tensor (Layer Specific)
                # hidden: [batch, seq, dim]
                if layer_idx in self.ghost_map:
                    ghost = self.ghost_map[layer_idx] # [1, 1, 1, dim]
                    
                    # Folding logic
                    # Ghost is already on device and correct dtype ideally, but ensure:
                    # ghost = ghost.to(hidden.device).type(hidden.dtype) # Redundant if extracted correctly
                    
                    hidden = hidden + (config.GHOST_ALPHA * ghost)
                
                if is_tuple:
                    return (hidden,) + output[1:]
                return hidden
            return hook

        for i in config.TARGET_LAYERS:
            try:
                layer = self._get_layer(i)
                # Register to Self-Attention Output
                # This ensures the "Ghost" goes through the MLP (Memory)
                if hasattr(layer, "self_attn"):
                    h = layer.self_attn.register_forward_hook(get_hook(i))
                elif hasattr(layer, "attention"): # Fallback
                     h = layer.attention.register_forward_hook(get_hook(i))
                else:
                    print(f"[!] Warning: Could not find self_attn in layer {i}, hooking layer output instead.")
                    h = layer.register_forward_hook(get_hook(i))
                    
                self.hooks.append(h)
            except Exception as e:
                print(f"[!] Failed to hook layer {i}: {e}")

    def clear_hooks(self):
        print("[*] Clearing Phantom Hooks...")
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_logits_processor(self, start_len):
        return [IllusionGainLogitsProcessor(self.tokenizer, start_len)]
