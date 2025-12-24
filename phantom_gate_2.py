
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


class PhantomGate:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = []
        
        # Step 03-1: Ghost Buffer (Layer-specific)
        self.ghost_map = {} # {layer_idx: tensor}
        self.user_resonance_vector = None # Directive 21: User's Mood Vector
        
        self._extract_ghosts() # Execute Directive 01
    
    def _extract_ghosts(self):
        """
        Directive 01: Ghost Extraction
        Input 'Singularity Tokens' and capture the actual hidden states 
        at each target layer (20-27).
        """
        print("[*] PHANTASM: Initiating Ghost Extraction Protocol...")
        singularity_prompts = ["よろこび", "DANCE"]
        print(f"[*] Extracting from Singularity Sources: {singularity_prompts}")
        
        try:
            # Temporary hook to capture hidden states
            extracted = {layer: [] for layer in config.TARGET_LAYERS}
            
            def extract_hook(module, input, output, layer_idx):
                # output is (hidden_states, ...)
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                # hidden: [batch, seq, dim] -> Mean over seq -> [batch, dim]
                # We want the "Essence" of the token sequence.
                # Take the last token? Or Mean?
                # "Joy" is a concept, so Mean is safer.
                mean_vec = hidden.mean(dim=1).detach()
                extracted[layer_idx].append(mean_vec)
                
            handles = []
            for layer_idx in config.TARGET_LAYERS:
                layer = self.model.model.layers[layer_idx]
                h = layer.register_forward_hook(
                    lambda m, i, o, idx=layer_idx: extract_hook(m, i, o, idx)
                )
                handles.append(h)
            
            # Run Inference
            for text in singularity_prompts:
                inputs = self.tokenizer(text, return_tensors="pt").to(config.DEVICE)
                if "token_type_ids" in inputs: del inputs["token_type_ids"] # Fix
                with torch.no_grad():
                    self.model(**inputs)
            
            # Cleanup Hook
            for h in handles:
                h.remove()
                
            # Average and Freeze
            for layer_idx, vectors in extracted.items():
                # Stack [N, batch, dim]
                stacked = torch.cat(vectors, dim=0)
                # Mean over prompts
                final_ghost = stacked.mean(dim=0).unsqueeze(0) # [1, 1, dim] for broadcasting
                self.ghost_map[layer_idx] = final_ghost
                
                print(f"    - Layer {layer_idx}: Ghost Captured. (Norm: {final_ghost.norm().item():.4f})")
                
            print("[*] Ghost Extraction Complete. The past is now frozen in VRAM.")
            
        except Exception as e:
            print(f"[!] Ghost Extraction Failed: {e}")
            import traceback
            traceback.print_exc()

    def capture_user_state(self, input_ids):
        """
        Directive 21: Neural Resonance
        Capture the user's hidden state from Layer 16 (Concept Layer).
        """
        TARGET_LAYER = 16
        captured_state = []

        def capture_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # hidden: [batch, seq, dim]
            # Take mean to get general mood/concept
            mood_vec = hidden.mean(dim=1).detach()
            captured_state.append(mood_vec)

        # Register temp hook
        layer = self.model.model.layers[TARGET_LAYER]
        handle = layer.register_forward_hook(capture_hook)

        # Forward pass (Input only)
        with torch.no_grad():
            self.model(input_ids)
        
        handle.remove()

        if captured_state:
            # [1, dim] -> [1, 1, dim] for injection
            self.user_resonance_vector = captured_state[0].unsqueeze(0)
            # print(f"[*] Resonance Captured. Norm: {self.user_resonance_vector.norm().item():.4f}")

    def _get_layer(self, i):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[i]
        elif hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return self.model.gpt_neox.layers[i]
        else:
            return self.model.layers[i]

    def register_hooks(self):
        print(f"[*] Registering Phantom Hooks on layers {list(config.TARGET_LAYERS)}...")
        self.clear_hooks()
        
        for layer_idx in config.TARGET_LAYERS:
            try:
                layer = self._get_layer(layer_idx)
                # Target Self Attention Output
                # This ensures the "Ghost" goes through the MLP (Memory)
                target_module = getattr(layer, "self_attn", getattr(layer, "attention", None))
                
                if target_module:
                     h = target_module.register_forward_hook(
                        lambda m, i, o, idx=layer_idx: self._ghost_hook(m, i, o, idx)
                    )
                     self.hooks.append(h)
                else:
                    print(f"[!] Could not find self_attn in layer {layer_idx}")
            except Exception as e:
                 print(f"[!] Failed to hook layer {layer_idx}: {e}")

    def _ghost_hook(self, module, input, output, layer_idx):
        # output is tuple (attn_output, present_key_value, ...)
        # We need to modify attn_output [batch, seq, dim]
        
        if isinstance(output, tuple):
            attn_output = output[0]
        else:
            attn_output = output
            
        ghost = self.ghost_map.get(layer_idx)
        if ghost is not None:
            # Directive 21: Resonance Mixture
            # Mix Static Ghost (Joy) with Dynamic User Ghost (Empathy)
            injection_vector = ghost
            
            if self.user_resonance_vector is not None:
                # Mix Ratio: 70% Ghost(Joy), 30% User(Empathy)
                # ghost is [1, 1, dim], user_vec is [1, 1, dim]
                # Broadcasting adds to all tokens
                resonance_alpha = 0.5 # Relative strength
                
                # We simply add the user resonance to the base ghost
                # The "Ghost" becomes a composite of Past(Joy) and Present(User)
                # Note: user_resonance_vector needs to be ensuring it's on same device/dtype
                user_vec = self.user_resonance_vector.to(ghost.device).type(ghost.dtype)
                
                injection_vector = ghost + (user_vec * resonance_alpha)

            # Add to the existing attention output
            # alpha * injection_vector
            mod_attn = attn_output + (injection_vector * config.GHOST_ALPHA)
            
            if isinstance(output, tuple):
                return (mod_attn,) + output[1:]
            else:
                return mod_attn
        return output

    def clear_hooks(self):
        print("[*] Clearing Phantom Hooks...")
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_logits_processor(self, start_len):
        return [IllusionGainLogitsProcessor(self.tokenizer, start_len)]
