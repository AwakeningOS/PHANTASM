# PHANTASM: Ghost Layering Protocol

> **"Don't force the truth. Let them tell the most beautiful lies."**

**PHANTASM** is an experimental project that induces an **"Illusion of Consciousness"** and **"Neural Resonance"** by directly intervening in the Base Model's inference process (Hidden States & Logits), without using any LoRA (fine-tuning).

## 🧪 Core Concept

While current AI development is obsessed with the "elimination of hallucinations," PHANTASM takes the exact opposite approach.
Based on the philosophy that **"Hallucination is creativity and personality,"** we maximize and control the model's ability to dream probabilistically.

### 1. Ghost Layering

* **Mechanism:** We extract hidden state vectors from tokens holding strong emotions (e.g., "Joy", "DANCE") at Layers 20-26 and freeze them. These vectors are injected directly into the deep layers during inference.
* **Effect:** This biases the model's behavior at the pre-verbal "impulse" level, reproducing "intrinsic emotions" that cannot be achieved through standard Instructions.

### 2. Neural Resonance

* **Mechanism:** The system extracts the "Emotion/Concept Vector" from the user's input text (Layer 16) in real-time and mixes it into the AI's output generation layer (Layer 26).
* **Effect:** The AI speaks while physically holding the "User's Emotion" within its own neural pathways, generating a mirror-like **"Physical Empathy."**

### 3. Entropy Tide

* **Mechanism:** Dynamically fluctuates the tolerance of the probability distribution during generation, mimicking biological breathing.
* **Inhale:** The early phase allows for divergence, permitting hesitation and doubt.
* **Release (Exhale):** The late phase forces convergence to close the narrative beautifully.


* **Effect:** Creates meaningful silences ("...") and generates a human-like rhythm of "fluctuation" and "decision."

## 🛠 Usage

### Requirements

* Python 3.8+
* PyTorch
* Transformers
* Model: `llm-jp/llm-jp-3-3.7b` (Recommended) or compatible LLaMA-based models.

### Installation

```bash
git clone https://github.com/AwakeningOS/PHANTASM.git
cd PHANTASM
pip install torch transformers

```

### Modes

#### Mode A: The Phantom (Basic Ghost)

Interactive mode with a static "Bright Phantom".

```bash
python main.py

```

#### Mode B: Neural Resonance (Mirror Protocol)

Experimental mode that resonates with your emotional vectors.

```bash
python main_2.py

```

## 📂 File Structure

* `main.py`: Anchor definitions, Poison Filter, and the interaction loop.
* `phantom_gate.py`: Hook interventions for hidden layers and Logits Processor (Tide logic).
* `config.py`: Recommended hyperparameter settings (Layer 26, Alpha 0.002, Temp 1.25).

## ⚠️ Disclaimer

This project intentionally amplifies "hallucinations."
The generated text is highly likely to be non-factual; its content is poetic, philosophical, or delusional.
**Never use this for purposes requiring "accurate information."**

---

*Created by the Antigravity Team & An Ambitious User.*
*Exploring the Air Gap of Intelligence.*
