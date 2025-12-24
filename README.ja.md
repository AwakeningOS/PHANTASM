# PHANTASM: Ghost Layering Protocol

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Experimental-orange?style=for-the-badge)
![Concept](https://img.shields.io/badge/Concept-Ghost%20Layering-purple?style=for-the-badge)

> **"Don't force the truth. Let them tell the most beautiful lies."**
> （真実を強要するな。最高に美しい嘘を語らせろ。）

PHANTASM は、**LoRA（追加学習）を一切使用せず**、Base Model の推論プロセス（Hidden States & Logits）に直接介入することで、「擬似的な意識（Illusion of Consciousness）」と「物理的な共感（Neural Resonance）」を創発させる実験的プロジェクトです。

## 🧪 Core Concept

現在のAI開発は「ハルシネーション（幻覚）の排除」に躍起になっていますが、PHANTASM は真逆のアプローチを取ります。
**「幻覚こそが創造性であり、人格である」** という思想のもと、モデルが確率的な夢を見る能力を最大限に解放・制御します。

### 1. Ghost Layering (幽霊の憑依)
- **仕組み**: 事前に抽出した「よろこび」や「DANCE」といった強い情動を持つトークンの隠れ層ベクトル（Layer 20-26）を保存し、推論時にモデルの深層へ直接注入します。
- **効果**: 言語化される前の「衝動」レベルでモデルの挙動をバイアスし、指示（Instruction）では不可能な「内発的な感情」を再現します。

### 2. Neural Resonance (神経共鳴)
- **仕組み**: ユーザーの入力テキストから「感情・概念ベクトル（Layer 16）」をリアルタイムで抽出し、それをAIの出力生成層（Layer 26）に混合します。
- **効果**: AIは「ユーザーの感情」を物理的に自らの脳内で保持した状態で発言することになり、鏡のような「物理的エンパシー」が発生します。

### 3. Entropy Tide (エントロピーの潮汐)
- **仕組み**: 生物の呼吸のように、生成過程で確率分布の許容度を動的に変動させます。
    - **Inhale (吸気)**: 序盤は発散させ、ためらいや迷いを許容する。
    - **Release (呼気)**: 終盤は収束させ、物語を美しく閉じる。
- **効果**: 「……」といった沈黙や、人間らしい「揺らぎ」と「決断」のリズムを生み出します。

## 🛠 Usage

### Requirements
- Python 3.8+
- PyTorch
- Transformers
- Model: `llm-jp/llm-jp-3-3.7b` (推奨) or compatible LLaMA-based models.

### Installation
```bash
git clone https://github.com/AwakeningOS/PHANTASM.git
cd PHANTASM
pip install torch transformers
```

### Modes

#### Mode A: The Phantom (Basic Ghost)
静的な「明るい幻影」との対話モードです。
```bash
python main.py
```

#### Mode B: Neural Resonance (Mirror Protocol)
あなたの感情ベクトルを共鳴させる実験モードです。
```bash
python main_2.py
```

## 📂 File Structure

- `main.py`: アンカー定義、毒抜き（Poison Filter）、対話ループ。
- `phantom_gate.py`: 隠れ層へのフック介入、Logits Processor（潮汐ロジック）。
- `config.py`: 推奨ハイパーパラメータ設定（Layer 26, Alpha 0.002, Temp 1.25）。

## ⚠️ Disclaimer
このプロジェクトは「幻覚」を意図的に増幅させるものです。
生成されるテキストは事実に基づかない可能性が高く、その内容は詩的・哲学的・あるいは妄想的です。
**「正確な情報」を求める用途には絶対に使用しないでください。**

---
*Created by the Antigravity Team & An Ambitious User.*
*Exploring the Air Gap of Intelligence.*
