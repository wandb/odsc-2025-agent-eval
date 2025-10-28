# ODSC West 2025 AI Agent Evaluation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QBiqqQOkYwRK9wvnYv4FA2C6VzGezbuw?usp=sharing)

Workshop materials for evaluating AI agents using **Weights & Biases Weave**.

## 🎯 Overview

This repository demonstrates comprehensive evaluation strategies for a Data Analysis Agent, including:

- **Ground truth evaluation** - Compare outputs to known correct answers
- **Tool selection validation** - Verify agents use appropriate tools
- **LLM-as-judge evaluation** - Use GPT-4 to assess answer quality
- **Execution metrics** - Track efficiency and error handling
- **Adversarial testing** - Test edge cases and failure modes

## 🚀 Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY=your-openai-api-key
export WANDB_API_KEY=your-wandb-api-key
```

Sign up for a free W&B account at https://wandb.ai and go to https://wandb.ai/authorize for an API key.

### 3. Run the Weave Evaluation

```bash
python evaluation.py
```

## 🛠 Example: Data Analysis Agent

The demo agent can:
- Load CSV datasets
- Calculate statistics
- Compute correlations
- Group and aggregate data
- Filter datasets

Example usage:
```python
from data_analysis_agent import DataAnalysisAgent

agent = DataAnalysisAgent()
agent.predict("Load tips.csv")
answer, trace = agent.predict("What is the average tip amount?")
```

## 💡 Key Takeaways

1. **Evaluation is critical** - You can't improve what you don't measure
2. **Multiple metrics needed** - No single evaluation captures everything
3. **Automation is key** - Manual evaluation doesn't scale
4. **Track over time** - Monitor improvements and catch regressions
5. **Tools help** - Weave makes evaluation easier and more comprehensive
