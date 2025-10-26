# ODSC West 2025 AI Agent Evaluation

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
```

### 3. Run the Weave Evaluation

```bash
python run_weave_eval.py
```

This will:
- Initialize Weave tracking
- Run evaluation tests
- Generate a link to view results in the Weave UI

## 📁 Project Structure

```
.
├── data_analysis_agent.py    # Core agent implementation
├── demo.py                    # Basic demo script
├── evaluation_weave.py        # ⭐ Weave-based evaluation harness
├── run_weave_eval.py          # Quick start evaluation script
├── WEAVE_GUIDE.md            # Comprehensive Weave documentation
└── pyproject.toml            # Dependencies (includes weave)
```

## 📊 Evaluation with Weave

The new `evaluation_weave.py` provides:

### Automatic Tracking
- All predictions logged automatically
- Full execution traces with tool calls
- Rich visualization in W&B UI

### Multiple Evaluation Types
```python
# Ground truth evaluation
run_ground_truth_evaluation()

# Tool selection validation
run_tool_selection_evaluation()

# LLM-as-judge
run_llm_judge_evaluation()

# Comprehensive (all together)
run_comprehensive_evaluation()
```

### Custom Scorers
Define your own evaluation metrics:

```python
@weave.op()
def custom_scorer(model_output: dict, expected: str) -> dict:
    return {"correct": expected in model_output["answer"], "score": 1.0}
```

## 🎓 Workshop Content

### Part 1: Agent Evaluation Challenges
- Why evaluating agents is hard
- Different types of correctness
- Scalability considerations

### Part 2: Evaluation Strategies
- Unit testing tools
- Ground truth comparisons
- Tool selection validation
- LLM-as-judge approaches
- Execution metrics

### Part 3: Weave Integration
- Automatic tracking
- Creating evaluations
- Building custom scorers
- Analyzing results

### Part 4: Best Practices
- Building eval datasets
- Combining multiple metrics
- Tracking improvements over time
- Production monitoring

## 📚 Documentation

- **[WEAVE_GUIDE.md](WEAVE_GUIDE.md)** - Complete guide to using Weave for evaluation
- **[evaluation_weave.py](evaluation_weave.py)** - Full implementation with examples

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
agent.run("Load tips.csv")
answer, trace = agent.run("What is the average tip amount?")
```

With Weave:
```python
from evaluation_weave import DataAnalysisWeaveModel

model = DataAnalysisWeaveModel()
result = model.predict("What is the average tip amount?")
# Automatically tracked in Weave!
```

## 🔗 Resources

- **Weave Documentation**: https://weave-docs.wandb.ai/
- **W&B Platform**: https://wandb.ai
- **ODSC West 2025**: Workshop materials

## 💡 Key Takeaways

1. **Evaluation is critical** - You can't improve what you don't measure
2. **Multiple metrics needed** - No single evaluation captures everything
3. **Automation is key** - Manual evaluation doesn't scale
4. **Track over time** - Monitor improvements and catch regressions
5. **Tools help** - Weave makes evaluation easier and more comprehensive

## 🤝 Contributing

This is workshop material. Feel free to:
- Extend the evaluation strategies
- Add new scorers
- Create more test cases
- Share your results

## 📄 License

MIT License - See workshop materials for details.