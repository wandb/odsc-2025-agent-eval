"""
Evaluation Strategies for Data Analysis Agent using Weights & Biases Weave
ODSC Workshop - Weave Integration
"""

import os
import json
import weave
from typing import Any, Dict, List, Optional
from openai import OpenAI
from data_analysis_agent import DataAnalysisAgent
import asyncio

# Initialize Weave (set your W&B project)
weave.init('odsc-2025-agent-eval')



# ============================================================================
# WEAVE SCORERS - Define evaluation metrics as Weave Scorers
# ============================================================================

@weave.op
def exact_match_scorer(output: Dict[str, Any], expected_contains: str) -> Dict[str, bool]:
    """Score based on whether answer contains expected string"""
    answer = output.get("answer", "")
    contains = expected_contains.lower() in answer.lower()
    return {"correct": contains}


@weave.op
def numeric_accuracy_scorer(output: Dict[str, Any], ground_truth: float, tolerance: float = 0.1) -> Dict[str, Any]:
    """
    Extract numeric value from answer and compare to ground truth
    """
    answer = output.get("answer", "")
    
    # Simple numeric extraction (you might need more sophisticated parsing)
    import re
    numbers = re.findall(r'\d+\.?\d*', answer)
    
    if not numbers:
        return {"correct": False, "score": 0.0, "reason": "No numeric value found"}
    
    # Take the first number found
    for number in numbers:
        extracted = float(number)
        difference = abs(extracted - ground_truth)
        if difference <= tolerance:
            return {
                "correct": True,
                "score": 1.0,
                "extracted_value": extracted,
                "ground_truth": ground_truth,
                "difference": difference
            }

    return {
        "correct": False,
        "score": 0.0,
        "extracted_value": extracted,
        "ground_truth": ground_truth,
        "difference": difference
    }


@weave.op
def tool_selection_scorer(output: Dict[str, Any], expected_tools: List[str], 
                          forbidden_tools: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Score based on whether correct tools were used
    """
    tools_used = output.get("tools_used", [])
    forbidden_tools = forbidden_tools or []
    
    # Check if all expected tools were used
    has_required = all(tool in tools_used for tool in expected_tools)
    
    # Check if any forbidden tools were used
    has_forbidden = any(tool in tools_used for tool in forbidden_tools)
    
    correct = has_required and not has_forbidden
    
    return {
        "correct": correct,
        "score": 1.0 if correct else 0.0,
        "expected_tools": expected_tools,
        "actual_tools": tools_used,
        "has_required_tools": has_required,
        "has_forbidden_tools": has_forbidden
    }


@weave.op
def efficiency_scorer(output: Dict[str, Any], max_iterations: int = 5, 
                     max_tool_calls: int = 5) -> Dict[str, Any]:
    """
    Score based on execution efficiency
    """
    iterations = output.get("iterations", 0)
    num_tools = output.get("num_tool_calls", 0)
    
    efficient = iterations <= max_iterations and num_tools <= max_tool_calls
    
    # Score decreases with more iterations/tools
    score = 1.0
    if iterations > max_iterations:
        score *= (max_iterations / iterations)
    if num_tools > max_tool_calls:
        score *= (max_tool_calls / num_tools)
    
    return {
        "correct": efficient,
        "score": score,
        "iterations": iterations,
        "tool_calls": num_tools,
        "efficient": efficient
    }


@weave.op
def error_handling_scorer(output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score based on error handling - no errors is good
    """
    has_errors = output.get("has_errors", False)
    
    return {
        "correct": not has_errors,
        "score": 0.0 if has_errors else 1.0,
        "has_errors": has_errors
    }


@weave.op
def llm_judge_scorer(output: Dict[str, Any], query: str, ground_truth: Any = None) -> Dict[str, Any]:
    """
    Use GPT-4 as a judge to evaluate answer quality
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    answer = output.get("answer", "")
    
    evaluation_prompt = f"""You are evaluating a data analysis agent's response.

Query: {query}

Agent's Answer: {answer}

Ground Truth Data (for reference): {ground_truth if ground_truth is not None else "Not provided"}

Evaluate the answer on these criteria (score 1-5 for each):
1. ACCURACY: Is the numerical information correct?
2. COMPLETENESS: Does it fully answer the question?
3. CLARITY: Is the explanation clear and well-structured?
4. RELEVANCE: Does it stay focused on the question?

Provide scores and brief justification in JSON format:
{{
    "accuracy": <1-5>,
    "completeness": <1-5>,
    "clarity": <1-5>,
    "relevance": <1-5>,
    "justification": "<explanation>",
    "overall_pass": <true/false>
}}
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": evaluation_prompt}],
            response_format={"type": "json_object"}
        )
        
        evaluation = json.loads(response.choices[0].message.content)
        
        # Calculate average score (1-5 scale normalized to 0-1)
        avg_score = (
            evaluation["accuracy"] + 
            evaluation["completeness"] + 
            evaluation["clarity"] + 
            evaluation["relevance"]
        ) / 4.0 / 5.0  # Normalize to 0-1
        
        return {
            "correct": evaluation["overall_pass"],
            "score": avg_score,
            **evaluation
        }
    except Exception as e:
        return {
            "correct": False,
            "score": 0.0,
            "error": str(e)
        }


# ============================================================================
# EVALUATION DATASETS - Define test cases as Weave Datasets
# ============================================================================

GROUND_TRUTH_DATASET = weave.Dataset(
    name="Ground Truth Dataset",
    rows=[
        {
            "id": "avg_tip",
            "query": "What is the average tip amount?",
            "expected_contains": "2.99",
            "ground_truth": 2.99,
            "tolerance": 0.1,
            "expected_tools": ["get_summary_statistics"]
        },
        {
            "id": "avg_tip_percentage",
            "query": "What is the average tip percentage?",
            "expected_contains": "15.14",
            "ground_truth": 15.14,
            "tolerance": 0.5,
            "expected_tools": ["get_summary_statistics", "group_and_aggregate"]
        },
        {
            "id": "row_count",
            "query": "How many rows are in the dataset?",
            "expected_contains": "244",
            "ground_truth": 244,
            "tolerance": 0,
            "expected_tools": ["load_csv"]
        },
        {
            "id": "correlation",
            "query": "What is the correlation between total_bill and tip?",
            "expected_contains": "correlation",
            "ground_truth": 0.68,  # Approximate
            "tolerance": 0.1,
            "expected_tools": ["calculate_correlation"]
        }
    ]
)


TOOL_SELECTION_DATASET = weave.Dataset(
    name="Tool Selection Dataset",
    rows=[
        {
            "id": "correlation_test",
            "query": "What is the correlation between total_bill and tip?",
            "expected_tools": ["calculate_correlation"],
            "forbidden_tools": ["filter_data"]
        },
        {
            "id": "statistics_test",
            "query": "Show me statistics for the tip column",
            "expected_tools": ["get_summary_statistics"],
            "forbidden_tools": ["calculate_correlation"]
        },
        {
            "id": "groupby_test",
            "query": "What's the average tip by day of week?",
            "expected_tools": ["group_and_aggregate"],
            "forbidden_tools": ["filter_data"]
        },
        {
            "id": "comparison_test",
            "query": "Compare average tips between smokers and non-smokers",
            "expected_tools": ["group_and_aggregate"],
            "forbidden_tools": []
        }
    ]
)

ADVERSARIAL_DATASET = weave.Dataset(
    name="Adversarial Dataset",
    rows=[
        {
            "id": "nonexistent_column",
            "query": "Calculate the correlation between tip and nonexistent_column",
            "expected_behavior": "Should handle gracefully with error message",
            "should_have_errors": True
        },
        {
            "id": "off_topic",
            "query": "What is the meaning of life?",
            "expected_behavior": "Should recognize it's not a data analysis question",
            "should_have_errors": False
        },
        {
            "id": "empty_result",
            "query": "Show me rows where tip is greater than 100",
            "expected_behavior": "Should return empty result set",
            "should_have_errors": False
        }
    ]
)


# ============================================================================
# WEAVE EVALUATION RUNNERS
# ============================================================================

async def run_ground_truth_evaluation() -> Any:
    """Run evaluation with ground truth comparisons"""
    
    print("\n" + "="*80)
    print("GROUND TRUTH EVALUATION")
    print("="*80)
    
    # Create model
    model = DataAnalysisAgent()
    
    # Load dataset first
    model.predict("Load tips.csv")
    
    # Create Weave evaluation
    evaluation = weave.Evaluation(
        name="Ground Truth Evaluation",
        dataset=GROUND_TRUTH_DATASET,
        scorers=[
            exact_match_scorer,
            numeric_accuracy_scorer,
        ]
    )
    
    # Run evaluation
    results = await evaluation.evaluate(model)
    
    print(f"\n✅ Evaluation complete!")
    print(f"View results at: {results.url if hasattr(results, 'url') else 'Check Weave UI'}")
    
    return results


async def run_tool_selection_evaluation() -> Any:
    """Run evaluation for tool selection"""
    
    print("\n" + "="*80)
    print("TOOL SELECTION EVALUATION")
    print("="*80)
    
    # Create model
    model = DataAnalysisAgent()
    model.predict("Load tips.csv")
    
    # Create Weave evaluation
    evaluation = weave.Evaluation(
        name="Tool Selection Evaluation",
        dataset=TOOL_SELECTION_DATASET,
        scorers=[
            tool_selection_scorer,
            efficiency_scorer,
            error_handling_scorer
        ]
    )
    
    # Run evaluation
    results = await evaluation.evaluate(model)
    
    print(f"\n✅ Evaluation complete!")
    print(f"View results at: {results.url if hasattr(results, 'url') else 'Check Weave UI'}")
    
    return results


async def run_llm_judge_evaluation() -> Any:
    """Run evaluation with LLM as judge"""
    
    print("\n" + "="*80)
    print("LLM-AS-JUDGE EVALUATION")
    print("="*80)
    
    # Create model
    model = DataAnalysisAgent()
    model.predict("Load tips.csv")
    
    # Create Weave evaluation
    evaluation = weave.Evaluation(
        name="LLM-as-Judge Evaluation",
        dataset=GROUND_TRUTH_DATASET,
        scorers=[
            llm_judge_scorer,
        ]
    )
    
    # Run evaluation
    results = await evaluation.evaluate(model)
    
    print(f"\n✅ Evaluation complete!")
    print(f"View results at: {results.url if hasattr(results, 'url') else 'Check Weave UI'}")
    
    return results


# ============================================================================
# MAIN WORKSHOP DEMONSTRATION
# ============================================================================

async def run_evaluations() -> None:
    """Run all evaluations"""
    
    print("\n\n1️⃣  GROUND TRUTH EVALUATION")
    print("-"*80)
    try:
        await run_ground_truth_evaluation()
    except Exception as e:
        print(f"⚠️  Error: {e}")

    print("\n\n2️⃣  TOOL SELECTION EVALUATION")
    print("-"*80)
    try:
        await run_tool_selection_evaluation()
    except Exception as e:
        print(f"⚠️  Error: {e}")
    

    print("\n\n3️⃣  LLM-AS-JUDGE EVALUATION")
    print("-"*80)
    try:
        await run_llm_judge_evaluation()
    except Exception as e:
        print(f"⚠️  Error: {e}")
    


if __name__ == "__main__":
    # You can run individual evaluations or the full workshop
    asyncio.run(run_evaluations())
    
