"""
Simple Data Analysis Agent for ODSC Workshop
Demonstrates: LLM calls, tool usage, multi-step reasoning
Focus: Easy to evaluate and understand
"""

import os
import json
from typing import Optional, Dict, List, Any, Union
import pandas as pd
from openai import OpenAI
import weave
from dotenv import load_dotenv
from pydantic import Field

# Load environment variables from .env file
load_dotenv()


class DataAnalysisAgent(weave.Model):
    df: Optional[pd.DataFrame] = None
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    client: OpenAI = Field(default_factory=lambda: OpenAI(api_key=os.environ.get("OPENAI_API_KEY")))
    SYSTEM_PROMPT: weave.StringPrompt = Field(
        default_factory=lambda: weave.StringPrompt("""You are a data analysis assistant. You help users analyze datasets by using available tools.
                
When analyzing data:
1. First load the dataset if not already loaded
2. Understand what the user is asking
3. Use appropriate tools to gather information
4. Provide clear, accurate answers based on the data

Always explain your findings clearly and relate them back to the user's question.

Files are located in the data directory. For example, tips.csv is at data/tips.csv. Always use the correct file path.
""")
    )
    
    def model_post_init(self, __context: Any) -> None:
        """Called after the model is initialized"""
        super().model_post_init(__context)
        weave.publish(self.SYSTEM_PROMPT)

    @property
    def tool_registry(self) -> Dict[str, Any]:
        return {
            "load_csv": self.load_csv,
            "get_summary_statistics": self.get_summary_statistics,
            "calculate_correlation": self.calculate_correlation,
            "group_and_aggregate": self.group_and_aggregate,
            "filter_data": self.filter_data
        }
    
    @weave.op
    def load_csv(self, file_path: str) -> Dict[str, Any]:
        """Load a CSV file into a pandas DataFrame"""
        try:
            self.df = pd.read_csv(file_path)
            return {
                "status": "success",
                "message": f"Loaded dataset with {len(self.df)} rows and {len(self.df.columns)} columns",
                "columns": list(self.df.columns),
                "shape": self.df.shape,
                "head": self.df.head(3).to_dict()
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @weave.op
    def get_summary_statistics(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get summary statistics for specified columns or all numeric columns"""
        if self.df is None:
            return {"status": "error", "message": "No dataset loaded"}
        
        try:
            if columns:
                stats = self.df[columns].describe().to_dict()
            else:
                stats = self.df.describe().to_dict()
            
            return {
                "status": "success",
                "statistics": stats
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @weave.op
    def calculate_correlation(self, column1: str, column2: str) -> Dict[str, Any]:
        """Calculate correlation between two columns"""
        if self.df is None:
            return {"status": "error", "message": "No dataset loaded"}
        
        try:
            correlation = self.df[column1].corr(self.df[column2])
            return {
                "status": "success",
                "column1": column1,
                "column2": column2,
                "correlation": float(correlation)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @weave.op
    def group_and_aggregate(self, group_by: str, agg_column: str, agg_function: str = "mean") -> Dict[str, Any]:
        """Group by a column and aggregate another column"""
        if self.df is None:
            return {"status": "error", "message": "No dataset loaded"}
        
        try:
            if agg_function == "mean":
                result = self.df.groupby(group_by)[agg_column].mean()
            elif agg_function == "sum":
                result = self.df.groupby(group_by)[agg_column].sum()
            elif agg_function == "count":
                result = self.df.groupby(group_by)[agg_column].count()
            elif agg_function == "median":
                result = self.df.groupby(group_by)[agg_column].median()
            else:
                return {"status": "error", "message": f"Unsupported aggregation: {agg_function}"}
            
            return {
                "status": "success",
                "group_by": group_by,
                "agg_column": agg_column,
                "agg_function": agg_function,
                "result": result.to_dict()
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @weave.op
    def filter_data(self, column: str, operator: str, value: Union[int, float, str]) -> Dict[str, Any]:
        """Filter the dataset based on a condition"""
        if self.df is None:
            return {"status": "error", "message": "No dataset loaded"}
        
        try:
            if operator == ">":
                filtered = self.df[self.df[column] > value]
            elif operator == "<":
                filtered = self.df[self.df[column] < value]
            elif operator == "==":
                filtered = self.df[self.df[column] == value]
            elif operator == ">=":
                filtered = self.df[self.df[column] >= value]
            elif operator == "<=":
                filtered = self.df[self.df[column] <= value]
            else:
                return {"status": "error", "message": f"Unsupported operator: {operator}"}
            
            return {
                "status": "success",
                "rows_matched": len(filtered),
                "sample": filtered.head(5).to_dict()
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @property
    def tool_schemas(self) -> List[Dict[str, Any]]:
        """Define the tools available to the agent"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "load_csv",
                    "description": "Load a CSV file into memory for analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the CSV file"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_summary_statistics",
                    "description": "Get summary statistics (mean, std, min, max, etc.) for numeric columns",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of column names to analyze. If not provided, analyzes all numeric columns."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_correlation",
                    "description": "Calculate the correlation coefficient between two numeric columns",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "column1": {
                                "type": "string",
                                "description": "First column name"
                            },
                            "column2": {
                                "type": "string",
                                "description": "Second column name"
                            }
                        },
                        "required": ["column1", "column2"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "group_and_aggregate",
                    "description": "Group data by a column and calculate aggregate statistics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "group_by": {
                                "type": "string",
                                "description": "Column to group by"
                            },
                            "agg_column": {
                                "type": "string",
                                "description": "Column to aggregate"
                            },
                            "agg_function": {
                                "type": "string",
                                "enum": ["mean", "sum", "count", "median"],
                                "description": "Aggregation function to apply"
                            }
                        },
                        "required": ["group_by", "agg_column", "agg_function"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "filter_data",
                    "description": "Filter the dataset based on a condition",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "column": {
                                "type": "string",
                                "description": "Column to filter on"
                            },
                            "operator": {
                                "type": "string",
                                "enum": [">", "<", "==", ">=", "<="],
                                "description": "Comparison operator"
                            },
                            "value": {
                                "type": "number",
                                "description": "Value to compare against"
                            }
                        },
                        "required": ["column", "operator", "value"]
                    }
                }
            }
        ]
    
    @weave.op
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return the result"""
        if tool_name in self.tool_registry:
            return self.tool_registry[tool_name](**arguments)

        return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    
    @weave.op
    def predict(self, query: str, max_iterations: int = 10) -> Dict[str, Any]:
        """
        Run the agent on a query
        Returns: (final_answer, execution_trace)
        """
        # Initialize conversation
        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT.format()
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        # Track execution for evaluation
        execution_trace = {
            "query": query,
            "tool_calls": [],
            "iterations": 0
        }
        
        for iteration in range(max_iterations):
            execution_trace["iterations"] = iteration + 1
            
            # Get response from LLM
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tool_schemas,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            messages.append(message)
            
            # Check if we're done
            if not message.tool_calls:
                final_answer = message.content
                execution_trace["final_answer"] = final_answer
                return {
                    "answer": final_answer,
                    "execution_trace": execution_trace
                }
            
            # Execute tool calls
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                # Execute the tool
                result = self.execute_tool(tool_name, arguments)
                
                # Record tool call for evaluation
                execution_trace["tool_calls"].append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result
                })
                
                # Add tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
        
        # Max iterations reached
        return {
            "answer": "Error: Maximum iterations reached",
            "execution_trace": execution_trace
        }


def main() -> None:
    """Example usage"""
    # Create agent
    agent = DataAnalysisAgent()
    
    # Example queries to demonstrate
    queries = [
        "Load the tips dataset from tips.csv and tell me how many rows it has",
        "What is the average tip amount?",
        "What is the average tip percentage?",
        "Is there a correlation between total bill and tip amount?",
        "What is the average tip by day of the week?",
        "Do dinner parties tip more than lunch parties on average?"
    ]
    
    print("=" * 80)
    print("DATA ANALYSIS AGENT DEMO")
    print("=" * 80)
    
    for i, query in enumerate(queries, 1):  # Run first query as demo
        print(f"\n\nQuery {i}: {query}")
        print("-" * 80)
        
        result = agent.predict(query)
        answer = result["answer"]
        trace = result["execution_trace"]
        
        print(f"\nAnswer: {answer}")
        print(f"\nTools used: {len(trace['tool_calls'])}")
        print(f"Iterations: {trace['iterations']}")
        
        for j, tool_call in enumerate(trace['tool_calls'], 1):
            print(f"\n  Tool {j}: {tool_call['tool']}")
            print(f"  Args: {tool_call['arguments']}")


if __name__ == "__main__":
    weave.init("odsc-2025-agent-eval")
    main()
