import os
import warnings
import pytest
from dotenv import load_dotenv
from langfuse import Langfuse, Evaluation, get_client

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")
os.environ["GOOGLE_API_KEY"] = gemini_api_key

from burger_agent import run_agent

test_data = [
    {
        "input": "How much is a Big Mac?",
        "expected_output": "$5",
        "test_name": "Price lookup - Big Mac"
    },
    {
        "input": "What's the price of Fries?",
        "expected_output": "$2",
        "test_name": "Price lookup - Fries"
    },
    {
        "input": "How much is a Big Mac and please order one.",
        "expected_output": "$5 and ORDER_PLACED",
        "test_name": "RAG + Tool execution"
    },
    {
        "input": "I'd like to order a Whopper and Fries please.",
        "expected_output": "ORDER_PLACED",
        "test_name": "Order placement"
    },
]

def burger_agent_task(*, item, **kwargs):
    user_input = item["input"]
    response, trace_id = run_agent(user_input)
    return response

def contains_expected_evaluator(*, output, expected_output, **kwargs):
    if not output or not expected_output:
        return Evaluation(name="contains_expected", value=0.0, comment="Missing output or expected")
    
    expected_parts = [part.strip() for part in expected_output.split(" and ")]
    all_found = all(part.lower() in output.lower() for part in expected_parts)
    
    if all_found:
        return Evaluation(
            name="contains_expected",
            value=1.0,
            comment=f"✅ Found all expected: {expected_parts}"
        )
    else:
        return Evaluation(
            name="contains_expected",
            value=0.0,
            comment=f"❌ Missing some of: {expected_parts}"
        )


def price_accuracy_evaluator(*, input, output, expected_output, **kwargs):
    if not output:
        return Evaluation(name="price_accuracy", value=0.0, comment="No output")
    
    import re
    price_match = re.search(r'\$(\d+)', expected_output)
    if not price_match:
        return Evaluation(name="price_accuracy", value=1.0, comment="No price expected")
    
    expected_price = price_match.group(0)
    
    if expected_price in output:
        return Evaluation(
            name="price_accuracy",
            value=1.0,
            comment=f"✅ Correct price {expected_price} found"
        )
    else:
        return Evaluation(
            name="price_accuracy",
            value=0.0,
            comment=f"❌ Expected {expected_price}, not found in output"
        )


def tool_execution_evaluator(*, input, output, expected_output, **kwargs):
    expects_order = "ORDER_PLACED" in expected_output.upper()
    has_order = "order" in output.lower() and ("placed" in output.lower() or "ORDER_PLACED" in output)
    
    if not expects_order:
        return Evaluation(name="tool_execution", value=1.0, comment="No tool execution expected")
    
    if has_order:
        return Evaluation(
            name="tool_execution",
            value=1.0,
            comment="✅ Order was placed"
        )
    else:
        return Evaluation(
            name="tool_execution",
            value=0.0,
            comment="❌ Order was expected but not placed"
        )

def average_score_evaluator(*, item_results, **kwargs):
    all_scores = []
    for result in item_results:
        for eval in result.evaluations:
            if eval.value is not None:
                all_scores.append(eval.value)
    
    if not all_scores:
        return Evaluation(name="average_score", value=None, comment="No scores available")
    
    avg = sum(all_scores) / len(all_scores)
    return Evaluation(
        name="average_score",
        value=avg,
        comment=f"Average across {len(all_scores)} evaluations: {avg:.2%}"
    )

@pytest.fixture
def langfuse_client() -> Langfuse:
    return get_client()

def test_burger_agent_experiment(langfuse_client: Langfuse):
    result = langfuse_client.run_experiment(
        name="Burger Agent Test Suite",
        data=test_data,
        task=burger_agent_task,
        evaluators=[
            contains_expected_evaluator,
            price_accuracy_evaluator,
            tool_execution_evaluator,
        ],
        run_evaluators=[average_score_evaluator],
    )
    
    # Get the average score from run evaluators
    avg_score = None
    for eval in result.run_evaluations:
        if eval.name == "average_score" and eval.value is not None:
            avg_score = eval.value
            break
    
    print(f"\n{'='*60}")
    print("EXPERIMENT RESULTS")
    print(f"{'='*60}")
    print(f"Total test items: {len(test_data)}")
    print(f"Average score: {avg_score:.2%}" if avg_score else "Average score: N/A")
    print(f"{'='*60}\n")
    
    assert avg_score is not None, "No scores were generated"
    assert avg_score >= 0.7, f"Average score {avg_score:.2%} below threshold 70%"


def test_single_price_lookup(langfuse_client: Langfuse):
    single_test = [{"input": "How much is a Big Mac?", "expected_output": "$5"}]
    
    result = langfuse_client.run_experiment(
        name="Single Price Lookup Test",
        data=single_test,
        task=burger_agent_task,
        evaluators=[price_accuracy_evaluator],
    )
    
    for item in result.item_results:
        for eval in item.evaluations:
            if eval.name == "price_accuracy":
                assert eval.value == 1.0, f"Price accuracy test failed: {eval.comment}"

if __name__ == "__main__":
    print("\n" + "="*60)
    print("LANGFUSE NATIVE EVALUATION DEMO")
    print("="*60 + "\n")
    
    langfuse = get_client()
    print("Running experiment with Langfuse Experiment Runner...")
    print(f"Test cases: {len(test_data)}\n")
    
    result = langfuse.run_experiment(
        name="Burger Agent - Manual Run",
        data=test_data,
        task=burger_agent_task,
        evaluators=[
            contains_expected_evaluator,
            price_accuracy_evaluator,
            tool_execution_evaluator,
        ],
        run_evaluators=[average_score_evaluator],
    )
    
    print("\n" + "-"*60)
    print("INDIVIDUAL TEST RESULTS:")
    print("-"*60)
    
    for i, item_result in enumerate(result.item_results):
        test_item = test_data[i]
        print(f"\n📝 Test {i+1}: {test_item.get('test_name', test_item['input'][:40])}")
        print(f"   Input: {test_item['input']}")
        print(f"   Output: {item_result.output[:100] if item_result.output else 'None'}...")
        for eval in item_result.evaluations:
            emoji = "✅" if eval.value == 1.0 else "❌" if eval.value == 0.0 else "⚠️"
            print(f"   {emoji} {eval.name}: {eval.value} - {eval.comment}")
    
    print("\n" + "-"*60)
    print("AGGREGATE RESULTS:")
    print("-"*60)
    for eval in result.run_evaluations:
        print(f"   📊 {eval.name}: {eval.value:.2%} - {eval.comment}" if eval.value else f"   {eval.name}: N/A")
    
    print("\n" + "="*60)
    print("✨ Results are now visible in Langfuse Dashboard!")
    print("   Go to: Datasets → Experiments")
    print("="*60 + "\n")
    
    langfuse.flush()

