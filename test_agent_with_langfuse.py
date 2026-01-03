import os
import warnings
import pytest
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.models import GeminiModel
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langfuse import Langfuse

warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.genai")
warnings.filterwarnings("ignore", message=".*Module already imported.*")
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

os.environ["GOOGLE_API_KEY"] = gemini_api_key

from burger_agent import run_agent

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "test"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", "test"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
)

evaluation_model = GeminiModel(
    model="gemini-2.0-flash",
    api_key=gemini_api_key,
    temperature=0
)


class TestBurgerAgentWithLangfuse:
    def test_rag_and_tool(self):
        user_input = "How much is a Big Mac and please order one."
        actual_output, trace_id = run_agent(user_input)
        
        expected_output = (
            "The agent should retrieve that a Big Mac costs $5 from the menu database, "
            "inform the customer of the price, and place an order for a Big Mac using "
            "the place_order tool, confirming the order was placed."
        )
        
        test_case = LLMTestCase(
            input=user_input,
            actual_output=actual_output,
            expected_output=expected_output,
        )
        
        correctness_metric = GEval(
            name="Correctness",
            model=evaluation_model,
            criteria=(
                "Determine if the actual output demonstrates that: "
                "1) The correct price for Big Mac ($5) was retrieved and mentioned, AND "
                "2) An order was placed (indicated by ORDER_PLACED confirmation or acknowledgment of placing the order). "
                "The response should show both the price lookup and order placement occurred."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=0.7,
        )
        
        try:
            assert_test(test_case, [correctness_metric])
            score = 1.0
            passed = True
        except AssertionError:
            score_result = correctness_metric.measure(test_case)
            score = score_result.score if hasattr(score_result, 'score') else 0.0
            passed = score >= correctness_metric.threshold
        
        if trace_id:
            langfuse.create_score(
                name="RAG and Tool Correctness",
                value=score,
                trace_id=trace_id,
                data_type="NUMERIC",
                comment=f"Test: {user_input[:50]}...",
            )
            langfuse.flush()
        
        assert passed, f"Test failed with score: {score}"
    
    def test_price_lookup_only(self):
        user_input = "What's the price of Fries?"
        actual_output, trace_id = run_agent(user_input)
        
        expected_output = (
            "The agent should retrieve that Fries cost $2 from the menu database "
            "and inform the customer of the price."
        )
        
        test_case = LLMTestCase(
            input=user_input,
            actual_output=actual_output,
            expected_output=expected_output,
        )
        
        correctness_metric = GEval(
            name="Price Retrieval Correctness",
            model=evaluation_model,
            criteria=(
                "Determine if the actual output correctly states that Fries cost $2. "
                "The price should be accurate based on the menu lookup."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=0.7,
        )
        
        try:
            assert_test(test_case, [correctness_metric])
            score = 1.0
            passed = True
        except AssertionError:
            score_result = correctness_metric.measure(test_case)
            score = score_result.score if hasattr(score_result, 'score') else 0.0
            passed = score >= correctness_metric.threshold
        
        if trace_id:
            langfuse.create_score(
                name="Price Lookup Correctness",
                value=score,
                trace_id=trace_id,
                data_type="NUMERIC",
                comment=f"Test: {user_input}",
            )
            langfuse.flush()
        assert passed, f"Test failed with score: {score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

