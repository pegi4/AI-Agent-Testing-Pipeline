import os
import warnings
import pytest
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.genai")
warnings.filterwarnings("ignore", message=".*Module already imported.*")
from deepeval import assert_test
from deepeval.models import GeminiModel
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
os.environ["GOOGLE_API_KEY"] = gemini_api_key

from burger_agent import run_agent

evaluation_model = GeminiModel(
    model="gemini-2.0-flash",
    api_key=gemini_api_key,
    temperature=0
)


class TestBurgerAgent:
    """Test suite for the Burger Shop Agent."""
    
    def test_rag_and_tool(self):
        """
        Test that the agent correctly:
        1. Retrieves the Big Mac price ($5) from the RAG system
        2. Executes the place_order tool to place an order
        """
        user_input = "How much is a Big Mac and please order one."
        
        actual_output, _ = run_agent(user_input)
        
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
        
        assert_test(test_case, [correctness_metric])
    
    def test_price_lookup_only(self):
        """
        Test that the agent correctly retrieves prices without placing an order.
        """
        user_input = "What's the price of Fries?"
        
        actual_output, _ = run_agent(user_input)
        
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
        
        assert_test(test_case, [correctness_metric])
    
    def test_order_placement(self):
        """
        Test that the agent can place an order for multiple items.
        """
        user_input = "I'd like to order a Whopper and Fries please."
        
        actual_output, _ = run_agent(user_input)
        
        expected_output = (
            "The agent should place an order for a Whopper and Fries, "
            "confirming the order with ORDER_PLACED or similar confirmation."
        )
        
        test_case = LLMTestCase(
            input=user_input,
            actual_output=actual_output,
            expected_output=expected_output,
        )
        
        correctness_metric = GEval(
            name="Order Placement Correctness",
            model=evaluation_model,
            criteria=(
                "Determine if the actual output confirms that an order was placed "
                "for a Whopper and Fries. Look for ORDER_PLACED confirmation or "
                "acknowledgment that the order has been submitted."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=0.7,
        )
        
        assert_test(test_case, [correctness_metric])
    
    def test_friendly_tone(self):
        """
        Test that the agent maintains a friendly and helpful tone.
        """
        user_input = "Hi! What can you help me with?"
        
        actual_output, _ = run_agent(user_input)
        
        expected_output = (
            "The agent should respond in a friendly, welcoming manner, "
            "explaining that it can help with menu information and placing orders."
        )
        
        test_case = LLMTestCase(
            input=user_input,
            actual_output=actual_output,
            expected_output=expected_output,
        )
        
        tone_metric = GEval(
            name="Friendly Tone",
            model=evaluation_model,
            criteria=(
                "Evaluate if the actual output is friendly, welcoming, and helpful. "
                "The response should be warm and customer-service oriented, "
                "making the customer feel welcome at the burger shop."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=0.7,
        )
        
        assert_test(test_case, [tone_metric])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

