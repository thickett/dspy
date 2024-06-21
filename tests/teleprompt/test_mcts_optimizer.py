import textwrap
import dspy
from dspy.teleprompt.mcts_optimizer import MCTS
from dspy.utils.dummies import DummyLM
from dspy import Example

# Define a simple metric function for testing
def simple_metric(example, prediction):
    # Simplified metric for testing: true if prediction matches expected output
    return example.output == prediction.output

trainset = [
    Example(input="Question: What is the color of the sky?", output="blue").with_inputs("input"),
    Example(input="Question: What does the fox say?", output="Ring-ding-ding-ding-dingeringeding!").with_inputs("input"),
]

def test_signature_optimizer_initialization():
    optimizer = MCTS(metric=simple_metric, breadth=2, depth=1, init_temperature=1.4)
    assert optimizer.metric == simple_metric, "Metric not correctly initialized"
    assert optimizer.breadth == 2, "Breadth not correctly initialized"
    assert optimizer.depth == 1, "Depth not correctly initialized"
    assert optimizer.init_temperature == 1.4, "Initial temperature not correctly initialized"
