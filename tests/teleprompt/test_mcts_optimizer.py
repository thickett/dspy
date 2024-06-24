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
    Example(input="Question: What is the color of the sky?", output="blue").with_inputs(
        "input"
    ),
    Example(
        input="Question: What does the fox say?",
        output="Ring-ding-ding-ding-dingeringeding!",
    ).with_inputs("input"),
]


class SimpleModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        # MCTS doesn't work with dspy.Predict
        self.predictor = dspy.ChainOfThought(signature)

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


def test_signature_optimizer_initialization():
    optimizer = MCTS(metric=simple_metric, breadth=2, depth=1, init_temperature=1.4)
    assert optimizer.metric == simple_metric, "Metric not correctly initialized"
    assert optimizer.breadth == 2, "Breadth not correctly initialized"
    assert optimizer.depth == 1, "Depth not correctly initialized"
    assert (
        optimizer.init_temperature == 1.4
    ), "Initial temperature not correctly initialized"


def test_mctc_tree_initialization():
    optimizer = MCTS(metric=simple_metric, breadth=2, depth=1, init_temperature=1.4)
    module = SimpleModule("input -> output")
    optimizer.create_tree(module, "Question: What is the color of the sky?")
    assert (
        optimizer.tree.root.instructions == "Question: What is the color of the sky?"
    ), "Root node instructions not correctly initialized"
    assert (
        len(optimizer.tree.root.children) == 1
    ), "Root node children not correctly initialized. should be breadth - 1."
    # add node to root
    optimizer.tree.add_node(optimizer.tree.root)
    assert (
        len(optimizer.tree.root.children) == 2
    ), f"Node not correctly added to root, got {len(optimizer.tree.root.children)} children but should have breadth -1 + 1"
    assert (
        optimizer.tree.root.children[0].parent == optimizer.tree.root
    ), "Parent not correctly set"
    assert isinstance(
        optimizer.tree.root.module, SimpleModule
    ), "Root node's module is not of type SimpleModule"
    for children in optimizer.tree.root.children:
        assert isinstance(
            children.module, SimpleModule
        ), "child node's module is not of type SimpleModule"


if __name__ == "__main__":
    test_signature_optimizer_initialization()
    test_mctc_tree_initialization()
    print("All tests passed!")
