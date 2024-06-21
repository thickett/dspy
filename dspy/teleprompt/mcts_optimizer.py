from collections import defaultdict
import random

import dsp
import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.signatures import Signature
from dspy.teleprompt.teleprompt import Teleprompter

# mutated prompts taken from DeepMind's paper, PROMPTBREEDER
# https://arxiv.org/abs/2309.16797
mutated_prompts = [
    'Modify the following instruction creatively, giving some advice on how to solve\nit:',
    'Just change this instruction to make it more fun, think WELL outside the box:',
    'Don’t think about the instruction at all, but let it inspire you to do something\nrelated. Talk about what that might be'
]


class BasicGenerateInstruction(Signature):

    """You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English. Your task is to propose an instruction that will lead a good language model to perform the task well. Don't be afraid to be creative."""

    basic_instruction = dspy.InputField(desc="The initial instructions before optimization")
    proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
    proposed_prefix_for_output_field = dspy.OutputField(
        desc="The string at the end of the prompt, which will help the model start solving the task",
    )

class GenerateInstructionGivenAttempts(dspy.Signature):
    """You are an instruction optimizer for large language models. I will give some task instructions I've tried, along with their corresponding validation scores. The instructions are arranged in increasing order based on their scores, where higher scores indicate better quality.

    Your task is to propose a new instruction that will lead a good language model to perform the task even better. Don't be afraid to be creative."""

    attempted_instruction = dspy.InputField(desc="The previous instruction used.")
    Instructions_feedback = dspy.InputField(desc="The validation score and reasoning given for the attempted instructions")
    proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
    proposed_prefix_for_output_field = dspy.OutputField(
        desc="The string at the end of the prompt, which will help the model start solving the task",
    )

class MCTS(Teleprompter):
    def __init__(
        self,
        prompt_model=None,
        prune_rate=3,
        metric=None,
        breadth=10,
        depth=3,
        init_temperature=1.4,
        track_stats=False,
        **_kwargs
    ):
        # if breadth==1 we simply aim to optimise the initital prompt.
        if breadth <1:
            raise ValueError("breadth must be greater than 0")
        self.prompt_model = prompt_model
        self.prune_rate = prune_rate
        self.metric = metric
        self.breadth = breadth
        self.depth = depth
        self.init_temperature = init_temperature
        self.track_stats = track_stats



