from textattack.attack_results import AttackResult
from textattack.shared import utils as textattack_utils

class TimedoutAttackResult(AttackResult):
    def __init__(self, original_result, perturbed_result=None):
        perturbed_result = perturbed_result or original_result
        super().__init__(original_result, perturbed_result)

    def str_lines(self, color_method=None):
        lines = (
            self.goal_function_result_str(color_method),
            self.original_text(color_method),
        )
        return tuple(map(str, lines))

    def goal_function_result_str(self, color_method=None):
        timedout_str = textattack_utils.color_text("[TIMEDOUT]", "gray", color_method)
        return (
            self.original_result.get_colored_output(color_method)
            + " --> "
            + timedout_str
        )