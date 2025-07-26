# Solution Branch Overview

The `solution` branch is dedicated to reproducing the OpenR method used in the ablation studies of our paper. OpenR is a solution-level approach that integrates **Monte Carlo Tree Search (MCTS)** with **self-refinement**.

Compared to the `main` branch, the key difference lies in the modified `import` statements in the following two files, which directly reuse components from the [OpenR](https://github.com/openreasoner/openr/tree/critic_mcts) framework:

```python
# File: /envs/critic_MATH/__init__.py
# Original: from .step_critic_math import Env
Modified: from .solution_critic_math import Env

# File: /reason/evaluation/methods.py
# Original: from reason.guided_search.step_critic_mcts import CriticSearchTree
Modified: from reason.guided_search.solution_critic_mcts import CriticSearchTree
```