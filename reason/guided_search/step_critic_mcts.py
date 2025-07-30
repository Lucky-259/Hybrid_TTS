"""
Critic-guided MCTS Reasoning
"""

from typing import List, Dict, Optional, Callable, Type
from distributed.utils import print_rank_0, print_with_rank
from envs.base_env import CoTEnv
from .tree import Node, LanguageNode, SearchTree

class CriticLanguageNode(LanguageNode):
    def __init__(self, from_review: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.from_review = from_review


class CriticSearchTree(SearchTree):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.show_tree_expansion = False

    def critic_mcts(
        self,
        simulate_env: Type[CoTEnv],
        num_path: int,
        reward_model_fn: Optional[Callable] = None,
        select_by_prior: bool = False,
    ) -> List[Dict]:

        api_call_completion_tokens = 0
        _, info = simulate_env.reset(update_legal_action=True)
        api_call_completion_tokens += info["api_completion_token"]
        if self.root is None:
            root = CriticLanguageNode(text_state=simulate_env.get_state())
            self._expand_leaf_node(root, simulate_env, reward_model_fn)
            self.root = root

        traj_list = []
        for i_path in range(num_path):
            node = self.root
            env_copy = simulate_env.copy()
            done = False
            review_traj = []
            rewrite_traj = []
            prm_traj = [] 
            dept = 1
            path_tokens = 0
            refine_info_list=[]
            while not done:
                parent = node
                if node.visit_count > 0: # root.visit_count=0
                    action, node = self._select_child(node, env_copy)
                else:
                    if select_by_prior: # False
                        action, node = self._select_by_prior(node, env_copy)
                    else:
                        action, node = self._select_child(node, env_copy)
                
                env_copy._next_state_terminated = {}
                assert node.last_action == action
                env_copy._next_state_terminated[action] = node.terminated

                review_traj.append(node.from_review)
                rewrite_traj.append(node.last_action)
                prm_traj.append(node.prm_value)
                step_rewrite_list, step_prm_list, refine_tokens, rewrite_text, _, terminated, truncated, info = env_copy.step(
                    action,node._initial_value,reward_model_fn, update_legal_action=node.is_leaf()
                )
                parent.children[rewrite_text] = parent.children.pop(action)
                
                refine_info_list.append(step_rewrite_list)
                refine_info_list.append(step_prm_list)

                node.rewrite_action(rewrite_text,refine_tokens) 
                path_tokens += node.return_tokens()
                done = terminated or truncated
                if not done and node.is_leaf():
                    self._expand_leaf_node(node, env_copy, reward_model_fn)
                api_call_completion_tokens += info["api_completion_token"]
                dept +=1
            else:
                if node.visit_count > 0:
                    leaf_value = node.value
                else:
                    if self._init_critic_value:
                        leaf_value = node._initial_value
                    else:
                        leaf_value = reward_model_fn(env_copy.get_state()).item()
            node.update_recursive(leaf_value, env_copy.mcts_mode)
            print(f"'final answer' : {env_copy.get_lastaction_str()+rewrite_text}")
            traj_data = {
                "path_idx": i_path,
                #"text": action,
                "text": env_copy.get_lastaction_str()+rewrite_text,       # final answer
                "path_tokens": path_tokens,
                "value": leaf_value,
                "api_completion_tokens": api_call_completion_tokens,
                "tree_completion_tokens": self._completion_tokens,
                "review_path": review_traj,
                "rewrite_path": rewrite_traj,
                "stepprm_path": prm_traj,
                "refine_info_list": refine_info_list
            }

            traj_list.append(traj_data)

            # reset api_call_completion_tokens
            api_call_completion_tokens = 0

        return traj_list

    def _expand_leaf_node(
            self,
            node: Node,
            simulate_env: Type[CoTEnv],
            reward_fn: Optional[Callable] = None,
    ) -> float:
        """
        Overview:
            expand the node with the reward_fn.
        Arguments:
            - node (:obj:`Class Node`): current node when performing mcts search.
            - simulate_env (:obj:`Class BaseGameEnv`): the class of simulate env.
            - reward_fn (:obj:`Function`): the Callable to compute the state value.
        Returns:
            - leaf_value (:obj:`Bool`): the leaf node's value.
        """
        """
        action_probs_dict, leaf_value = reward_fn(simulate_env)
        for action, prior_p in action_probs_dict.items():
            if action in simulate_env.legal_actions:
                node.children[action] = Node(parent=node, prior_p=prior_p)
        """

        text_state = simulate_env.get_state()
        if not self._init_critic_value:
            leaf_value = reward_fn(text_state)

        else:
            leaf_value = node._initial_value 
            assert len(simulate_env.legal_actions) > 0
            last_str = simulate_env.get_allaction_str()
            i_try=0
            

            prms = []
            for x in simulate_env.legal_actions:
                reward = reward_fn([
                    (simulate_env.question + last_str, simulate_env.answer + x["action"])
                ])
                prms.extend(reward)
            
            
            child_values = []

            # get the last reward
            last_r = [i[-1] for i in prms]

            # PRM get last r as single reward
            for act, rs in zip(simulate_env.legal_actions, last_r):
                    # prm-last
                    child_values.append(rs)
                    # # prm-min
                    # child_values.append(min(rs))
                    # # prob-prm
                    # child_values.append(act['prob'])

        assert len(node.children) == 0

        for i, action_dict in enumerate(simulate_env.legal_actions):
            action, prob = action_dict["action"], action_dict["prob"]

            if self._init_critic_value:
                child_value = child_values[i]
            else:

                child_value = 0.0

            node.children[action] = CriticLanguageNode(
                parent=node,
                prior_p=prob,
                prm_value=prms[i],
                text_state=text_state,
                last_action=action,
                initial_value=child_value,
                num_generated_token=action_dict["num_token"],
                from_review=action_dict['from_review']
            )
            # set terminal node here

            if simulate_env._next_state_terminated[action]:
                node.children[action].set_as_terminate_node()
        if len(node.children) == 0:
            print_rank_0(
                "Prune all current children at node {}".format(node.last_action)
            )

        # collect tokens
        if not node.has_collected_token_num:
            self._completion_tokens += sum(
                c.num_generated_token for c in node.children.values()
            )
            node.has_collected_token_num = True
        else:
            raise RuntimeError("Token number has been collected again.")

        return leaf_value

    def draw_tree(self):

        def display_tree(node):
            print("|" + "-" * (node.depth * 4) + str(node))
            for child in node.children:
                display_tree(child)

        print(f"\n---------Expanded Tree---------")
        display_tree(self.root)