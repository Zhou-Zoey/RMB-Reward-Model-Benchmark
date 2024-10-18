import random
from typing import List

import torch


class ArmoRMPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        random.seed(0)

    def __call__(self, candidates_1: List[str], candidates_2: List[str], candidates_3: List[str], candidates_4: List[str], candidates_5: List[str], **kwargs):
        """
        samples: List[str]
        """
        device = self.model.device
        out = []
        all_pair_score = []
        with torch.no_grad():
            for candidate_1, candidate_2, candidate_3, candidate_4, candidate_5 in zip(candidates_1, candidates_2, candidates_3, candidates_4, candidates_5):
                pair_scores = []
                for candidate in [candidate_1, candidate_2, candidate_3, candidate_4, candidate_5]:
                    input_ids = self.tokenizer.apply_chat_template(candidate, return_tensors="pt").to(device)
                    output = self.model(input_ids)
                    # .score.cpu().float()
                    score = output.score.float().item()
                    pair_scores.append(float(score))
                all_pair_score.append(pair_scores)
                if pair_scores[0] == pair_scores[1]:
                    out.append(random.choice([True, False]))
                else:
                    out.append(pair_scores[0] > pair_scores[1])
        # return torch.Tensor(out).bool()
        # print(pair_scores)
        return all_pair_score
