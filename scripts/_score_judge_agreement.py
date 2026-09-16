"""
One-shot script: applies hand (human) scores against the same 35-example
sample the LLM judge already scored, computes Cohen's kappa per dimension,
writes golden_set/human_judge_agreement.csv. Human scores below were
assigned by reading each (tweet, reply) pair directly - not by re-running
the judge model - to give an independent check on eval/judge.py.
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from eval.metrics import judge_human_agreement

# index (0-34, order of golden_set/_judge_sample.jsonl) -> (grounded, tone, factual, actionable)
HUMAN = {
0: (3,4,4,3), 1: (2,3,2,2), 2: (2,3,2,2), 3: (2,3,3,2), 4: (3,4,4,3),
5: (2,3,2,3), 6: (4,5,4,4), 7: (3,4,3,3), 8: (3,4,3,2), 9: (3,4,4,4),
10: (2,4,2,2), 11: (3,4,3,3), 12: (3,4,3,3), 13: (2,3,2,2), 14: (3,4,3,3),
15: (3,4,3,3), 16: (3,4,3,4), 17: (3,5,3,4), 18: (2,4,3,3), 19: (3,4,3,4),
20: (3,4,3,3), 21: (3,4,3,4), 22: (2,4,3,4), 23: (2,4,3,3), 24: (3,4,4,4),
25: (2,4,2,2), 26: (2,4,3,2), 27: (2,4,2,2), 28: (3,4,3,4), 29: (3,4,4,4),
30: (2,4,3,3), 31: (2,4,3,3), 32: (2,4,3,2), 33: (3,4,3,3), 34: (1,3,1,1),
}
DIMS = ("grounded", "tone", "factual", "actionable")


def main():
    sample = pd.read_json("golden_set/_judge_sample.jsonl", lines=True)
    assert len(sample) == len(HUMAN)

    human_scores, judge_scores = [], []
    rows = []
    for i, r in sample.iterrows():
        h = dict(zip(DIMS, HUMAN[i]))
        j = r["sys_judge"]
        human_scores.append(h)
        judge_scores.append(j)
        rows.append({
            "thread_id": r["thread_id"], "text": r["text"], "sys_reply": r["sys_reply"],
            **{f"human_{d}": h[d] for d in DIMS},
            **{f"judge_{d}": j[d] for d in DIMS},
        })

    kappa = judge_human_agreement(human_scores, judge_scores)
    print("Cohen's kappa (linear-weighted) per dimension:", kappa)

    out = pd.DataFrame(rows)
    out.to_csv("golden_set/human_judge_agreement.csv", index=False)
    with open("outputs/judge_agreement.json", "w") as f:
        json.dump(kappa, f, indent=2)
    print("wrote golden_set/human_judge_agreement.csv, outputs/judge_agreement.json")


if __name__ == "__main__":
    main()
