# Whole-model GPTQ fixed prompts

The 13 calibration prompts are synthetic declarations, not a training dataset.
The workflow observes actual projection activations for each prompt and starts
a fresh cache each time. The first three prompts (21 token rows) were also used
by the earlier single-projection test and a failed initial whole-model run.

The initial whole-model Q4_0 run exceeded the unchanged 0.5-nat KL gate on the
first development-validation prompt (0.5525935556259604). Before the expanded
calibration run, ten calibration prompts and two additional validation prompts
were declared. The first two validation lines are development samples; the
last two were newly declared before that expanded run. No validation line is
included in calibration. These are a bounded regression set, not an unbiased
benchmark or evidence of general task accuracy.

The real-model workflow keeps maximum KL 0.5 and minimum Top-10 overlap 6 for
each of the four validation inputs. It records exact dataset hashes, every
projection's calibration row count and source/packed hashes, and verifies
published-model parity. Numeric quality and container parity are separate gates.
