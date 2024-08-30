---
layout: post
title: OpenAI Superalignment
permalink: /2024-03-24-superaligment-openai.html
---

* OpenAI team proposed and working on Superalignment problem. It could be solved in 10 years (by 2034)
* What *Alignment*?
  * The current AI systems are aligned with human response (feedback through RLHF)
* Why alignment will be hard?
  * The future (superhuman) machines will be so smart that human feedback will be challenging and sufficient.
  * Human will be not be able to understand.
  * *Example*: If the AI Systems generate millions of code, humans cann't possibly evaluate whether the code is safe or dangerous to execute.
  * *Question*: How can humans evaluate, teach/feedback and trust AI systems which are way smarter than them? (Superalignment)
* How will this be superalignment be solved?
  * No concrete idea or directions.
  * OpenAI have 10M USD grant for researchers and non-profits who work on this problem.
  * It is hard to predict because the super-intelligence doesn't exist.
* Possible directions:
  * Weak-to-strong generalization
    * Can we use weak models to supervise and evaluate the strong models?
    * OpenAI made some research into this area - They tried to GPT 2 teach GPT 4 to train a specific task which resulted in GPT 4 performing better than the GPT 2 model (weak teacher/supervisor).
    * The paper is [here](https://cdn.openai.com/papers/weak-to-strong-generalization.pdf)
  * Interpretability
    * Can we understand the model internals?
    * Can we de-cipher how did the model came up with an particular answer?
    * Similar to human lie-detector
  * Scalable oversight
    * Can AI systems assist humans to evaluate the output of other AI systems?    
  * Honesty
  * CoT faithfulness
  * Adversarial robustness
* Reference
  * [open ai blog](https://openai.com/blog/superalignment-fast-grants)
  * [mit tech review](https://www.technologyreview.com/2023/12/14/1085344/openai-super-alignment-rogue-agi-gpt-4/)
  *  
