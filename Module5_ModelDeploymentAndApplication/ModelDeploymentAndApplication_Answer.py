"""
Module 5 - Model Deployment and Application (LLM Deployment)
Question
"""

# ============================================================
# Question: When deploying the LLM, which parameters have the
# greatest impact on the model output?
# ============================================================
"""
The --temp (temperature) parameter has the greatest impact on the model's
output. Temperature controls how the model selects the next word from its
probability distribution over possible next tokens.

- Low temperature (e.g., 0.1-0.3): the model almost always picks the highest
  probability word, producing consistent but sometimes repetitive or overly
  "safe" text.
- High temperature (e.g., 1.0-2.0): the model is more willing to pick
  lower-probability words, producing more creative/varied text, but with a
  higher risk of incoherent or "hallucinated" output.
- Temperature 0.6 (used in this experiment, as recommended by DeepSeek):
  provides a balance between coherent and slightly varied responses.

Other parameters like -c (context length) mainly control how much text the
model can process, and --top-k works together with temperature to limit
the candidate pool of next words - but they have a smaller direct effect
on the overall tone/quality of the generated text compared to temperature,
which directly reshapes the entire probability distribution used for
word selection.
"""

