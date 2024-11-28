import dspy
import os

api_key = os.getenv('OPENAI_API_KEY')
lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key)
dspy.configure(lm=lm)

print("2. Math")
math = dspy.ChainOfThought("question -> answer: float")
mathResponse = math(question="Two dice are tossed. What is the probability that the sum equals two?")
print(mathResponse)
print(mathResponse.answer)