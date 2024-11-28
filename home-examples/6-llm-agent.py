import dspy
import os

api_key = os.getenv('OPENAI_API_KEY')
lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key)
dspy.configure(lm=lm)

print("6. Agent")

def evaluate_math(expression: str) -> float:
    return dspy.PythonInterpreter({}).execute(expression)

def search_wikipedia(query: str) -> str:
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])

pred = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?")
print(pred.answer)