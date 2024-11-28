import dspy
import os

api_key = os.getenv('OPENAI_API_KEY')
lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key)
dspy.configure(lm=lm)

print("3. RAG")
def search_wikipedia(query: str) -> list[str]:
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

rag = dspy.ChainOfThought('context, question -> response')

question = "What's the name of the castle that David Gregory inherited?"
ragResponse = rag(context=search_wikipedia(question), question=question)
print(ragResponse)