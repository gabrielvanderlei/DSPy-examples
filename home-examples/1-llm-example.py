import dspy
import os

api_key = os.getenv('OPENAI_API_KEY')
lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key)
dspy.configure(lm=lm)

print("1.1. Simple test")
example = lm("Say this is a test!", temperature=0.7)  # => ['This is a test!']
print(example)

print("1.2. Simple conversation test")
exampleConversation = lm(messages=[{"role": "user", "content": "Say this is a test!"}])  # => ['This is a test!']
print(exampleConversation)
