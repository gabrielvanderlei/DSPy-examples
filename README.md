# DSPy-examples

This repository contains practical examples demonstrating the usage of DSPy, Stanford's Declarative Self-improving Python framework for language models.

## What is DSPy?

DSPy is an open-source framework developed by [Stanford NLP](https://nlp.stanford.edu/) that enables:
- Programming (rather than prompting) language models
- Building modular AI systems
- Optimizing prompts and weights automatically
- Creating everything from simple classifiers to complex RAG pipelines and Agent loops

## Quick Start

```sh
pip install -U dspy
```

## Examples Overview

### 1. Basic LLM Interaction
Simple examples of direct interactions with language models:
```python
import dspy
import os

lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
dspy.configure(lm=lm)

# Direct prompt
response = lm("Say this is a test!", temperature=0.7)

# Chat format
chat_response = lm(messages=[{"role": "user", "content": "Say this is a test!"}])
```

### 2. Mathematical Reasoning
Demonstrates step-by-step problem solving:
```python
math = dspy.ChainOfThought("question -> answer: float")
result = math(question="Two dice are tossed. What is the probability that the sum equals two?")
```

### 3. RAG (Retrieval-Augmented Generation)
Shows how to combine search and generation:
```python
def search_wikipedia(query: str) -> list[str]:
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

rag = dspy.ChainOfThought('context, question -> response')
result = rag(
    context=search_wikipedia("What's the name of the castle that David Gregory inherited?"),
    question="What's the name of the castle that David Gregory inherited?"
)
```

### 4. Classification Tasks
Example of sentiment analysis with confidence scoring:
```python
class Classify(dspy.Signature):
    """Classify sentiment of a given sentence."""
    sentence: str = dspy.InputField()
    sentiment: Literal['positive', 'negative', 'neutral'] = dspy.OutputField()
    confidence: float = dspy.OutputField()

classify = dspy.Predict(Classify)
result = classify(sentence="This book was super fun to read, though not the last chapter.")
```

### 5. Information Extraction
Structured information extraction from text:
```python
class ExtractInfo(dspy.Signature):
    """Extract structured information from text."""
    text: str = dspy.InputField()
    title: str = dspy.OutputField()
    headings: list[str] = dspy.OutputField()
    entities: list[dict[str, str]] = dspy.OutputField(desc="entities and metadata")

module = dspy.Predict(ExtractInfo)
text = "Apple Inc. announced its latest iPhone 14 today. The CEO, Tim Cook, highlighted its new features."
result = module(text=text)
```

### 6. Agent Implementation
Combining tools with language model capabilities:
```python
def evaluate_math(expression: str) -> float:
    return dspy.PythonInterpreter({}).execute(expression)

def search_wikipedia(query: str) -> str:
    return dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)

react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])
result = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?")
```

## Key Concepts

### Signatures
Define input/output specifications for language models:
- Simple: `question -> answer`
- Complex: `context: list[str], question: str -> answer: str`

### Modules
Building blocks for LM-based programs:
- `Predict`: Direct LM responses
- `ChainOfThought`: Step-by-step reasoning
- `ProgramOfThought`: Code-based responses
- `ReAct`: Agent with tool usage
- `MultiChainComparison`: Output comparison and selection

## Resources
- [Official Documentation](https://dspy.ai/)
- [Stanford NLP Group](https://nlp.stanford.edu/)
