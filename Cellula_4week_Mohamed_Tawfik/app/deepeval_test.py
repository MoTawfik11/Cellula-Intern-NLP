import json
from deepeval import EvaluationDataset
from deepeval.models import DeepEvalOpenAI
from deepeval.metrics import FaithfulnessMetric
from deepeval.evaluate import evaluate

def run_deepeval(rag_chain, retriever):
    data = [
        {"query": "What is RAG?", "expected_output": "RAG combines retrieval and generation."}
    ]
    evaluator = DeepEvalOpenAI(model="gpt-4-turbo")
    metric = FaithfulnessMetric(model=evaluator)
    results = []
    for d in data:
        q = d["query"]
        ans = rag_chain.run(q)
        context = retriever.get_relevant_documents(q)
        results.append({"input": q, "actual_output": ans, "expected_output": d["expected_output"], "context": context})
    evaluation = evaluate(results, [metric])
    print(json.dumps(evaluation.to_dict(), indent=2))
