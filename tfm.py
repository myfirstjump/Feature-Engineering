from transformers import pipeline

classifier = pipeline("sentiment-analysis")
res = classifier("If I am in your situation, the better way to figure this out is to do nothing.")
print(res)