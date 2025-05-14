from graphrag import GraphRag

gr = GraphRag(config="settings.yaml", echo_prompt=True, log_level="DEBUG")

questions = ["熵的基本性质包括非负性，即对于任意随机变量X，其熵H(X)满足H(X)≥0。请用数学表达式说明这一性质，并给出证明过程。"]
for q in questions:
    print("\n=== 问题:", q)
    subgraph = gr.retrieve_subgraph(q, top_k=5)
    print(">> 检索到的 triples:", subgraph)
    answer = gr.chat(q)
    print(">> 模型回答:", answer)
