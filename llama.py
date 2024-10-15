from langchain_nvidia_ai_endpoints import ChatNVIDIA

client = ChatNVIDIA(
  model="meta/llama-3.2-1b-instruct",
  api_key="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", 
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
)

for chunk in client.stream([{"role":"user","content":k about the wonders of GPU computing."}]): 
  print(chunk.content, end="")

  
