"""Assistant-authored provisional labels; require independent domain review.

Labels measure evidence of individual skills, not hireability. Synthetic
profiles broaden regression coverage without representing real-world accuracy.
"""
CAPTURED_LABELS = {
    "4457797982": {
        "positive": ["Python", "API development", "LangChain", "embeddings", "vector databases", "retrieval-augmented generation", "prompt engineering"],
        "negative": ["Claude", "GitHub Copilot", "Codex", "Claude Code", "ADK", "Microsoft Copilot", "AWS", "GCP"]},
    "4448211255": {
        "positive": ["Python", "Git", "Docker", "LangChain", "LlamaIndex", "RAG", "vector search", "embeddings", "chunking", "reranking", "vector databases", "FastAPI", "SQL"],
        "negative": ["pytest", "LangGraph", "Pydantic", "pgvector", "Pinecone", "Weaviate", "OpenSearch", "Snowflake vector functions", "Azure", "AWS", "GCP"]},
    "4456063178": {
        "positive": ["Python", "Git", "Docker", "LangChain", "LlamaIndex", "RAG", "vector search", "embeddings", "chunking", "reranking", "FastAPI", "SQL"],
        "negative": ["pytest", "LangGraph", "Pydantic", "Pinecone", "Weaviate", "Flask", "Django", "Azure", "AWS", "GCP"]},
    "4349250403": {
        "positive": ["Python", "RAG patterns", "LangChain", "machine learning models and algorithms"],
        "negative": ["ReAct", "knowledge graphs"]},
    "4354750897": {
        "positive": ["RAG techniques", "RAG architectures", "embeddings", "containerized deployments", "PyMuPDF", "Python development", "prompt engineering"],
        "negative": ["Kubernetes", "Tesseract OCR", "Pillow", "BM25", "nDCG", "MRR", "Cognitive Services"]},
}
SYNTHETIC_PROFILES = [
    {"id": "junior-backend", "skills": ["Python", "FastAPI", "PostgreSQL", "Git", "Docker"],
     "positive": ["Python", "REST APIs", "PostgreSQL", "Git", "Docker", "SQL"],
     "negative": ["Java", "Kubernetes", "React", "MongoDB", "AWS ECS", "Django"]},
    {"id": "senior-frontend", "skills": ["JavaScript", "TypeScript", "React", "CSS", "HTML", "Playwright"],
     "positive": ["JavaScript", "TypeScript", "React", "CSS", "HTML", "Playwright"],
     "negative": ["Java", "Angular", "Vue", "Python", "Selenium", "React Native"]},
    {"id": "data-analyst", "skills": ["SQL", "Excel", "Power BI", "Python", "Pandas", "statistics"],
     "positive": ["SQL", "Excel", "Power BI", "Python", "Pandas", "statistics"],
     "negative": ["Tableau", "Spark", "Airflow", "Kubernetes", "PyTorch", "Scala"]},
    {"id": "cloud-engineer", "skills": ["AWS ECS", "Docker", "Terraform", "Linux", "Bash", "Git"],
     "positive": ["AWS ECS", "Docker", "Terraform", "Linux", "Bash", "Git"],
     "negative": ["AWS Bedrock", "Azure", "GCP", "Ansible", "Kubernetes", "PowerShell"]},
    {"id": "qa-engineer", "skills": ["Python", "pytest", "Selenium", "regression testing", "Git", "Jenkins"],
     "positive": ["Python", "pytest", "Selenium", "regression testing", "Git", "Jenkins"],
     "negative": ["regression", "PyTorch", "Playwright", "Cypress", "Java", "Docker"]},
]
