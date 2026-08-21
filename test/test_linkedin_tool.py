import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tools.linkedin_search_tool import search_linkedin_jobs

result = search_linkedin_jobs.invoke({
    "query": "Machine Learning Engineer ",
    "location": "germany",
    "results_count": 3,
})

print(result)