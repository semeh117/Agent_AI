# test_job_search.py
from dotenv import load_dotenv
load_dotenv()
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from job_search import search_real_jobs

result = search_real_jobs.invoke({
    "query": "Data Scientist Machine Learning",
    "results_count": 5
})
print(result)