# test/test_job_search.py
from dotenv import load_dotenv
load_dotenv()
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.job_search import search_real_jobs

result = search_real_jobs.invoke({
    "query": "ai  Engineer  ",
    "results_count": 3
})
print(result)
