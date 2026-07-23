from dotenv import load_dotenv
load_dotenv()
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from job_parser import extract_job_requirements

title = "Principal Data Scientist & Machine Learning Researcher"
desc = ("Raytheon UK has a perm opportunity for a Principal Data Scientist "
        "and Machine Learning Researcher to join our Strategic Research "
        "Group (SRG). As a Principal Data Scientist and Machine Learning "
        "Researcher, you will work with high levels of autonomy and be "
        "responsible for the des…")

result = extract_job_requirements(title, desc)
print(result.model_dump_json(indent=2))
