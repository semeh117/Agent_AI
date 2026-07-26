# test/test_job_parser_live.py
from dotenv import load_dotenv
load_dotenv()
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from job_parser import extract_job_requirements

title = "Backend Developer"
description = """This is a remote position. We are seeking a talented and motivated 
Backend Engineer to join our client dynamic team. As a Backend Engineer, you will be 
responsible for developing and maintaining our backend services, ensuring they are 
scalable, robust, and efficient. You will work closely with our client's team to deliver 
high-quality solutions that meet our client users' needs. Key Responsibilities Develop, 
maintain, and optimize backend services and APIs. Collaborate with frontend-developers 
to integrate user-facing elements with server-side logic. Ensure the scalability and 
performance of applications. Write clean, maintainable, and efficient code. Troubleshoot 
and debug applications. Participate in code reviews and contribute to a high-performing 
team culture. Implement security and data protection measures. Stay up-to-date with 
emerging technologies and industry trends Requirements Proven experience as a Backend 
Engineer or similar role. Proficiency in Go is preferred; experience with PHP, Python, 
or other languages is acceptable depending on the candidate. Experience with AWS 
(Amazon Web Services) is preferred. Strong understanding of RESTful APIs and web 
services. Familiarity with database technologies such as SQL, NoSQL, and in-memory 
databases. Knowledge of version control systems (e.g., Git). Excellent problem-solving 
skills and attention to detail. Strong communication skills and fluency in English."""

result = extract_job_requirements(title, description)
print(result.model_dump_json(indent=2))