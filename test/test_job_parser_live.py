# test/test_job_parser_live.py
from dotenv import load_dotenv
load_dotenv()
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from job_parser import extract_job_requirements

title = "DATA SCIENTIST / MACHINE LEARNING ENGINEER "
description = """

About the Opportunity
Our client is seeking passionate and ambitious Data Science Interns to join their remote team across the European Union. This internship is designed for students, recent graduates, and aspiring data professionals looking to gain practical experience by working on real-world datasets and industry-focused projects.
If you're eager to strengthen your technical skills, collaborate with experienced professionals, and build a strong portfolio, this opportunity is for you.
Key Responsibilities
Collect, clean, and preprocess structured and unstructured datasets.
Perform exploratory data analysis (EDA) to uncover trends and insights.
Develop and evaluate machine learning models using Python.
Work with tools and libraries such as Pandas, NumPy, Scikit-learn, Matplotlib, and Jupyter Notebook.
Create meaningful data visualizations and reports.
Assist in solving real-world business problems using data-driven approaches.
Collaborate with team members on live projects.
Present findings and recommendations in a clear and professional manner.
Continuously learn and apply new data science techniques and technologies.


Requirements
Currently pursuing or recently completed a Bachelor's or Master's degree in Data Science, Computer Science, Artificial Intelligence, Mathematics, Statistics, or a related field.
Basic knowledge of Python programming.
Understanding of statistics, probability, and machine learning fundamentals.
Familiarity with SQL and data analysis concepts.
Experience with Pandas, NumPy, or Scikit-learn is an advantage.
Strong analytical and problem-solving skills.
Good written and verbal communication skills.
Ability to work independently in a remote, multicultural environment.
Self-motivated with a willingness to learn and grow.
"""
result = extract_job_requirements(title, description)
print(result.model_dump_json(indent=2))