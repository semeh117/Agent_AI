# test/test_matcher.py
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from cv_parser import extract_text_from_pdf, extract_cv_info
from job_parser import extract_job_requirements
from matcher import calculate_compatibility

# --- Pick a CV ---
CV_FOLDER = "cv"
pdf_files = sorted(Path(CV_FOLDER).glob("*.pdf"))
print("Available CVs:")
for i, path in enumerate(pdf_files, start=1):
    print(f"  {i}. {path.name}")
choice = input("\nWhich CV? (enter number): ").strip()
selected_path = pdf_files[int(choice) - 1]

print(f"\nExtracting CV info from '{selected_path.name}'...")
raw_text = extract_text_from_pdf(str(selected_path))
cv_info = extract_cv_info(raw_text)
print(f"  -> {cv_info.full_name}, {len(cv_info.skills)} skills, "
      f"{cv_info.experience_years} yrs exp, {cv_info.highest_education_level}\n")

# --- Real job posting (paste in whichever one you want to test against) ---
job_title = "AI/ML Engineer"
job_description = """Are you interested in building an exciting IT career at Dev.Pro ? Join our exclusive screening process to gain valuable career insights and access personalized feedback on your skill set. 🟩 What you’ll get by applying: Personalized career growth plan tailored to your aspirations Expert feedback to elevate your skills for lasting success Priority consideration by Dev.Pro for suitable job openings within the company Opportunity to work with top global corporations and participate in industry-shaping projects Give us a chance to get to know you better so that when the right position becomes available, you’ll be the first one we reach out to! ✅ Is that you? 4+ years’ of experience as a DevOps Engineer Experience with highly available and reliable Cloud platforms Hand-on experience with Git, CI/CD tools, Docker, Kubernetes, Terraform, Monitoring (New Relic, Datadog, etc.) Experience with Azure/AWS/GCP Experience of utilizing logging, monitoring, and alerting solutions, and/or similar solutions as a system administrator Expertise in troubleshooting, system/network administration Strong understanding of modern operating environments and disciplines: firewalls, proxies, certificates, network, security, etc. Upper Intermediate English level Get started now: 1. Send us your CV in English. 2. Schedule a 15-30 minute call with Dev.Pro recruiters. 3. Participate in a 30-minute experience interview focused on testing your soft skills. 4. Get a 1-hour online evaluation of your technical skills by our technical experts. In under 2 hours, you'll know where you stand in your career and be ready for action when Dev.Pro presents you with an exciting job offer! 🎾 About Dev.Pro Dev.Pro is a US-based outsource company with an ambitious and creative mindset that has been delivering superior software products since 2011. Known for its strong human focus, Dev.Pro promotes a work environment that is fair, inclusive, open-minded, and friendly toward people of every race, religion, gender, cultural background, marital/parental status, etc. By joining Dev.Pro , you'll feel what it's like to grow with professionals who support your journey"""
print("Extracting job requirements...")
job_req = extract_job_requirements(job_title, job_description)
print(f"  -> {len(job_req.required_skills)} required skills, "
      f"{job_req.required_experience_years} yrs required, "
      f"{job_req.required_education_level} required\n")

print("Calculating compatibility...")
result = calculate_compatibility(cv_info, job_req)

print("\n" + "=" * 60)
print(f"COMPATIBILITY SCORE: {result['score_percent']}%")
print("=" * 60)
print(f"\nSkills match: {result['skills']['score']*100:.1f}%")
print(f"  Matching: {result['skills']['matching']}")
print(f"  Missing:  {result['skills']['missing']}")
print(f"\nExperience match: {result['experience']['score']*100:.1f}%")
print(f"  {result['experience']}")
print(f"\nEducation match: {result['education']['score']*100:.1f}%")
print(f"  {result['education']}")