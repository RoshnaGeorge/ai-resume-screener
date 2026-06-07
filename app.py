"""
Flask API for the AI Resume Screener
"""

import os
import json
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from screener import screen_resumes, parse_resume, SKILL_TAXONOMY

app = Flask(__name__, static_folder="static")
CORS(app)

UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "txt"}

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/skills", methods=["GET"])
def get_skills():
    return jsonify(SKILL_TAXONOMY)


@app.route("/api/screen", methods=["POST"])
def screen():
    # Validate job description
    jd = request.form.get("job_description", "").strip()
    if not jd:
        return jsonify({"error": "Job description is required"}), 400

    # Validate files
    files = request.files.getlist("resumes")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "At least one resume file is required"}), 400

    session_id = uuid.uuid4().hex
    session_dir = UPLOAD_FOLDER / session_id
    session_dir.mkdir()

    saved_paths = []
    for f in files:
        if f and allowed_file(f.filename):
            fname = secure_filename(f.filename)
            dest = session_dir / fname
            f.save(str(dest))
            saved_paths.append(str(dest))

    if not saved_paths:
        return jsonify({"error": "No valid files uploaded (PDF/DOCX/TXT only)"}), 400

    try:
        results = screen_resumes(saved_paths, jd)
    except Exception as e:
        return jsonify({"error": f"Screening failed: {e}"}), 500
    finally:
        # Clean up uploaded files
        import shutil
        shutil.rmtree(str(session_dir), ignore_errors=True)

    return jsonify({"candidates": results, "total": len(results)})


@app.route("/api/demo", methods=["POST"])
def demo():
    """Run screening on synthetic demo resumes so users can try without uploads."""
    jd = request.json.get("job_description", "").strip() if request.is_json else ""
    if not jd:
        jd = DEFAULT_JD

    results = screen_resumes_from_text(DEMO_RESUMES, jd)
    return jsonify({"candidates": results, "total": len(results)})


# ---------------------------------------------------------------------------
# Demo data (synthetic resumes as plain text)
# ---------------------------------------------------------------------------

DEFAULT_JD = """
We are looking for a Senior Python Developer with strong experience in machine learning and backend development.
Requirements:
- 3+ years of Python development experience
- Strong knowledge of machine learning frameworks: TensorFlow, PyTorch, or scikit-learn
- Experience with REST API development using Flask or FastAPI
- Proficiency in SQL and NoSQL databases (PostgreSQL, MongoDB)
- Familiarity with Docker, Kubernetes, and CI/CD pipelines
- Excellent problem-solving and communication skills
- Bachelor's degree in Computer Science or related field
Nice to have: NLP experience, AWS/GCP, React
"""

DEMO_RESUMES = [
    {
        "name": "Priya Sharma",
        "text": """
Priya Sharma
priya.sharma@email.com | +91-9876543210

SUMMARY
Senior Python Developer with 5 years of experience in machine learning and backend systems.

SKILLS
Python, TensorFlow, PyTorch, scikit-learn, Flask, FastAPI, PostgreSQL, MongoDB, Docker, Kubernetes,
AWS, REST API, Git, CI/CD, NLP, deep learning, data science, React

EXPERIENCE
Senior ML Engineer – TechCorp India (2021–Present)
- Built NLP pipelines using Python and PyTorch for customer sentiment analysis
- Designed REST APIs with FastAPI serving 10K+ requests/day
- Deployed models on AWS using Docker and Kubernetes

Python Developer – DataDriven Solutions (2019–2021)
- Developed ML models using scikit-learn and TensorFlow
- Created PostgreSQL schemas and optimised queries

EDUCATION
B.Tech Computer Science – IIT Delhi, 2019
""",
    },
    {
        "name": "Arjun Mehta",
        "text": """
Arjun Mehta
arjun.mehta@gmail.com | 9823456710

Objective: Seeking a Python developer role.

Skills: Python, Flask, MySQL, HTML, CSS, JavaScript, Git, Agile

Work Experience
Junior Developer – Startup XYZ (2022–2024)
Worked on Flask APIs and MySQL databases. Built simple dashboards.

EDUCATION
B.Sc Computer Science – Mumbai University, 2022
""",
    },
    {
        "name": "Dr. Kavita Reddy",
        "text": """
Dr. Kavita Reddy
kavita.reddy@research.edu

PhD in Machine Learning | Computer Vision | NLP

EXPERTISE
Python, PyTorch, TensorFlow, scikit-learn, deep learning, NLP, computer vision,
data analysis, statistics, research, PostgreSQL, REST, Docker, AWS, GCP, Kubernetes, FastAPI

EXPERIENCE
Research Scientist – AI Lab (2018–Present)
- Published 8 papers on NLP and deep learning
- Led team of 4 engineers building production ML systems
- 6 years of Python, 4 years of cloud infrastructure

Postdoctoral Researcher (2016–2018) – ML pipelines and data engineering

EDUCATION
PhD Machine Learning – IISc Bangalore, 2016
Master of Science – IIT Bombay, 2012
""",
    },
    {
        "name": "Rahul Gupta",
        "text": """
Rahul Gupta | rahul@dev.io

Full Stack Developer – 3 years experience

Tech Stack: JavaScript, React, Node.js, Python (basic), MongoDB, PostgreSQL, Docker, Git, REST APIs

Work:
- Frontend Developer at WebAgency 2021–2024
- Built React dashboards and Node.js backends
- Some Python scripting for automation tasks

Education: B.E. Information Technology – VTU, 2021
""",
    },
    {
        "name": "Sneha Joshi",
        "text": """
Sneha Joshi
sneha.joshi@ml.com | Bengaluru

PROFILE
Data Scientist with 4 years experience in machine learning, NLP, and data analysis.

TECHNICAL SKILLS
Python, scikit-learn, TensorFlow, pandas, numpy, SQL, PostgreSQL, Flask, REST API,
machine learning, NLP, statistics, A/B testing, Tableau, Git, Docker, AWS

WORK HISTORY
Data Scientist – Analytics Firm (2020–2024)
- Built and deployed ML classification models using scikit-learn and TensorFlow
- Developed Flask APIs to serve model predictions
- 4 years Python, strong SQL and data engineering background

EDUCATION
MBA + B.Sc Statistics – Pune University, 2020
""",
    },
]


def screen_resumes_from_text(demo_data: list[dict], jd: str) -> list[dict]:
    """Screen demo resumes given as dicts with 'name' and 'text'."""
    import tempfile, os
    from screener import extract_skills, extract_email, extract_phone, \
        extract_experience_years, extract_education, score_candidate

    results = []
    for item in demo_data:
        text = item["text"]
        skills = extract_skills(text)
        scores = score_candidate(text, jd, skills)
        result = {
            "file": f"{item['name'].replace(' ', '_')}.pdf",
            "name": item["name"],
            "email": extract_email(text) or "—",
            "phone": extract_phone(text) or "—",
            "skills": skills,
            "experience_years": extract_experience_years(text),
            "education": extract_education(text),
            **scores,
        }
        results.append(result)

    results.sort(key=lambda x: x.get("total", 0), reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1
        r["tier"] = (
            "Excellent"     if r["total"] >= 75 else
            "Good"          if r["total"] >= 55 else
            "Average"       if r["total"] >= 35 else
            "Below Average"
        )
    return results


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
