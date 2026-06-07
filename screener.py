"""
AI Resume Screener - Core NLP Engine
Parses resumes, matches skills, and ranks candidates using TF-IDF + cosine similarity
"""

import os
import re
import json
import string
from pathlib import Path

import fitz  # PyMuPDF
import docx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------------------------------------------------------
# Ensure NLTK data is available
# ---------------------------------------------------------------------------
for pkg in ("punkt", "stopwords", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Comprehensive skill taxonomy
# ---------------------------------------------------------------------------
SKILL_TAXONOMY = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "perl", "bash",
        "shell", "sql", "html", "css", "dart", "lua",
    ],
    "Frameworks & Libraries": [
        "react", "angular", "vue", "django", "flask", "fastapi", "spring", "nodejs",
        "express", "nextjs", "nuxt", "tensorflow", "pytorch", "keras", "scikit-learn",
        "pandas", "numpy", "scipy", "matplotlib", "seaborn", "bootstrap", "tailwind",
        "jquery", "graphql", "rest", "grpc",
    ],
    "Databases": [
        "mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle", "cassandra",
        "elasticsearch", "dynamodb", "firebase", "neo4j", "mariadb", "mssql",
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
        "jenkins", "github actions", "ci/cd", "linux", "nginx", "apache", "helm",
        "prometheus", "grafana", "cloudformation",
    ],
    "Data & ML": [
        "machine learning", "deep learning", "nlp", "computer vision", "data science",
        "data analysis", "data engineering", "etl", "spark", "hadoop", "airflow",
        "mlflow", "feature engineering", "statistics", "a/b testing", "tableau",
        "power bi", "looker",
    ],
    "Soft Skills": [
        "leadership", "communication", "teamwork", "problem solving", "agile",
        "scrum", "project management", "collaboration", "analytical", "critical thinking",
    ],
}

ALL_SKILLS = {skill for skills in SKILL_TAXONOMY.values() for skill in skills}

# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(path)
    elif ext == ".txt":
        return Path(path).read_text(errors="ignore")
    raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# NLP helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s\+\#\/\.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_skills(text: str) -> dict[str, list[str]]:
    text_lower = text.lower()
    found: dict[str, list[str]] = {}
    for category, skills in SKILL_TAXONOMY.items():
        matched = []
        for skill in skills:
            # word-boundary aware match
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, text_lower):
                matched.append(skill)
        if matched:
            found[category] = matched
    return found


def extract_email(text: str) -> str:
    m = re.search(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", text, re.I)
    return m.group(0) if m else ""


def extract_phone(text: str) -> str:
    m = re.search(r"(\+?\d[\d\s\-().]{7,}\d)", text)
    return m.group(0).strip() if m else ""


def extract_name(text: str) -> str:
    """Heuristic: first non-empty line that looks like a name."""
    for line in text.split("\n")[:10]:
        line = line.strip()
        if 2 <= len(line.split()) <= 4 and line.replace(" ", "").isalpha():
            return line.title()
    return "Unknown Candidate"


def extract_experience_years(text: str) -> float:
    patterns = [
        r"(\d+)\+?\s*years?\s*of\s*experience",
        r"experience\s*of\s*(\d+)\+?\s*years?",
        r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:work|professional|industry)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            return float(m.group(1))

    # Count year ranges like 2020-2023 or 2018 – Present
    ranges = re.findall(r"(20\d{2}|19\d{2})\s*[-–]\s*(20\d{2}|[Pp]resent)", text)
    total = 0.0
    for start, end in ranges:
        end_yr = 2024 if end.lower() == "present" else int(end)
        total += max(0, end_yr - int(start))
    return round(min(total, 30), 1)  # cap at 30


def extract_education(text: str) -> list[str]:
    degrees = ["phd", "ph.d", "doctorate", "master", "msc", "mba", "bachelor",
               "b.tech", "btech", "b.e", "b.sc", "b.com", "associate"]
    text_lower = text.lower()
    found = []
    for deg in degrees:
        if deg in text_lower:
            found.append(deg.upper())
    return list(dict.fromkeys(found))  # deduplicate preserving order


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

def score_candidate(resume_text: str, jd_text: str, skills_found: dict) -> dict:
    """Return a breakdown of scores."""
    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(jd_text)

    # 1. TF-IDF cosine similarity (overall text match)
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")
    try:
        tfidf = vec.fit_transform([clean_jd, clean_resume])
        text_score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except Exception:
        text_score = 0.0

    # 2. Skill match score
    jd_skills = extract_skills(jd_text)
    jd_all = {s for skills in jd_skills.values() for s in skills}
    resume_all = {s for skills in skills_found.values() for s in skills}
    if jd_all:
        matched = jd_all & resume_all
        skill_score = len(matched) / len(jd_all)
        missing_skills = sorted(jd_all - resume_all)
        matched_skills = sorted(matched)
    else:
        skill_score = text_score  # fallback
        missing_skills = []
        matched_skills = sorted(resume_all)

    # 3. Experience score (diminishing returns)
    exp_years = extract_experience_years(resume_text)
    exp_score = min(exp_years / 10.0, 1.0)

    # 4. Education score
    edu = extract_education(resume_text)
    edu_score = 0.0
    edu_weights = {"PHD": 1.0, "PH.D": 1.0, "DOCTORATE": 1.0,
                   "MASTER": 0.8, "MSC": 0.8, "MBA": 0.8,
                   "BACHELOR": 0.6, "B.TECH": 0.6, "BTECH": 0.6,
                   "B.E": 0.6, "B.SC": 0.5, "B.COM": 0.4, "ASSOCIATE": 0.3}
    if edu:
        edu_score = max(edu_weights.get(d, 0.3) for d in edu)

    # Weighted total
    total = (
        text_score  * 0.35 +
        skill_score * 0.40 +
        exp_score   * 0.15 +
        edu_score   * 0.10
    )

    return {
        "total": round(total * 100, 1),
        "text_similarity": round(text_score * 100, 1),
        "skill_match": round(skill_score * 100, 1),
        "experience": round(exp_score * 100, 1),
        "education": round(edu_score * 100, 1),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "experience_years": exp_years,
        "education_level": edu,
    }


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def parse_resume(file_path: str) -> dict:
    """Parse a single resume file into structured data."""
    text = extract_text(file_path)
    skills = extract_skills(text)
    return {
        "file": Path(file_path).name,
        "raw_text": text,
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": skills,
        "experience_years": extract_experience_years(text),
        "education": extract_education(text),
    }


def screen_resumes(resume_paths: list[str], job_description: str) -> list[dict]:
    """Parse + score all resumes, return ranked list."""
    results = []
    for path in resume_paths:
        try:
            parsed = parse_resume(path)
            scores = score_candidate(parsed["raw_text"], job_description, parsed["skills"])
            result = {**parsed, **scores}
            result.pop("raw_text", None)  # don't send back huge text
            results.append(result)
        except Exception as e:
            results.append({"file": Path(path).name, "error": str(e), "total": 0})

    results.sort(key=lambda x: x.get("total", 0), reverse=True)

    # Add rank
    for i, r in enumerate(results):
        r["rank"] = i + 1
        r["tier"] = (
            "Excellent" if r.get("total", 0) >= 75 else
            "Good"      if r.get("total", 0) >= 55 else
            "Average"   if r.get("total", 0) >= 35 else
            "Below Average"
        )

    return results
