"""
================================================================================
PROJECT TITLE: AI-Powered Interview Question Generator for Intern Recruitment
================================================================================

DESCRIPTION:
    This project builds an automated system that generates custom technical and
    behavioral interview questions tailored to specific intern roles. It uses a
    real-world job postings dataset from Kaggle, processes intern profiles and
    job descriptions, and leverages a transformer-based language model (GPT-2 via
    HuggingFace Transformers) to produce role-specific question sets. The pipeline
    includes data ingestion, preprocessing, a curated question bank, and an LLM-
    powered generation layer — all runnable in Google Colab.

DATASET:
    "Job Description Dataset" from Kaggle:
    https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset
    (CSV with columns: Job Title, Job Description, skills, etc.)

    Fallback synthetic dataset is auto-generated if Kaggle is unavailable.

GOOGLE COLAB SETUP:
    Run Step 0 first to install dependencies, then proceed sequentially.

STEPS:
    0. Install Dependencies
    1. Import Libraries
    2. Load & Explore Dataset
    3. Preprocess Job Descriptions
    4. Build Curated Question Bank
    5. Define Intern Profile Schema
    6. Extract Role Keywords (NLP)
    7. Rule-Based Question Matching (Baseline)
    8. LLM-Based Question Generation (GPT-2)
    9. Assemble Final Question Sets per Role
   10. Export Results & Evaluation Summary
================================================================================
"""

# ============================================================
# STEP 0 — Install Dependencies (Run this cell first in Colab)
# ============================================================
"""
Paste the following into a Colab cell and run it:

!pip install transformers torch pandas scikit-learn kaggle nltk openpyxl --quiet
!python -m nltk.downloader stopwords punkt punkt_tab averaged_perceptron_tagger

To use the Kaggle dataset:
1. Go to https://www.kaggle.com/account → Create API Token → download kaggle.json
2. Upload kaggle.json to Colab
3. Run:
   !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   !kaggle datasets download -d ravindrasinghrana/job-description-dataset --unzip
"""

# ============================================================
# STEP 1 — Import Libraries
# ============================================================
import os
import re
import json
import random
import warnings
import pandas as pd
import numpy as np
from collections import defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

warnings.filterwarnings("ignore")

# Download NLTK data (safe to re-run)
for resource in ["stopwords", "punkt", "averaged_perceptron_tagger"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

print("✅ Libraries loaded successfully.")

# ============================================================
# STEP 2 — Load & Explore Dataset
# ============================================================

def load_dataset(filepath: str = "job_descriptions.csv") -> pd.DataFrame:
    """
    Load the Kaggle job description dataset.
    Falls back to a synthetic dataset for demonstration if file not found.
    """
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        print("Columns:", df.columns.tolist())
    else:
        print("⚠️  Kaggle dataset not found. Generating synthetic dataset...")
        df = generate_synthetic_dataset()
        df.to_csv("job_descriptions_synthetic.csv", index=False)
        print(f"✅ Synthetic dataset created: {df.shape[0]} rows")
    return df


def generate_synthetic_dataset() -> pd.DataFrame:
    """Create a representative synthetic intern job description dataset."""
    records = [
        {
            "Job Title": "Software Engineering Intern",
            "Job Description": (
                "We are looking for a Software Engineering Intern to join our team. "
                "You will work on building scalable backend services using Python and Java. "
                "Responsibilities include writing clean code, participating in code reviews, "
                "and collaborating with senior engineers on system design. "
                "You should have knowledge of data structures, algorithms, and REST APIs."
            ),
            "skills": "Python, Java, REST APIs, Data Structures, Algorithms, Git",
            "Role Category": "Engineering",
            "Experience": "0-1 years",
        },
        {
            "Job Title": "Data Science Intern",
            "Job Description": (
                "Join our Data Science team to help build predictive models and analytics dashboards. "
                "You will preprocess large datasets, run statistical analyses, and visualize insights "
                "using Python libraries such as Pandas, NumPy, and Matplotlib. "
                "Familiarity with machine learning frameworks like Scikit-learn or TensorFlow is a plus."
            ),
            "skills": "Python, Pandas, NumPy, Scikit-learn, SQL, Statistics, Matplotlib",
            "Role Category": "Data & Analytics",
            "Experience": "0-1 years",
        },
        {
            "Job Title": "Product Management Intern",
            "Job Description": (
                "As a Product Management Intern, you will assist in defining product roadmaps, "
                "gathering customer feedback, writing user stories, and coordinating between engineering "
                "and design teams. Strong communication and analytical skills are essential. "
                "Experience with Agile methodology and tools like Jira is preferred."
            ),
            "skills": "Agile, Jira, User Stories, Market Research, Communication, Roadmapping",
            "Role Category": "Product",
            "Experience": "0-1 years",
        },
        {
            "Job Title": "Marketing Intern",
            "Job Description": (
                "We are hiring a Marketing Intern to support our digital marketing campaigns. "
                "Responsibilities include content creation, managing social media accounts, "
                "analyzing campaign performance using Google Analytics, and assisting with SEO strategies. "
                "Creativity and strong writing skills are required."
            ),
            "skills": "Content Writing, SEO, Google Analytics, Social Media, Canva, Campaign Management",
            "Role Category": "Marketing",
            "Experience": "0-1 years",
        },
        {
            "Job Title": "Cybersecurity Intern",
            "Job Description": (
                "Join our Security Operations Center (SOC) as a Cybersecurity Intern. "
                "You will monitor network traffic, investigate security alerts, assist with vulnerability "
                "assessments, and document security incidents. Familiarity with Linux, networking protocols, "
                "and tools like Wireshark or Nmap is expected."
            ),
            "skills": "Linux, Networking, Wireshark, Nmap, Vulnerability Assessment, Python",
            "Role Category": "Security",
            "Experience": "0-1 years",
        },
        {
            "Job Title": "UI/UX Design Intern",
            "Job Description": (
                "We are looking for a UI/UX Design Intern passionate about creating intuitive user interfaces. "
                "You will conduct user research, create wireframes and prototypes in Figma, "
                "and collaborate with developers to implement designs. "
                "A portfolio demonstrating design projects is required."
            ),
            "skills": "Figma, Adobe XD, Wireframing, Prototyping, User Research, HTML, CSS",
            "Role Category": "Design",
            "Experience": "0-1 years",
        },
        {
            "Job Title": "Finance Intern",
            "Job Description": (
                "Support our Finance team in financial modeling, budgeting, and reporting. "
                "You will assist with data analysis in Excel, prepare financial presentations, "
                "and support month-end close activities. "
                "Strong quantitative skills and knowledge of accounting principles are required."
            ),
            "skills": "Excel, Financial Modeling, Accounting, PowerPoint, Bloomberg, SQL",
            "Role Category": "Finance",
            "Experience": "0-1 years",
        },
        {
            "Job Title": "Machine Learning Intern",
            "Job Description": (
                "As a Machine Learning Intern, you will work alongside ML engineers to design, train, "
                "and evaluate models for NLP and computer vision tasks. "
                "You will use PyTorch or TensorFlow, run experiments, and document results. "
                "Strong Python skills and understanding of deep learning concepts are necessary."
            ),
            "skills": "Python, PyTorch, TensorFlow, NLP, Computer Vision, Scikit-learn, Deep Learning",
            "Role Category": "Engineering",
            "Experience": "0-1 years",
        },
    ]
    return pd.DataFrame(records)


# Load the dataset
df = load_dataset("job_descriptions.csv")

# Standardize column names
col_map = {
    "Job Title": "job_title",
    "Job Description": "job_description",
    "skills": "skills",
    "Role Category": "role_category",
    "Experience": "experience",
}
df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

# Filter for intern-related rows if using the full Kaggle dataset
if "job_title" in df.columns and df.shape[0] > 20:
    intern_mask = df["job_title"].str.lower().str.contains("intern", na=False)
    df_intern = df[intern_mask].copy().reset_index(drop=True)
    if df_intern.shape[0] < 5:
        df_intern = df.head(30).copy()  # fallback to top rows
else:
    df_intern = df.copy()

# Fill missing values
for col in ["job_title", "job_description", "skills"]:
    if col in df_intern.columns:
        df_intern[col] = df_intern[col].fillna("")

print(f"\n📊 Intern job postings available: {df_intern.shape[0]}")
print(df_intern[["job_title", "skills"]].head())

# ============================================================
# STEP 3 — Preprocess Job Descriptions
# ============================================================

STOP_WORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """Remove special characters and lowercase."""
    text = re.sub(r"[^a-zA-Z0-9\s,.]", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def extract_keywords(text: str, top_n: int = 15) -> list:
    """
    Extract meaningful keywords from text using TF-IDF-style token filtering.
    Returns top_n keywords sorted by frequency.
    """
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in STOP_WORDS and len(t) > 2]
    freq = defaultdict(int)
    for t in tokens:
        freq[t] += 1
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_tokens[:top_n]]


# Apply preprocessing
df_intern["clean_description"] = df_intern["job_description"].apply(clean_text)
df_intern["keywords"] = df_intern.apply(
    lambda row: extract_keywords(row["job_description"] + " " + row.get("skills", "")),
    axis=1,
)

print("\n🔍 Sample keywords extracted:")
for _, row in df_intern.head(3).iterrows():
    print(f"  [{row['job_title']}]: {', '.join(row['keywords'][:8])}")

# ============================================================
# STEP 4 — Build Curated Question Bank
# ============================================================

QUESTION_BANK = {
    # ── Behavioral ──────────────────────────────────────────
    "behavioral": [
        "Tell me about a time you worked on a team project with a tight deadline. How did you manage your time?",
        "Describe a situation where you had to learn a new technology quickly. What was your approach?",
        "Give an example of a project where you received critical feedback. How did you respond?",
        "Tell me about a conflict you experienced in a team setting and how you resolved it.",
        "Describe a goal you set for yourself and the steps you took to achieve it.",
        "Have you ever made a mistake at work or school? How did you handle it and what did you learn?",
        "Tell me about a time you had to manage multiple tasks simultaneously. What was your strategy?",
        "Describe an instance where you took initiative without being asked. What was the outcome?",
        "Tell me about a time you had to communicate a complex idea to a non-technical audience.",
        "Give an example of how you've contributed to an inclusive or collaborative team environment.",
    ],
    # ── General Technical ───────────────────────────────────
    "general_technical": [
        "What is the difference between object-oriented and functional programming? Give an example.",
        "Explain what version control is and why it's important in software development.",
        "What is the difference between a stack and a queue? When would you use each?",
        "Describe the process of debugging a failing program. What steps do you take?",
        "What is a REST API? How does it differ from a GraphQL API?",
        "Explain the concept of time complexity. What is O(n log n)?",
        "What is a relational database? How does it differ from a NoSQL database?",
        "What is agile methodology? How does it compare to waterfall?",
        "Describe the MVC (Model-View-Controller) design pattern.",
        "What is CI/CD and why is it important for modern software teams?",
    ],
    # ── Domain-Specific ──────────────────────────────────────
    "python": [
        "What is the difference between a list and a tuple in Python?",
        "Explain Python's GIL (Global Interpreter Lock). How does it affect multi-threaded programs?",
        "What are Python decorators? Write a simple example.",
        "How does memory management work in Python? What is garbage collection?",
        "Explain the difference between `deepcopy` and `copy` in Python.",
    ],
    "machine_learning": [
        "What is the difference between supervised and unsupervised learning?",
        "Explain overfitting. How do you detect and prevent it?",
        "What is cross-validation and why is it used in model evaluation?",
        "Describe the bias-variance tradeoff in machine learning.",
        "What is gradient descent? How does learning rate affect convergence?",
        "Explain the difference between precision and recall. When do you optimize for each?",
        "What is a confusion matrix? Interpret a sample one for a binary classifier.",
    ],
    "data_science": [
        "How would you handle missing values in a dataset?",
        "What is feature engineering? Provide an example of a useful feature you might create.",
        "Explain the difference between normalization and standardization.",
        "How would you approach an exploratory data analysis (EDA) on a new dataset?",
        "What statistical tests would you use to compare means across two groups?",
    ],
    "sql": [
        "What is the difference between INNER JOIN, LEFT JOIN, and FULL OUTER JOIN?",
        "Explain what an index is in SQL and how it improves query performance.",
        "Write a SQL query to find the second highest salary in a table.",
        "What is the difference between WHERE and HAVING in SQL?",
        "Explain GROUP BY. Give an example use case.",
    ],
    "cybersecurity": [
        "What is the CIA triad in information security? Give a real-world example of each.",
        "Explain the difference between symmetric and asymmetric encryption.",
        "What is a SQL injection attack? How can it be prevented?",
        "Describe the process of a penetration test. What are the key phases?",
        "What is a firewall and how does it differ from an intrusion detection system (IDS)?",
        "Explain what a man-in-the-middle (MITM) attack is.",
    ],
    "product_management": [
        "How would you prioritize features in a product backlog?",
        "Walk me through how you would gather and incorporate user feedback.",
        "What metrics would you track to measure the success of a product feature launch?",
        "How do you write a strong user story? Provide an example.",
        "Describe how you would manage competing priorities from different stakeholders.",
    ],
    "marketing": [
        "What is a conversion funnel? Describe each stage.",
        "How would you measure the ROI of a digital marketing campaign?",
        "Explain the difference between SEO and SEM.",
        "What are key performance indicators (KPIs) you would track for a social media campaign?",
        "How would you approach A/B testing for a new email campaign?",
    ],
    "design": [
        "Walk me through your UX design process from research to delivery.",
        "What is the difference between UX and UI design?",
        "How do you prioritize usability when working under tight deadlines?",
        "Explain what accessibility means in design. How do you ensure your designs are accessible?",
        "Describe a time when user research changed the direction of your design.",
    ],
    "finance": [
        "Explain the three core financial statements and how they connect.",
        "What is the time value of money? Give an example.",
        "How would you value a company using the DCF (Discounted Cash Flow) method?",
        "What is working capital and why is it important?",
        "Explain what EBITDA stands for and why analysts use it.",
    ],
}

print(f"✅ Question bank loaded: {sum(len(v) for v in QUESTION_BANK.values())} total questions across {len(QUESTION_BANK)} categories.")

# ============================================================
# STEP 5 — Define Intern Profile Schema
# ============================================================

def create_intern_profile(
    name: str,
    job_title: str,
    skills: list,
    education: str = "Computer Science",
    experience_months: int = 0,
    notes: str = "",
) -> dict:
    """
    Create a structured intern candidate profile used to personalize questions.
    """
    return {
        "name": name,
        "job_title": job_title,
        "skills": skills,
        "education": education,
        "experience_months": experience_months,
        "notes": notes,
    }


# Example profiles
SAMPLE_PROFILES = [
    create_intern_profile(
        name="Alex Johnson",
        job_title="Software Engineering Intern",
        skills=["Python", "Java", "Git", "REST APIs"],
        education="Computer Science",
        experience_months=0,
    ),
    create_intern_profile(
        name="Maria Chen",
        job_title="Data Science Intern",
        skills=["Python", "Pandas", "NumPy", "SQL", "Scikit-learn"],
        education="Statistics",
        experience_months=3,
        notes="Prior research assistant experience",
    ),
    create_intern_profile(
        name="David Kim",
        job_title="Cybersecurity Intern",
        skills=["Linux", "Networking", "Python", "Wireshark"],
        education="Information Security",
        experience_months=0,
    ),
    create_intern_profile(
        name="Sophie Williams",
        job_title="Product Management Intern",
        skills=["Agile", "Jira", "User Stories", "Communication"],
        education="Business Administration",
        experience_months=6,
        notes="Co-founded a student startup",
    ),
]

print("✅ Sample intern profiles created:")
for p in SAMPLE_PROFILES:
    print(f"   → {p['name']} | {p['job_title']} | Skills: {', '.join(p['skills'][:3])}...")

# ============================================================
# STEP 6 — Extract Role Keywords via TF-IDF
# ============================================================

def build_tfidf_keyword_map(dataframe: pd.DataFrame) -> dict:
    """
    Use TF-IDF to extract the most discriminative keywords per job title
    from the dataset of job descriptions.
    """
    if dataframe.shape[0] < 2:
        return {}

    tfidf = TfidfVectorizer(max_features=200, stop_words="english", ngram_range=(1, 2))
    corpus = dataframe["clean_description"].tolist()
    try:
        tfidf_matrix = tfidf.fit_transform(corpus)
    except Exception:
        return {}

    feature_names = tfidf.get_feature_names_out()
    keyword_map = {}

    for i, row in dataframe.iterrows():
        scores = tfidf_matrix[i].toarray().flatten()
        top_indices = scores.argsort()[-10:][::-1]
        top_keywords = [feature_names[j] for j in top_indices if scores[j] > 0]
        keyword_map[row["job_title"]] = top_keywords

    return keyword_map


TFIDF_KEYWORD_MAP = build_tfidf_keyword_map(df_intern)
print("\n🔑 TF-IDF keyword map sample:")
for title, kws in list(TFIDF_KEYWORD_MAP.items())[:3]:
    print(f"  [{title}]: {', '.join(kws[:5])}")

# ============================================================
# STEP 7 — Rule-Based Question Matching (Baseline System)
# ============================================================

# Keyword → question category mapping
CATEGORY_KEYWORDS = {
    "python": ["python", "flask", "django", "pandas", "numpy"],
    "machine_learning": ["machine learning", "deep learning", "pytorch", "tensorflow", "neural", "model", "training"],
    "data_science": ["data", "analysis", "statistics", "visualization", "eda", "feature"],
    "sql": ["sql", "database", "query", "mysql", "postgresql", "relational"],
    "cybersecurity": ["security", "linux", "network", "wireshark", "nmap", "vulnerability", "firewall", "encryption"],
    "product_management": ["product", "roadmap", "agile", "jira", "user story", "stakeholder", "prioritiz"],
    "marketing": ["marketing", "seo", "campaign", "social media", "analytics", "content", "digital"],
    "design": ["figma", "ux", "ui", "wireframe", "prototype", "user research", "design"],
    "finance": ["finance", "excel", "accounting", "financial model", "budget", "dcf", "valuation"],
}


def identify_categories(profile: dict, job_desc: str = "") -> list:
    """
    Match a candidate profile to question categories based on skills and job description.
    Returns ordered list of relevant categories.
    """
    combined_text = (
        " ".join(profile["skills"]).lower()
        + " "
        + profile["job_title"].lower()
        + " "
        + job_desc.lower()
    )
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in combined_text)
        if score > 0:
            scores[category] = score

    # Always include behavioral and general technical
    result = ["behavioral", "general_technical"]
    result += sorted(scores, key=scores.get, reverse=True)
    return result


def rule_based_question_set(
    profile: dict,
    job_desc: str = "",
    n_behavioral: int = 3,
    n_technical: int = 5,
    n_domain: int = 4,
) -> dict:
    """
    Select questions from the curated bank using rule-based category matching.
    Returns a structured question set for the given intern profile.
    """
    categories = identify_categories(profile, job_desc)
    selected = {"behavioral": [], "technical": [], "domain_specific": []}

    # Behavioral questions
    behavioral_pool = QUESTION_BANK.get("behavioral", [])
    selected["behavioral"] = random.sample(behavioral_pool, min(n_behavioral, len(behavioral_pool)))

    # General technical questions
    tech_pool = QUESTION_BANK.get("general_technical", [])
    selected["technical"] = random.sample(tech_pool, min(n_technical, len(tech_pool)))

    # Domain-specific questions from matched categories
    domain_pool = []
    for cat in categories:
        if cat not in ("behavioral", "general_technical"):
            domain_pool.extend(QUESTION_BANK.get(cat, []))

    domain_pool = list(set(domain_pool))  # deduplicate
    selected["domain_specific"] = random.sample(domain_pool, min(n_domain, len(domain_pool))) if domain_pool else []

    return selected


# Test the rule-based system
print("\n" + "="*60)
print("📋 RULE-BASED QUESTION SET — Sample Output")
print("="*60)
profile_test = SAMPLE_PROFILES[1]  # Data Science Intern
job_row = df_intern[df_intern["job_title"].str.lower().str.contains("data", na=False)].head(1)
job_desc_sample = job_row["job_description"].values[0] if len(job_row) > 0 else ""

qset = rule_based_question_set(profile_test, job_desc=job_desc_sample)

print(f"\nCandidate: {profile_test['name']} | Role: {profile_test['job_title']}\n")
for section, questions in qset.items():
    print(f"  [{section.upper().replace('_', ' ')}]")
    for i, q in enumerate(questions, 1):
        print(f"    {i}. {q}")
    print()

# ============================================================
# STEP 8 — LLM-Based Question Generation (GPT-2)
# ============================================================

print("⏳ Loading GPT-2 language model (this may take ~1 min on first run)...")

try:
    tokenizer_gpt = GPT2Tokenizer.from_pretrained("gpt2")
    model_gpt = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer_gpt.pad_token = tokenizer_gpt.eos_token
    text_generator = pipeline(
        "text-generation",
        model=model_gpt,
        tokenizer=tokenizer_gpt,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.85,
        top_p=0.92,
        repetition_penalty=1.3,
        pad_token_id=tokenizer_gpt.eos_token_id,
    )
    LLM_AVAILABLE = True
    print("✅ GPT-2 loaded successfully.")
except Exception as e:
    LLM_AVAILABLE = False
    print(f"⚠️  Could not load GPT-2: {e}\n   Falling back to rule-based generation only.")


def build_prompt(profile: dict, question_type: str, context_skills: list = None) -> str:
    """
    Construct a structured prompt to guide GPT-2 question generation.
    """
    skills = context_skills or profile["skills"][:4]
    skill_str = ", ".join(skills)

    if question_type == "technical":
        return (
            f"Interview question for a {profile['job_title']} with skills in {skill_str}:\n"
            f"Technical question: How would you"
        )
    elif question_type == "behavioral":
        return (
            f"Behavioral interview question for a {profile['job_title']}:\n"
            f"Tell me about a time when you"
        )
    elif question_type == "scenario":
        return (
            f"Scenario-based question for a {profile['job_title']} intern:\n"
            f"Imagine you are given a task involving {skills[0]}. What steps would you"
        )
    return f"Interview question for {profile['job_title']}:"


def clean_generated_question(text: str) -> str:
    """Post-process LLM output to extract a clean question."""
    # Split on newline and take first meaningful line
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 20]
    if not lines:
        return text.strip()
    candidate = lines[0]
    # Ensure it ends with a question mark
    if "?" not in candidate:
        candidate = candidate.rstrip(".") + "?"
    # Truncate at second sentence to avoid run-on
    sentences = re.split(r"(?<=[.?!])\s", candidate)
    return sentences[0].strip()


def generate_llm_questions(
    profile: dict,
    n_technical: int = 2,
    n_behavioral: int = 1,
    n_scenario: int = 1,
) -> dict:
    """
    Use GPT-2 to generate novel interview questions tailored to the intern profile.
    """
    generated = {"technical": [], "behavioral": [], "scenario": []}

    if not LLM_AVAILABLE:
        return generated

    configs = [
        ("technical", n_technical),
        ("behavioral", n_behavioral),
        ("scenario", n_scenario),
    ]

    for q_type, count in configs:
        for _ in range(count):
            try:
                prompt = build_prompt(profile, q_type)
                outputs = text_generator(prompt, num_return_sequences=1)
                raw_text = outputs[0]["generated_text"]
                # Strip prompt prefix
                raw_text = raw_text[len(prompt):].strip()
                cleaned = clean_generated_question(raw_text)
                if len(cleaned) > 15:
                    generated[q_type].append(cleaned)
            except Exception as ex:
                print(f"  ⚠️  Generation error for {q_type}: {ex}")

    return generated


# Test LLM generation
print("\n" + "="*60)
print("🤖 LLM-GENERATED QUESTIONS — Sample Output")
print("="*60)
llm_profile = SAMPLE_PROFILES[0]  # Software Engineering Intern
print(f"\nGenerating for: {llm_profile['name']} | {llm_profile['job_title']}")

llm_questions = generate_llm_questions(llm_profile, n_technical=2, n_behavioral=1, n_scenario=1)

for section, questions in llm_questions.items():
    if questions:
        print(f"\n  [LLM-GENERATED {section.upper()}]")
        for i, q in enumerate(questions, 1):
            print(f"    {i}. {q}")

# ============================================================
# STEP 9 — Assemble Final Question Sets per Role
# ============================================================

def generate_full_interview_set(
    profile: dict,
    dataframe: pd.DataFrame,
    include_llm: bool = True,
) -> dict:
    """
    Assemble a complete, deduplicated interview question set for an intern candidate.
    Combines rule-based selection and LLM generation.
    """
    # Find best matching job description from dataset
    job_desc = ""
    for _, row in dataframe.iterrows():
        if profile["job_title"].lower() in row.get("job_title", "").lower():
            job_desc = row.get("job_description", "")
            break

    # Rule-based questions
    rb_questions = rule_based_question_set(
        profile, job_desc=job_desc, n_behavioral=3, n_technical=4, n_domain=4
    )

    # LLM questions (if available)
    llm_questions = {}
    if include_llm and LLM_AVAILABLE:
        llm_questions = generate_llm_questions(profile, n_technical=2, n_behavioral=1, n_scenario=1)

    # Merge and deduplicate
    final_set = {
        "candidate": profile["name"],
        "role": profile["job_title"],
        "skills": profile["skills"],
        "questions": {
            "behavioral": rb_questions["behavioral"],
            "technical_core": rb_questions["technical"],
            "domain_specific": rb_questions["domain_specific"],
            "llm_generated_technical": llm_questions.get("technical", []),
            "llm_generated_behavioral": llm_questions.get("behavioral", []),
            "llm_generated_scenario": llm_questions.get("scenario", []),
        },
    }

    # Count totals
    total = sum(len(v) for v in final_set["questions"].values())
    final_set["total_questions"] = total

    return final_set


# Generate question sets for all sample profiles
print("\n" + "="*60)
print("📦 FINAL INTERVIEW QUESTION SETS — All Profiles")
print("="*60)

all_interview_sets = []
for profile in SAMPLE_PROFILES:
    interview_set = generate_full_interview_set(profile, df_intern, include_llm=LLM_AVAILABLE)
    all_interview_sets.append(interview_set)
    print(f"\n✅ {profile['name']} ({profile['job_title']}): {interview_set['total_questions']} questions generated")

# ============================================================
# STEP 10 — Export Results & Evaluation Summary
# ============================================================

def export_to_excel(interview_sets: list, filename: str = "interview_questions.xlsx"):
    """Export all interview question sets to a formatted Excel file."""
    rows = []
    for record in interview_sets:
        for section, questions in record["questions"].items():
            for q in questions:
                rows.append({
                    "Candidate": record["candidate"],
                    "Role": record["role"],
                    "Skills": ", ".join(record["skills"]),
                    "Question Section": section.replace("_", " ").title(),
                    "Question": q,
                    "Source": "LLM-Generated" if "llm" in section else "Curated Bank",
                })
    df_export = pd.DataFrame(rows)
    df_export.to_excel(filename, index=False)
    print(f"📁 Exported to: {filename} ({len(df_export)} rows)")
    return df_export


def export_to_json(interview_sets: list, filename: str = "interview_questions.json"):
    """Export all interview question sets to JSON."""
    with open(filename, "w") as f:
        json.dump(interview_sets, f, indent=2)
    print(f"📁 Exported to: {filename}")


def print_evaluation_summary(interview_sets: list):
    """Print a statistical evaluation summary of the generated question sets."""
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    total_candidates = len(interview_sets)
    total_questions = sum(s["total_questions"] for s in interview_sets)
    avg_questions = total_questions / total_candidates if total_candidates else 0

    curated_count = sum(
        len(s["questions"]["behavioral"])
        + len(s["questions"]["technical_core"])
        + len(s["questions"]["domain_specific"])
        for s in interview_sets
    )
    llm_count = sum(
        len(s["questions"]["llm_generated_technical"])
        + len(s["questions"]["llm_generated_behavioral"])
        + len(s["questions"]["llm_generated_scenario"])
        for s in interview_sets
    )

    print(f"  Total Candidates Processed : {total_candidates}")
    print(f"  Total Questions Generated  : {total_questions}")
    print(f"  Avg Questions per Candidate: {avg_questions:.1f}")
    print(f"  Curated Bank Questions     : {curated_count}")
    print(f"  LLM-Generated Questions    : {llm_count}")
    print(f"  LLM Available              : {'Yes (GPT-2)' if LLM_AVAILABLE else 'No (rule-based only)'}")
    print(f"  Unique Roles Covered       : {len(set(s['role'] for s in interview_sets))}")
    print(f"  Question Categories Used   : {list(QUESTION_BANK.keys())}")
    print()

    print("  Per-Candidate Breakdown:")
    for s in interview_sets:
        print(f"    {s['candidate']:<20} | {s['role']:<35} | {s['total_questions']} questions")

    print("\n✅ Pipeline complete. Ready for recruiter review.")


# Run exports and summary
df_results = export_to_excel(all_interview_sets, "interview_questions.xlsx")
export_to_json(all_interview_sets, "interview_questions.json")
print_evaluation_summary(all_interview_sets)

# ── Quick preview of exported DataFrame ──────────────────────
print("\n🔎 Preview of exported data:")
print(df_results[["Candidate", "Role", "Question Section", "Question"]].head(10).to_string(index=False))

print("\n" + "="*60)
print("🎉 PROJECT COMPLETE")
print("="*60)
print("""
OUTPUT FILES:
  • interview_questions.xlsx  — Formatted Excel with all question sets
  • interview_questions.json  — Machine-readable JSON for downstream use

NEXT STEPS FOR IMPROVEMENT:
  • Fine-tune GPT-2 on a domain-specific question corpus
  • Integrate GPT-3.5/4 via OpenAI API for higher-quality generation
  • Add a Streamlit or Gradio front-end for recruiter interaction
  • Expand curated question bank with community-sourced data
  • Add difficulty levels (Easy / Medium / Hard) to each question
  • Integrate LeetCode-style coding challenge links for SWE roles
""")