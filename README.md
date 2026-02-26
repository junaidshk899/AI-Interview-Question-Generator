# 🤖 AI-Powered Interview Question Generator

> Automated, role-specific technical and behavioral interview question sets for intern recruitment — combining a curated question bank with GPT-2 language model generation.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## 📌 Project Title

**AI-Powered Interview Question Generator for Intern Recruitment**

---

## 📝 Description

This project is an end-to-end Python pipeline that automatically generates customized interview question sets tailored to specific intern roles. It ingests real-world job postings from Kaggle, processes structured intern candidate profiles, and uses a hybrid approach — combining a hand-curated question bank with a GPT-2 transformer model — to produce complete, role-specific question sets ready for recruiter use.

The pipeline covers 8 intern roles across Engineering, Data Science, Cybersecurity, Product Management, Marketing, Design, Finance, and Machine Learning. Each generated set includes behavioral, technical, domain-specific, and LLM-generated questions, all exportable to Excel and JSON. A standalone HTML dashboard allows recruiters to browse, filter, search, and export question sets directly in any web browser — no server required.

---

## 🗂️ Project Structure

```
ai-interview-question-generator/
│
├── main.py                            # Full Python pipeline (10 steps)
├── requirements.txt                   # Python dependencies
├── Interview question generator.html  # Standalone browser dashboard
├── interview_questions.json           # Sample output (JSON)
├── interview_questions.xlsx           # Sample output (Excel)
└── README.md                          # This file
```

---

## ✨ Features

- **Hybrid question generation** — curated expert bank + GPT-2 LLM for novel questions
- **Role-aware personalization** — TF-IDF keyword extraction maps job descriptions to question categories
- **8 intern roles covered** — SWE, Data Science, ML, Cybersecurity, PM, Marketing, Design, Finance
- **60+ curated questions** across 11 categories (behavioral, Python, SQL, ML, security, and more)
- **Interactive HTML dashboard** — filter by role, search by keyword, export to CSV/JSON
- **Dual export** — Excel `.xlsx` and JSON outputs for downstream recruiter workflows
- **Kaggle dataset integration** with automatic synthetic fallback
- **Google Colab ready** — runs without any local setup

---

## 🖥️ Live Dashboard Preview

Open `Interview question generator.html` directly in your browser — no installation needed.

| Feature | Description |
|---|---|
| Role filters | Filter candidates by Engineering, Data Science, Cybersecurity, Product |
| Live search | Highlights matching keywords across all questions in real time |
| Collapsible cards | Expand/collapse per candidate and per question section |
| Source badges | Each question labeled as `Bank` (curated) or `LLM` (AI-generated) |
| Export buttons | Download all questions as CSV or JSON with one click |

---

## 📦 Dataset

| Source | Details |
|---|---|
| **Kaggle** | [Job Description Dataset](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset) by Ravindra Singh Rana |
| **Fallback** | Auto-generated synthetic dataset (8 roles) — no Kaggle account needed |

---

## 🚀 Getting Started

### Option 1 — Run in Google Colab (Recommended)

1. Upload `main.py` to a new Colab notebook
2. Run the setup cell:

```python
!pip install transformers torch pandas scikit-learn kaggle nltk openpyxl --quiet
!python -m nltk.downloader stopwords punkt averaged_perceptron_tagger
```

3. Execute the script:

```python
exec(open("main.py").read())
```

### Option 2 — Run Locally in VS Code

**Step 1** — Clone the repository:
```bash
git clone https://github.com/your-username/ai-interview-question-generator.git
cd ai-interview-question-generator
```

**Step 2** — Create and activate a virtual environment:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

**Step 3** — Install dependencies:
```bash
pip install -r requirements.txt
```

**Step 4** — Download NLTK data:
```bash
python -m nltk.downloader stopwords punkt averaged_perceptron_tagger
```

**Step 5** — Run the pipeline:
```bash
python main.py
```

**Step 6** — Open the dashboard:

Simply double-click `Interview question generator.html` — it opens in your browser instantly.

---

## ⚙️ Pipeline Steps

| Step | Description |
|---|---|
| 1 | Import libraries and configure environment |
| 2 | Load Kaggle dataset or generate synthetic fallback |
| 3 | Clean and preprocess job descriptions (tokenization, stopword removal) |
| 4 | Build curated question bank (60+ questions, 11 categories) |
| 5 | Define structured intern profile schema |
| 6 | Extract role-discriminative keywords via TF-IDF |
| 7 | Rule-based question matching (baseline system) |
| 8 | GPT-2 LLM question generation with role-aware prompts |
| 9 | Assemble and deduplicate final question sets per candidate |
| 10 | Export to Excel + JSON and print evaluation summary |

---

## 📊 Sample Output

```
Total Candidates Processed : 4
Total Questions Generated  : 60
Avg Questions per Candidate: 15.0
Curated Bank Questions     : 44
LLM-Generated Questions    : 16
Unique Roles Covered       : 4
```

**Per-candidate breakdown:**

| Candidate | Role | Questions |
|---|---|---|
| Alex Johnson | Software Engineering Intern | 15 |
| Maria Chen | Data Science Intern | 15 |
| David Kim | Cybersecurity Intern | 15 |
| Sophie Williams | Product Management Intern | 15 |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| LLM | GPT-2 via HuggingFace Transformers |
| NLP | NLTK, Scikit-learn (TF-IDF) |
| Data | Pandas, NumPy |
| Export | openpyxl (Excel), JSON |
| Dashboard | Vanilla HTML / CSS / JavaScript |
| Dataset | Kaggle Job Description Dataset |

---

## 📋 Requirements

```
pandas
numpy
scikit-learn
transformers
torch
nltk
openpyxl
tqdm
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

- Fine-tune GPT-2 on a domain-specific interview question corpus for higher quality generation
- Integrate GPT-3.5 / GPT-4 via OpenAI API as a drop-in model upgrade
- Add a Streamlit or Gradio front-end for interactive recruiter use
- Add difficulty levels (Easy / Medium / Hard) per question
- Integrate LeetCode-style coding challenge links for SWE roles
- Support CSV upload of custom candidate profiles via the dashboard

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙋 Author

Built as a submission-ready academic and portfolio project demonstrating NLP, LLM integration, data preprocessing, and full-stack output generation in Python.

---

> ⭐ If you found this project useful, consider giving it a star on GitHub!
