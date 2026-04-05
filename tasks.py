"""
tasks.py — Task definitions for the Resume Screening OpenEnv.
Includes 3 difficulty levels with realistic, multi-line job descriptions and resumes.
"""

from typing import Dict, Any

TASKS: Dict[str, Dict[str, Any]] = {

    # ─── TASK 1: EASY — Clear Strong Match ────────────────────────────────────
    "task_easy": {
        "task_id": "task_easy",
        "difficulty": "easy",
        "expected_decision": "accept",
        "required_skills": [
            "python", "machine learning", "tensorflow", "sql",
            "data analysis", "scikit-learn", "deep learning"
        ],
        "good_reasoning_keywords": [
            "strong match", "qualified", "experience", "skills align",
            "meets requirements", "relevant", "exceeds"
        ],
        "job_description": """
Position: Senior Machine Learning Engineer
Company: DataCore Analytics Inc.
Location: Remote (US-based)

About the Role:
We are seeking a Senior Machine Learning Engineer to join our growing AI team.
You will design, build, and deploy production ML pipelines and collaborate closely
with data scientists and product teams to bring models to life at scale.

Key Responsibilities:
- Design and implement scalable machine learning models and pipelines
- Work with large datasets using SQL and Python to extract insights
- Build and maintain deep learning models using TensorFlow and PyTorch
- Collaborate with cross-functional teams to productionize ML solutions
- Conduct code reviews and mentor junior engineers

Required Qualifications:
- 5+ years of professional experience in machine learning or data science
- Strong proficiency in Python and SQL
- Hands-on experience with TensorFlow, PyTorch, or similar frameworks
- Experience with scikit-learn for classical ML algorithms
- Demonstrated ability to deploy ML models to production
- Familiarity with cloud platforms (AWS, GCP, or Azure)

Nice to Have:
- Experience with MLflow or similar experiment tracking tools
- Contributions to open-source ML projects
- Familiarity with distributed computing frameworks (Spark, Dask)

Compensation: $160,000 – $200,000 base salary + equity
        """.strip(),

        "resume": """
JANE SMITH
San Francisco, CA | jane.smith@email.com | linkedin.com/in/janesmith | github.com/janesmith

SUMMARY
Seasoned Machine Learning Engineer with 7 years of experience building and deploying
production-grade ML systems across healthcare, fintech, and e-commerce domains.
Deep expertise in Python, TensorFlow, and scalable data pipelines.

EXPERIENCE

Senior ML Engineer — HealthAI Corp (2021–Present)
- Led a team of 4 engineers to build a clinical risk prediction model using TensorFlow
  serving 2M+ patient records; reduced false negatives by 18%
- Designed end-to-end ML pipelines with Airflow, reducing model deployment time from
  3 weeks to 2 days
- Wrote complex SQL queries and dbt models to prepare training datasets from 10TB+ warehouses
- Mentored 3 junior data scientists; introduced MLflow for experiment tracking

ML Engineer — FinEdge Technologies (2018–2021)
- Built fraud detection models using scikit-learn and XGBoost; improved AUC from 0.87 to 0.94
- Implemented deep learning recommendation system using TensorFlow; increased CTR by 22%
- Collaborated with product teams to define ML requirements and delivery timelines
- Deployed models to AWS SageMaker; maintained uptime above 99.9%

Data Analyst — RetailNow (2016–2018)
- Conducted data analysis using Python (pandas, numpy) and SQL on customer behavior datasets
- Built dashboards using Tableau; presented weekly insights to executive stakeholders

EDUCATION
M.S. Computer Science (Machine Learning focus) — Stanford University, 2016
B.S. Statistics — UC Berkeley, 2014

SKILLS
Languages: Python, SQL, Bash, R
Frameworks: TensorFlow, PyTorch, scikit-learn, XGBoost, Keras
Tools: MLflow, Airflow, Docker, AWS SageMaker, GCP Vertex AI, Spark
        """.strip(),
    },

    # ─── TASK 2: MEDIUM — Clear Mismatch ──────────────────────────────────────
    "task_medium": {
        "task_id": "task_medium",
        "difficulty": "medium",
        "expected_decision": "reject",
        "required_skills": [
            "kubernetes", "terraform", "ci/cd", "devops",
            "aws", "docker", "infrastructure as code", "linux"
        ],
        "good_reasoning_keywords": [
            "underqualified", "missing skills", "no experience", "does not meet",
            "lacks", "insufficient", "mismatch", "not relevant"
        ],
        "job_description": """
Position: Senior DevOps / Platform Engineer
Company: CloudShift Infrastructure
Location: Austin, TX (Hybrid)

About the Role:
CloudShift is hiring a Senior DevOps Engineer to own and evolve our cloud infrastructure
across AWS and GCP. You will build robust CI/CD pipelines, manage Kubernetes clusters,
and drive Infrastructure as Code (IaC) practices across engineering teams.

Key Responsibilities:
- Design and manage Kubernetes clusters (EKS/GKE) in production
- Build and maintain CI/CD pipelines using GitHub Actions and Jenkins
- Write and maintain Terraform modules for infrastructure provisioning
- Implement security best practices: IAM policies, network segmentation, secrets management
- On-call rotation for infrastructure incidents; drive post-mortems

Required Qualifications:
- 5+ years in DevOps, SRE, or Platform Engineering roles
- Expert-level knowledge of AWS or GCP
- Strong Kubernetes administration skills (CKA preferred)
- Proficiency in Terraform or Pulumi
- Deep Linux sysadmin background
- Experience with monitoring stacks (Prometheus, Grafana, ELK)

Nice to Have:
- Certified Kubernetes Administrator (CKA)
- AWS Solutions Architect certification
- Experience with service mesh (Istio or Linkerd)
        """.strip(),

        "resume": """
MICHAEL JOHNSON
Chicago, IL | mjohnson@email.com | linkedin.com/in/mjohnson

SUMMARY
Enthusiastic web developer with 2 years of experience building responsive websites
and e-commerce platforms. Passionate about front-end design and user experience.
Recently completed an online course in cloud computing fundamentals.

EXPERIENCE

Junior Web Developer — BrightPixel Agency (2022–Present)
- Built and maintained 15+ client websites using HTML, CSS, JavaScript, and WordPress
- Integrated third-party APIs (Stripe, Mailchimp) into web applications
- Collaborated with designers to translate Figma mockups into responsive pages
- Optimized page load speeds; improved Lighthouse scores from 62 to 89 on average

Freelance Web Designer (2021–2022)
- Designed and delivered 8 custom WordPress sites for local small businesses
- Managed hosting setup via cPanel; basic familiarity with FTP and shared hosting
- Created graphics using Canva and Adobe XD

EDUCATION
B.A. Graphic Design — DePaul University, 2021

Online Courses:
- AWS Cloud Practitioner Essentials (Coursera, 2023) — not yet certified
- Introduction to Docker (Udemy, 2023)

SKILLS
Languages: HTML, CSS, JavaScript, PHP (basic)
Tools: WordPress, Figma, Adobe XD, Canva, Git (basic)
Cloud: Familiar with AWS console basics
        """.strip(),
    },

    # ─── TASK 3: HARD — Partial / Ambiguous Match ─────────────────────────────
    "task_hard": {
        "task_id": "task_hard",
        "difficulty": "hard",
        "expected_decision": "shortlist",
        "required_skills": [
            "product management", "roadmap", "agile", "stakeholder",
            "saas", "b2b", "metrics", "analytics"
        ],
        "good_reasoning_keywords": [
            "partially qualified", "some experience", "potential", "transferable",
            "mixed", "ambiguous", "further evaluation", "shortlist", "assess"
        ],
        "job_description": """
Position: Senior Product Manager — B2B SaaS Platform
Company: Nexus Workflow Solutions
Location: New York, NY (On-site preferred)

About the Role:
We are looking for an experienced Senior Product Manager to own the roadmap for our
B2B SaaS platform used by 500+ enterprise clients. You will work closely with
engineering, design, sales, and customer success to ship high-impact features.

Key Responsibilities:
- Define and prioritize the product roadmap in alignment with business strategy
- Conduct customer discovery interviews; translate feedback into product requirements
- Write detailed PRDs and user stories; collaborate with engineering on delivery
- Own product metrics: activation rate, NRR, feature adoption, churn indicators
- Partner with sales and marketing for go-to-market planning
- Lead sprint reviews and quarterly business reviews with executive stakeholders

Required Qualifications:
- 5+ years of product management experience, with at least 2 years in B2B SaaS
- Proven track record of shipping features used by enterprise clients
- Strong analytical skills; comfort with product analytics tools (Mixpanel, Amplitude)
- Experience running Agile/Scrum ceremonies
- Excellent written and verbal communication

Nice to Have:
- MBA or equivalent business education
- Experience with Salesforce CRM integrations or enterprise workflow automation
- Background in customer success or solutions engineering
        """.strip(),

        "resume": """
PRIYA NAIR
Boston, MA | priya.nair@email.com | linkedin.com/in/priyanair

SUMMARY
Product-minded software engineer transitioning into product management, with 5 years
of engineering experience at a B2B SaaS startup and 1 year in an unofficial
"product lead" hybrid role. Strong analytical background; comfortable with data.

EXPERIENCE

Hybrid Product Lead / Senior Software Engineer — TaskFlow Inc. (2022–Present)
- Took on informal PM responsibilities when the PM team was restructured
- Defined feature requirements for 3 major releases; wrote PRDs reviewed by CTO
- Facilitated sprint planning and retrospectives for a team of 8 engineers
- Coordinated with customer success to collect enterprise client feedback;
  identified top 5 feature requests driving churn risk
- Tracked feature adoption in Mixpanel; presented monthly dashboards to leadership
- Also continued engineering work: built backend APIs in Node.js for core workflow features

Senior Software Engineer — TaskFlow Inc. (2020–2022)
- Developed and maintained core B2B SaaS platform features serving 200+ clients
- Participated in client discovery calls alongside sales; provided technical scoping
- Contributed to roadmap discussions; advocated for technical debt reduction

Software Engineer — MarketReach (2018–2020)
- Built internal analytics dashboards; worked with marketing on campaign performance data
- No PM experience in this role

EDUCATION
B.S. Computer Science — Northeastern University, 2018

Certifications:
- Pragmatic Institute Product Management Foundations (2023)

SKILLS
Product: PRD writing, user story mapping, Agile/Scrum, Mixpanel, JIRA, roadmapping (informal)
Engineering: Node.js, Python, PostgreSQL, AWS, REST APIs, Git
Soft Skills: Cross-functional collaboration, stakeholder communication
        """.strip(),
    },
}


def get_task(task_id: str) -> Dict[str, Any]:
    """Return a task definition by ID. Raises KeyError if not found."""
    if task_id not in TASKS:
        raise KeyError(f"Task '{task_id}' not found. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_tasks():
    """Return a summary list of all available tasks."""
    return [
        {
            "task_id": t["task_id"],
            "difficulty": t["difficulty"],
            "expected_decision": t["expected_decision"],
        }
        for t in TASKS.values()
    ]
