# HealthSense — Automated Clinical Telemetry & Patient Portal SaaS

*HealthSense* is an advanced, production-grade digital health application that transitions clinical telemetry tracking from stateless calculators into a persistent, multi-user **SaaS Patient Portal**. The architecture integrates a hybrid Machine Learning diagnostic pipeline (**Random Forest** and **Gaussian Naive Bayes**) with an evolutionary **Genetic Algorithm** treatment protocol optimizer and an **Explainable AI (XAI)** clinical insights engine.

---

## Core Engine Features

### 1. Persistent SaaS Account Ecosystem
* **Database-Free Session Management**: Utilizes a highly structured JSON local flat-file storage database system (`users.json` and `patient_history.json`) managed with secure password hashing (`Werkzeug`).
* **Dashboard-First Interface**: Patients land directly on an integrated panel displaying localized historical diagnostics, telemetry trend metrics, and lifestyle tracker controls.

### 2. Hybrid AI Predictive Pipeline
* **Primary Diagnosis (Random Forest)**: Evaluates complex, non-linear relationships across metabolic, hematological, and cardiovascular biomarkers to predict patient health classifications (`Healthy`, `Diabetes`, `Heart Disease`, `Anemia`).
* **Probabilistic Scoring (Gaussian Naive Bayes)**: Evaluates the specific category confidence values directly out of model probability distributions to track continuous probability scores.
* **AI Probability Matrix**: Dynamically streams the complete array distribution to a glowing, glassmorphic UI progress panel revealing active trends for all conditions simultaneously.

### 3. Evolutionary Care Protocol (Genetic Algorithm)
* **Evolved Treatment Matrix**: Instantly loops your `TreatmentGeneticAlgorithm` through 50 generations to mathematically adapt a personalized 7-day nutritional, caloric, and activity protocol tailored directly to the patient's BMI and AI diagnosis category.
* **Persistent Tracker**: Integrates checkbox progress persistence allowing patients to check off daily milestones from either the Protocol View or the Home Dashboard.

### 4. Explainable AI (XAI) & Advanced Tracking Analytics
* **Dynamic Age Correlation**: Explicitly calculates metabolic and cardiovascular drift parameters according to the patient's Age, appending descriptive physiological correlation risk notifications onto the home interface.
* **Multi-Metric Trajectory Graphing**: A highly dynamic frontend widget utilizing `Chart.js` allowing patients to select and toggle between longitudinal trend lines and predictive AI projection lines for **Glucose, BMI, Cholesterol, and Blood Pressure**.

### 5. Python-Powered Chatbot with Attachment Parsing
* **Context-Aware LLM**: Backed by Groq's high-speed **Llama-3.1** infrastructure wrapped with strict domain medical guardrails.
* **Native Python File Processing**: Supports direct drag-and-drop or paperclip attachments (`.csv` or `.txt`) inside the chat client. The Flask backend intercepts files via multipart forms, reads text data natively, and feeds full context indexes directly into the model context window.
