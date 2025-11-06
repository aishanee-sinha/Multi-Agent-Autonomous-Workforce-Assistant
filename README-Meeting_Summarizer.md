Project Overview
This project develops three autonomous AI agents to address common workplace inefficiencies:

Meeting Summarizer Agent - Automatically generates structured summaries from meeting transcripts
Email Calendar Extractor Agent - Extracts calendar events from emails and creates .ics files
Resume Ranker Agent - Ranks resumes against job descriptions using semantic similarity

Objectives

Reduce time spent on manual documentation tasks
Improve accuracy and consistency of workplace automation
Evaluate multiple AI models to identify optimal solutions
Deliver production-ready, scalable agents

For Meeting Summarizer:

Quick Start
Prerequisites

Python 3.10+
NVIDIA GPU with CUDA support (recommended)
Google Colab account (for cloud execution)
Hugging Face account (for model access)

Installation
bash# Clone repository
git clone https://github.com/yourusername/autonomous-workforce-agents.git
cd autonomous-workforce-agents

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
Running the Agents
Each agent has its own Jupyter notebook:
bash# Meeting Summarizer
jupyter notebook meeting_summarizer/MSADI_Meeting_Summarizer_final_e2e_demo.ipynb

# Email Calendar Extractor
jupyter notebook email_calendar/calendar_extraction.ipynb

# Resume Ranker
jupyter notebook resume_ranker/resume_ranking.ipynb
Or use Google Colab:

Upload notebook to Google Colab
Runtime > Change runtime type > GPU
Run all cells

Project Structure
autonomous-workforce-agents/
├── README.md
├── requirements.txt
├── LICENSE
│
├── meeting_summarizer/
│   ├── MSADI_Meeting_Summarizer_final_e2e_demo.ipynb
│   ├── outputs/
│   │   ├── results/
│   │   └── meetings/
│   └── README.md

Agent Descriptions
1. Meeting Summarizer Agent
Purpose: Generate structured summaries from meeting transcripts
Key Features:

Processes transcripts of 1,200-15,000 words
Generates 5-section summaries (Overview, Key Points, Decisions, Action Items, Next Meeting)
Extracts action items with owners and due dates
Supports markdown, CSV, and iCalendar output formats

Models Evaluated: FLAN-T5-BASE, FLAN-T5-LARGE, BART-CNN, PEGASUS-XSUM
Best Model: FLAN-T5-LARGE (6.40/10 production score, 1.16s processing time)
Dataset: AMI Meeting Corpus (150 samples, 70/15/15 split)

Performance Summary
AgentModelDataset SizePerformance MetricScoreMeeting SummarizerFLAN-T5-LARGE150 samplesProduction Score6.40/10Email CalendarMistral-7B+LoRA537 emailsEvent Extraction AccuracyTBDResume RankerSentence-BERTTBDRanking AccuracyTBD

Configuration
Each agent can be configured by editing the Config class in its notebook:
pythonclass Config:
    # Dataset
    TOTAL_SAMPLES = 150
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Model
    MODEL_NAME = "google/flan-t5-large"
    
    # Output
    OUTPUT_DIR = "outputs/"

Requirements
Core Dependencies
  python>=3.10
  torch==2.2.2
  transformers==4.41.0
  datasets>=2.14.0
  sentence-transformers>=2.2.0

Troubleshooting
Common Issues
  CUDA Out of Memory
  python# Use smaller model or reduce batch size
  torch.cuda.empty_cache()
Version Incompatibility
  bashpip uninstall -y transformers
  pip install transformers==4.41.0
  NLTK Data Missing
  pythonimport nltk
  nltk.download('punkt')
See individual agent READMEs for specific troubleshooting.
