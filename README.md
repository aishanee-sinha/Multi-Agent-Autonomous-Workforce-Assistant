# Email Pipeline & Calendar API – Notebooks

A compact README summarizing the two notebooks in this repository.

## Contents

### `calendarAPI.ipynb`
- **Title:** calendarAPI
- **Kernel/Language:** Python 3 (python)
- **Notes:**
  - Uses Google APIs (auth/discovery).

### `pipeline_email.ipynb`
- **Title:** pipeline_email
- **Kernel/Language:** Python 3 (python)

## Quick Start

1. **Create a virtual environment**
   ```bash
   python -m venv .venv && source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install dateutil google googleapiclient icalendar peft pytz torch traceback transformers
   ```

3. **Open the notebooks**
   ```bash
   pip install notebook  # if you don't have Jupyter
   jupyter notebook
   ```

4. **Credentials & environment**
   No environment variables were auto-detected. If these notebooks use API keys or credentials, set them accordingly.

## Dependencies (detected)

  - dateutil
  - google
  - googleapiclient
  - icalendar
  - peft
  - pytz
  - torch
  - traceback
  - transformers

## Execution Order

- You can run each notebook independently. If they share artifacts (e.g., tokens, cached files), start with `calendarAPI.ipynb` if your workflow involves fetching calendar data before email processing.
