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

- You can run each notebook pipeline_email.ipynb get the output .ics file for the calendarAPI.ipynb file. The artifacts needed for running the pipeline_email.ipynb https://drive.google.com/drive/folders/1lkh8gaYWYcTcNW79zoDw1cb3JOPwsswp?usp=sharing  are in this drive link.
- doenload the model from this link , give this as the model path for pipleine_email file.
