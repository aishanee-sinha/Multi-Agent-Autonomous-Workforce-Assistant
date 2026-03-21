# Slack-Jira integration

## Overview
The Multi-Agent Autonomous Workforce Assistant is a project designed to streamline task management by integrating Slack messages with Jira issue tracking. This system utilizes natural language processing (NLP) techniques to extract actionable items from Slack conversations and create or update Jira tickets accordingly.

## Project Structure
- **DataPreprocessing.ipynb**: This notebook handles the preprocessing of Slack message data, including parsing XML files and cleaning the text for further analysis.
- **DataLabeling.ipynb**: This notebook uses the Gemini model to label Slack messages with task-related information, such as task summaries, assignees, and issue creation dates.
- **FinetuneLLM.ipynb**: This notebook is responsible for fine-tuning a language model (QLoRA) to extract task-related information from Slack messages and format it for Jira.
- **Requirements**: The project requires several Python packages, including `pandas`, `transformers`, `sentence-transformers`, `requests`, and others specified in the notebooks.

## Installation
To set up the project, ensure you have Python installed, then install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage
1. **Data Preprocessing**: Run the `DataPreprocessing.ipynb` notebook to parse Slack message XML files and save the cleaned messages to a CSV file.
2. **Data Labeling**: Use the `DataLabeling.ipynb` notebook to label Slack messages with task-related information using the Gemini model. The labeled data will be saved for further processing.
3. **Fine-tuning the Model**: Execute the `FinetuneLLM.ipynb` notebook to fine-tune the language model on the labeled data.
4. **Creating/Updating Jira Issues**: Use the provided functions to create or update Jira issues based on Slack messages. Ensure that your Jira credentials and project details are set in the environment variables.

## Environment Variables
Set the following environment variables for Jira integration:
- `JIRA_BASE_URL`: Your Jira instance URL (e.g., `https://your-domain.atlassian.net`)
- `JIRA_EMAIL`: Your Jira account email
- `JIRA_API_TOKEN`: Your Jira API token
- `JIRA_PROJECT_KEY`: The key of the Jira project where issues will be created
- `JIRA_ISSUE_TYPE`: The type of issue to create (e.g., Task, Bug)

## Example
To create a new Jira issue from a Slack message, use the following code snippet:

```python
slack_msg = {
    "timestamp": "2025-11-02T08:14:27.439600",
    "text": "<@User> Can you complete the task by tomorrow?"
}

new_key = create_issue(slack_msg)
print("Created:", new_key)
```

