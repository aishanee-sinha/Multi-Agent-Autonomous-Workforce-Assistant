"""
gmail_history.py — SSM-backed Gmail historyId persistence
==========================================================
Stores the last processed Gmail historyId in SSM Parameter Store so
history.list() always queries from the right checkpoint, not the
historyId in the Pub/Sub notification (which is the current state, not
the previous one).

SSM parameter: /agent/gmail_history_id  (SecureString)

Environment variables required:
    SSM_HISTORY_ID_PARAM  — SSM parameter name (default: /agent/gmail_history_id)
    AWS_DEFAULT_REGION    — e.g. us-east-1
"""

import logging, os
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

SSM_PARAM = os.environ.get("SSM_HISTORY_ID_PARAM", "/agent/gmail_history_id")
_ssm      = boto3.client("ssm")


def get_last_history_id() -> str | None:
    """
    Read the last processed historyId from SSM.
    Returns None if the parameter doesn't exist yet.
    """
    try:
        resp = _ssm.get_parameter(Name=SSM_PARAM, WithDecryption=True)
        value = resp["Parameter"]["Value"]
        logger.info("gmail_history: loaded historyId=%s from SSM", value)
        return value
    except ClientError as e:
        if e.response["Error"]["Code"] == "ParameterNotFound":
            logger.warning("gmail_history: SSM param %s not found — first run?", SSM_PARAM)
            return None
        raise


def set_last_history_id(history_id: str) -> None:
    """
    Write the latest historyId to SSM after successful processing.
    Creates the parameter if it doesn't exist.
    """
    _ssm.put_parameter(
        Name      = SSM_PARAM,
        Value     = str(history_id),
        Type      = "SecureString",
        Overwrite = True,
    )
    logger.info("gmail_history: saved historyId=%s to SSM", history_id)
