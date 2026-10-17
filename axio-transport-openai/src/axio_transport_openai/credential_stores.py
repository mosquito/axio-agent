"""ChatGPT type bindings for the shared credential backends."""

from __future__ import annotations

from pathlib import Path

from axio.credentials import Credentials
from axio.credentials import FileCredentialStore as FileStore
from axio.credentials import SQLiteCredentialStore as SQLiteStore
from axio.credentials import default_credentials_path as credentials_path

from .credentials import CredentialRecord

LocalCredentialStore = Credentials[CredentialRecord]


class FileCredentialStore(FileStore[CredentialRecord]):
    filename = "chatgpt.auth"


class SQLiteCredentialStore(SQLiteStore[CredentialRecord]):
    filename = "chatgpt.auth"


def default_credentials_path() -> Path:
    return credentials_path(FileCredentialStore.filename)
