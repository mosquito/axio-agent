"""Compatibility exports for the shared credential codec and file helpers."""

from __future__ import annotations

from axio.credentials import CredentialCodec, record_key
from axio.credentials.files import ProtectedDirectory as ProtectedDirectory
from axio.credentials.files import check_file as check_file
from axio.credentials.files import make_directory as make_directory
from axio.credentials.files import open_file as open_file
from axio.credentials.files import sync_directory as sync_directory

from .credentials import CredentialRecord

codec = CredentialCodec(CredentialRecord)
account_key = record_key
decode_record = codec.decode
encode_record = codec.encode
