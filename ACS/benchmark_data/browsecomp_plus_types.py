from __future__ import annotations

from typing import TypedDict


class EncryptedDocument(TypedDict):
    docid: str
    text: str
    url: str


class Document(TypedDict):
    docid: str
    text: str


class QueryRow(TypedDict):
    query_id: str
    query: str
    answer: str
    evidence_docs: list[EncryptedDocument]
    gold_docs: list[EncryptedDocument]
    negative_docs: list[EncryptedDocument]


class CorpusDocument(TypedDict):
    docid: str
    text: str
    url: str


