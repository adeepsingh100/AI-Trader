from __future__ import annotations

import uuid
from unittest.mock import MagicMock

from firebase_admin import firestore


# --- Firestore fake (Neon→Firebase migration — see src/db/models.py's
# module docstring). The psycopg2-shaped fakes this file used to also
# export (_fake_connection/_fake_cursor/_last_execute/_inserted_row/
# _updated_row) were deleted once Phase 3 removed the last function in
# models.py that talked to Postgres — nothing in the suite mocks
# get_client anymore, there's no get_client left to mock. ---


def _matches_filter(data: dict, field: str, op: str, value) -> bool:
    actual = data.get(field)
    if op == "==":
        return actual == value
    if op == ">=":
        return actual is not None and actual >= value
    if op == "<=":
        return actual is not None and actual <= value
    if op == "<":
        return actual is not None and actual < value
    if op == "in":
        return actual in value
    raise NotImplementedError(f"fake firestore: unsupported op {op!r}")


class _FakeDocSnapshot:
    def __init__(self, doc_id, data, store=None, collection=None):
        self.id = doc_id
        self._data = data
        self.exists = data is not None
        # store/collection are None for snapshots built outside a query
        # (e.g. a doc_ref.get() in code that never touches .reference) --
        # only the purge path's batch.delete(doc.reference) needs this.
        self.reference = _FakeDocRef(store, collection, doc_id) if store is not None else None

    def to_dict(self):
        return dict(self._data) if self._data is not None else None

    def get(self, field):
        return (self._data or {}).get(field)


class _FakeDocRef:
    def __init__(self, store, collection, doc_id):
        self._store = store
        self._collection = collection
        self.id = doc_id

    def get(self, transaction=None):
        data = self._store.setdefault(self._collection, {}).get(self.id)
        return _FakeDocSnapshot(self.id, data, self._store, self._collection)

    def set(self, fields, merge=False):
        coll = self._store.setdefault(self._collection, {})
        if merge and self.id in coll:
            coll[self.id] = {**coll[self.id], **fields}
        else:
            coll[self.id] = dict(fields)

    def delete(self):
        self._store.get(self._collection, {}).pop(self.id, None)


class _FakeQuery:
    def __init__(self, store, collection, filters=None, order_field=None, desc=False, limit=None):
        self._store = store
        self._collection = collection
        self._filters = filters or []
        self._order_field = order_field
        self._desc = desc
        self._limit = limit

    def where(self, *, filter):
        return _FakeQuery(
            self._store, self._collection,
            self._filters + [(filter.field_path, filter.op_string, filter.value)],
            self._order_field, self._desc, self._limit,
        )

    def order_by(self, field, direction=None):
        return _FakeQuery(
            self._store, self._collection, self._filters, field,
            direction == firestore.Query.DESCENDING, self._limit,
        )

    def limit(self, n):
        return _FakeQuery(self._store, self._collection, self._filters, self._order_field, self._desc, n)

    def stream(self):
        docs = self._store.get(self._collection, {})
        matches = [
            (doc_id, data) for doc_id, data in docs.items()
            if all(_matches_filter(data, f, op, v) for f, op, v in self._filters)
        ]
        if self._order_field:
            matches.sort(key=lambda item: item[1].get(self._order_field), reverse=self._desc)
        if self._limit is not None:
            matches = matches[: self._limit]
        return [_FakeDocSnapshot(doc_id, data, self._store, self._collection) for doc_id, data in matches]


class _FakeCollectionRef(_FakeQuery):
    def document(self, doc_id=None):
        if doc_id is None:
            # A per-instance counter would reset every fresh
            # client.collection(...) call (real code paths call it once
            # per insert, not once per test) and silently collide multiple
            # inserts into the same doc ID -- a real uuid, same as
            # Firestore's own auto-ID, is the only correct fake here.
            doc_id = f"auto_{uuid.uuid4().hex[:16]}"
        return _FakeDocRef(self._store, self._collection, doc_id)


class _FakeBatch:
    def __init__(self):
        self._ops = []

    def set(self, doc_ref, fields):
        self._ops.append(("set", doc_ref, fields))

    def delete(self, doc_ref):
        self._ops.append(("delete", doc_ref, None))

    def commit(self):
        for op, doc_ref, fields in self._ops:
            doc_ref.set(fields) if op == "set" else doc_ref.delete()
        self._ops = []


class _FakeFirestoreClient:
    def __init__(self, store):
        self._store = store

    def collection(self, name):
        return _FakeCollectionRef(self._store, name)

    def get_all(self, refs):
        return [ref.get() for ref in refs]

    def batch(self):
        return _FakeBatch()

    def transaction(self):
        # Not exercised by unit tests — try_acquire_risk_check_lock's
        # @firestore.transactional logic is covered by the real live
        # smoke test (Phase 1 verification), not fakeable meaningfully
        # here without reimplementing Firestore's optimistic-concurrency
        # retry machinery.
        raise NotImplementedError("fake firestore: transactions aren't faked, see docstring")


def _fake_firestore_client(seed: dict | None = None):
    """In-memory Firestore fake — a nested dict store
    ({collection: {doc_id: {field: value}}}), not a MagicMock/call-args
    inspector like _fake_connection above. A Firestore query has real
    filtering/ordering semantics (.where/.order_by/.limit chained calls
    that only resolve on .stream()) that a call-args-only mock can't
    exercise meaningfully — this fake actually stores and filters data.
    `seed` pre-populates the store: {collection: {doc_id: {...}}}.
    monkeypatch this onto models.get_firestore_client; assert against the
    returned `store` dict directly by collection/doc-id/field name — the
    Firestore-shaped equivalent of _inserted_row/_updated_row above."""
    store = {collection: dict(docs) for collection, docs in (seed or {}).items()}
    return _FakeFirestoreClient(store), store
