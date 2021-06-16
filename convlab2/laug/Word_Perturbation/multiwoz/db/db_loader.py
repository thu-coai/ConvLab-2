import re
import os
from functools import lru_cache
from typing import Optional
from abc import ABC
from .db import DB, BaseDB
from ..util import load_json


def list_db_filename(db_dir):
    filenames = os.listdir(db_dir)
    db_filenames = {}
    for filename in filenames:
        match = re.match(r'^(\w+)_db\.json$', filename)
        if match is not None and os.path.isfile(os.path.join(db_dir, filename)):
            domain = match.group(1)
            db_filenames[domain] = filename
    return db_filenames


def list_db_filepath(db_dir):
    return {domain: os.path.join(db_dir, filename) for domain, filename in list_db_filename(db_dir).items()}


class BaseDBLoader(ABC):
    def load_db(self, domain: str, slot: Optional[str] = None) -> Optional[BaseDB]:
        """given a domain and a slot, load corresponding db."""
        raise NotImplementedError


class DBLoader(BaseDBLoader):
    def __init__(self, db_dir):
        assert db_dir and os.path.isdir(db_dir)
        self.db_dir = db_dir
        self.db_files = list_db_filepath(db_dir)  # domain -->  filepath
        self.db_cache = {}  # domain --> List[dict]

    @lru_cache(maxsize=25)
    def _get_db_file(self, domain):
        if domain in self.db_files:
            return self.db_files[domain]
        for dom, filename in self.db_files.items():
            if domain.lower() in dom.lower():
                return filename

    def load_db(self, domain: str, slot: Optional[str] = None) -> Optional[DB]:
        filepath = self._get_db_file(domain)
        if filepath is None:
            return None
        if domain not in self.db_cache:
            self.db_cache[domain] = DB(load_json(filepath))
        return self.db_cache[domain]
