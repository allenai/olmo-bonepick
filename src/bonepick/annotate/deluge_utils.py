from lm_deluge.cache import SqliteCache
from lm_deluge import Conversation
from lm_deluge.api_requests.base import APIResponse

class SqliteInvalidableCache(SqliteCache):
    def __init__(self, path: str, cache_key: str = "default", invalidate: bool = False):
        super().__init__(path=path, cache_key=cache_key)
        self.invalidate = invalidate

    def get(self, prompt: Conversation) -> APIResponse | None:
        if self.invalidate:
            return None
        return super().get(prompt)
