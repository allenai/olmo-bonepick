import contextvars

from lm_deluge.cache import SqliteCache
from lm_deluge import Conversation
from lm_deluge.api_requests.base import APIResponse
from lm_deluge.api_requests.context import RequestContext


class SqliteInvalidableCache(SqliteCache):
    def __init__(self, path: str, cache_key: str = "default", invalidate: bool = False):
        super().__init__(path=path, cache_key=cache_key)
        self.invalidate = invalidate

    def get(self, prompt: Conversation) -> APIResponse | None:
        if self.invalidate:
            return None
        return super().get(prompt)


_batch_output_schema = contextvars.ContextVar("_batch_output_schema", default=None)


def _update_gpt5_model_definitions():
    from lm_deluge.models import registry

    # json support is mistakenly disabled for some gpt-5 models
    for model_name in (m for m in registry if m.startswith("gpt-5-")):
        registry[model_name].supports_json = True


def _patch_batch_output_schema():
    _original_init = RequestContext.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        if self.output_schema is None:
            schema = _batch_output_schema.get()
            if schema is not None:
                self.output_schema = schema

    RequestContext.__init__ = _patched_init


def lm_deluge_monkey_patch():
    _update_gpt5_model_definitions()
    _patch_batch_output_schema()
