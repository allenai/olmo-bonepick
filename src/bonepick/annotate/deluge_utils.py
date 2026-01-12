import re

from lm_deluge.cache import SqliteCache
from lm_deluge import Conversation
from lm_deluge.api_requests.base import APIResponse, RequestContext
from lm_deluge.models import APIModel


class SqliteInvalidableCache(SqliteCache):
    def __init__(self, path: str, cache_key: str = "default", invalidate: bool = False):
        super().__init__(path=path, cache_key=cache_key)
        self.invalidate = invalidate

    def get(self, prompt: Conversation) -> APIResponse | None:
        if self.invalidate:
            return None
        return super().get(prompt)


def _update_gpt5_model_definitions():
    from lm_deluge.models import registry

    # json support is mistakenly disabled for some gpt-5 models
    for model_name in (m for m in registry if m.startswith("gpt-5-")):
        registry[model_name].supports_json = True


def _fix_gpt5_effort_override_chat_api():
    import lm_deluge.api_requests.openai as openai_api_requests

    _build_oa_chat_request_old = openai_api_requests._build_oa_chat_request

    # monkey patch because gpt-5.x models don't support "minimal" reasoning effort, they support "none" instead
    # the library only handles gpt-5.1, but not gpt-5.2 and other gpt-5.x models
    async def _build_oa_chat_request_new(model: APIModel, context: RequestContext):
        request = await _build_oa_chat_request_old(model, context)
        if (
            re.match(r"gpt-5\.\d+", request.get("model", ""))
            and "reasoning_effort" in request
            and request["reasoning_effort"] == "minimal"
        ):
            request["reasoning_effort"] = "none"
        return request

    openai_api_requests._build_oa_chat_request = _build_oa_chat_request_new


def _fix_gpt5_effort_override_responses_api():
    import lm_deluge.api_requests.openai as openai_api_requests

    _build_oa_responses_request_old = openai_api_requests._build_oa_responses_request

    async def _build_oa_responses_request_new(model: APIModel, context: RequestContext):
        request = await _build_oa_responses_request_old(model, context)
        if (
            re.match(r"gpt-5\.\d+", request.get("model", ""))
            and "reasoning_effort" in request
            and request["reasoning_effort"] == "minimal"
        ):
            request["reasoning_effort"] = "none"
        return request

    openai_api_requests._build_oa_responses_request = _build_oa_responses_request_new


def lm_deluge_monkey_patch():
    _update_gpt5_model_definitions()
    _fix_gpt5_effort_override_chat_api()
    _fix_gpt5_effort_override_responses_api()
