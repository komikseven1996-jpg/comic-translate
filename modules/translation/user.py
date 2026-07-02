import requests
import base64
import numpy as np
import logging
import imkit as imk
import time
from typing import Any, List, Optional

from .base import TranslationEngine
from modules.utils.textblock import TextBlock 
from modules.utils.language_utils import resolve_auto_source_language
from modules.utils.exceptions import InsufficientCreditsException, ContentFlaggedException
from modules.utils.platform_utils import get_client_os


logger = logging.getLogger(__name__)


class UserTranslator(TranslationEngine):
    """
    Desktop translation engine that proxies requests to the web API endpoint.
    Note: Requires login/account on comic-translate.com to function.
    """

    def __init__(self, api_url: str = ""):
        self.api_url = api_url
        self.source_lang: str = None
        self.target_lang: str = None
        self.translator_key: str = None 
        self.settings = None 
        self.is_llm: bool = False
        self._session = requests.Session()
        self._profile_web_api = False

    def initialize(self, settings, source_lang: str, target_lang: str, translator_key: str, **kwargs) -> None:
        self.settings = settings
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.translator_key = translator_key
        self.is_llm = self._check_is_llm(translator_key)

    def _check_is_llm(self, translator_key: str) -> bool:
        llm_ids = ["GPT", "Claude", "Gemini", "Deepseek"]
        return any(identifier in translator_key for identifier in llm_ids)
    
    def _get_access_token(self) -> Optional[str]:
        """Retrieves the access token. Requires login to comic-translate.com."""
        try:
            if self.settings and hasattr(self.settings, 'auth_client'):
                if not self.settings.auth_client.validate_token():
                    logger.error("Access token invalid and refresh failed.")
                    return None
                try:
                    from app.account.auth.token_storage import get_token
                    token = get_token("access_token")
                    if not token:
                        logger.warning("Access token not found.")
                        return None
                    return token
                except ImportError:
                    logger.error("UserTranslator: Auth module not available. Web API requires account login.")
                    return None
            else:
                logger.warning("UserTranslator: No authentication available. Web API requires login to comic-translate.com.")
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve access token: {e}")
            return None

    def translate(self, blk_list: List[TextBlock], image: np.ndarray = None, extra_context: str = "") -> List[TextBlock]:
        start_t = time.perf_counter()
        logger.info(f"UserTranslator: Translating via web API ({self.api_url}) for {self.translator_key}")

        access_token = self._get_access_token()
        after_token_t = time.perf_counter()

        texts_payload = []
        for i, blk in enumerate(blk_list):
            block_id = getattr(blk, 'id', i)
            texts_payload.append({"id": block_id, "text": blk.text})

        llm_options_payload = None
        if self.is_llm and self.settings:
            llm_settings = self.settings.get_llm_settings()
            llm_options_payload = {
                "image_input_enabled": llm_settings.get('image_input_enabled', False)
            }

        image_base64_payload = None
        should_send_image = (
            self.is_llm
            and image is not None
            and llm_options_payload
            and llm_options_payload.get("image_input_enabled")
        )
         
        if should_send_image:
            buffer = imk.encode_image(image, "jpg")
            image_base64_payload = base64.b64encode(buffer).decode('utf-8')
            logger.debug("UserTranslator: Encoded image for web API request.")
        after_encode_t = time.perf_counter()

        api_source_language = resolve_auto_source_language(blk_list, self.source_lang)

        request_payload = {
            "translator": self.translator_key,
            "source_language": api_source_language,
            "target_language": self.target_lang,
            "texts": texts_payload,
        }

        if image_base64_payload is not None:
            request_payload["image_base64"] = image_base64_payload
        if llm_options_payload is not None:
            request_payload["llm_options"] = llm_options_payload
            request_payload["extra_context"] = extra_context

        client_os = get_client_os()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
            "X-Client-OS": client_os
        }

        response = self._session.post(
            self.api_url, 
            headers=headers, 
            json=request_payload, 
            timeout=120
        ) 
        after_request_t = time.perf_counter()
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            try:
                error_data = response.json()
                detail = error_data.get('detail')
                description = ""

                if isinstance(detail, dict):
                    description = detail.get('error_description') or detail.get('message')
                    if not description and detail.get('type'):
                        description = f"Error type: {detail.get('type')}"
                elif isinstance(detail, list):
                    msgs = []
                    for err in detail:
                        loc = ".".join(str(x) for x in err.get('loc', []))
                        msg = err.get('msg', '')
                        msgs.append(f"{loc}: {msg}")
                    description = "; ".join(msgs)
                else:
                    description = str(detail) if detail else ""

                if response.status_code == 402:
                    if isinstance(detail, dict) and detail.get('type') == 'INSUFFICIENT_CREDITS':
                        raise InsufficientCreditsException(description)
                    raise InsufficientCreditsException(description)
                
                if response.status_code == 400:
                    is_flagged = False
                    if isinstance(detail, dict) and detail.get('type') == 'CONTENT_FLAGGED_UNSAFE':
                        is_flagged = True
                    elif "flagged as unsafe" in str(description).lower() or "blocked by" in str(description).lower():
                        is_flagged = True
                    
                    if is_flagged:
                        raise ContentFlaggedException(description, context="Translation")

                if description:
                    raise Exception(f"Server Error ({response.status_code}): {description}") from e

            except ValueError:
                pass
            raise e

        if response.status_code == 200:
            response_data = response.json()
            translations_map = {item['id']: item['translation'] for item in response_data.get('translations', [])}
            credits_info = response_data.get('credits') or response_data.get('credits_remaining')

            logger.info(f"UserTranslator: Received successful response from web API. Credits: {credits_info}")

            for i, blk in enumerate(blk_list):
                block_id = getattr(blk, 'id', i)
                blk.translation = translations_map.get(block_id, "")

            self.update_credits(credits_info)

        if self._profile_web_api:
            total_t = time.perf_counter() - start_t
            server_ms = response.headers.get("X-CT-Server-Duration-Ms")
            logger.info(
                "UserTranslator timings: token=%.3fs encode=%.3fs http=%.3fs total=%.3fs (texts=%d image=%s server_ms=%s)",
                after_token_t - start_t,
                after_encode_t - after_token_t,
                after_request_t - after_encode_t,
                total_t,
                len(blk_list),
                "yes" if should_send_image else "no",
                server_ms,
            )
            print(
                f"UserTranslator timings: token={after_token_t - start_t:.3f}s "
                f"encode={after_encode_t - after_token_t:.3f}s http={after_request_t - after_encode_t:.3f}s "
                f"total={total_t:.3f}s (texts={len(blk_list)} image={'yes' if should_send_image else 'no'} server_ms={server_ms})"
            )

        return blk_list
    
    def update_credits(self, credits: Optional[Any]) -> None:
        if credits is None:
            return
        if not self.settings:
            return
        if isinstance(credits, dict):
            self.settings.user_credits = credits
        else:
            try:
                total = int(credits)
                self.settings.user_credits = {
                    'subscription': None,
                    'one_time': total,
                    'total': total,
                }
            except Exception:
                logger.warning(f"UserTranslator: Unexpected credits format: {credits}")
                return

        if hasattr(self.settings, '_save_user_info_to_settings'):
            self.settings._save_user_info_to_settings()
