"""Tests for UltraFineCommitNormalizer."""

import pytest

from bonepick.data.normalizers import get_normalizer


@pytest.fixture(scope="module")
def normalizer():
    return get_normalizer("ultrafine_commits")


class TestUltraFineCommitNormalizerMultipleEmails:
    def test_multiple_emails_replaced_and_tokenized(self, normalizer):
        text = (
            "Author: dev@company.io committed a fix. "
            "Reviewer: qa+test@company.io approved it. "
            "CC: alice.bob@subdomain.example.com"
        )
        result = normalizer.normalize(text)
        assert result == "author : ADDR committed a fix . reviewer : ADDR approved it . cc : ADDR"


class TestUltraFineCommitNormalizerMultipleURLs:
    def test_multiple_urls_replaced_and_tokenized(self, normalizer):
        text = (
            "See https://github.com/user/repo/pull/42 and http://docs.example.com/api/v2 for the fix description"
        )
        result = normalizer.normalize(text)
        assert result == "see URL and URL for the fix description"


class TestUltraFineCommitNormalizerMultipleHex:
    def test_commit_hashes_and_uuids_replaced(self, normalizer):
        text = "Merge abc123def456789a into 00112233-4455-6677-8899-aabbccddeeff and revert fedcba9876543210"
        result = normalizer.normalize(text)
        assert result == "merge CODE into CODE and revert CODE"
