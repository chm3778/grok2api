"""Tests for isReasoning field in chat payload based on think parameter."""

from unittest.mock import patch

from app.services.grok.chat import ChatRequestBuilder


class TestBuildPayloadIsReasoning:
    """Tests that isReasoning field correctly follows the think boolean."""

    @patch("app.services.grok.chat.get_config")
    def test_is_reasoning_true_when_think_explicit_true(self, mock_get_config):
        """isReasoning should be True when think=True is passed explicitly."""
        mock_get_config.return_value = True  # default config value

        payload = ChatRequestBuilder.build_payload(
            message="test",
            model="grok-3",
            mode="MODEL_MODE_FAST",
            think=True,
        )

        assert payload["isReasoning"] is True

    @patch("app.services.grok.chat.get_config")
    def test_is_reasoning_false_when_think_explicit_false(self, mock_get_config):
        """isReasoning should be False when think=False is passed explicitly."""
        mock_get_config.return_value = True  # default config value

        payload = ChatRequestBuilder.build_payload(
            message="test",
            model="grok-3",
            mode="MODEL_MODE_FAST",
            think=False,
        )

        assert payload["isReasoning"] is False

    @patch("app.services.grok.chat.get_config")
    def test_is_reasoning_follows_config_when_think_none_config_true(
        self, mock_get_config
    ):
        """isReasoning should follow config when think=None and config is True."""

        def mock_config(key, default=None):
            if key == "grok.temporary":
                return True
            if key == "grok.thinking":
                return True
            return default

        mock_get_config.side_effect = mock_config

        payload = ChatRequestBuilder.build_payload(
            message="test",
            model="grok-3",
            mode="MODEL_MODE_FAST",
            think=None,
        )

        assert payload["isReasoning"] is True

    @patch("app.services.grok.chat.get_config")
    def test_is_reasoning_follows_config_when_think_none_config_false(
        self, mock_get_config
    ):
        """isReasoning should follow config when think=None and config is False."""

        def mock_config(key, default=None):
            if key == "grok.temporary":
                return True
            if key == "grok.thinking":
                return False
            return default

        mock_get_config.side_effect = mock_config

        payload = ChatRequestBuilder.build_payload(
            message="test",
            model="grok-3",
            mode="MODEL_MODE_FAST",
            think=None,
        )

        assert payload["isReasoning"] is False

    @patch("app.services.grok.chat.get_config")
    def test_is_reasoning_is_always_boolean_never_none(self, mock_get_config):
        """isReasoning should always be a boolean, never None."""

        def mock_config(key, default=None):
            if key == "grok.temporary":
                return True
            if key == "grok.thinking":
                return True
            return default

        mock_get_config.side_effect = mock_config

        # Test with think=None
        payload = ChatRequestBuilder.build_payload(
            message="test",
            model="grok-3",
            mode="MODEL_MODE_FAST",
            think=None,
        )
        assert isinstance(payload["isReasoning"], bool)
        assert payload["isReasoning"] is not None
