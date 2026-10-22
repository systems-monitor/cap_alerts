"""The User-Agent sent to providers: a per-install token plus the project URL (issue #317)."""

import re

from custom_components.cap_alerts import user_agent_for
from custom_components.cap_alerts.const import USER_AGENT_TOKEN_LENGTH

SHAPE = re.compile(r"^HomeAssistant-CAPAlerts/[0-9a-f]{12} \(\+https://github\.com/seevee/cap_alerts\)$")
INSTANCE = "0f7c2a4e9d1b4c6e8a3f5d2b7e9c1a4f"


def test_shape() -> None:
    assert SHAPE.match(user_agent_for(INSTANCE))


def test_stable_per_install() -> None:
    assert user_agent_for(INSTANCE) == user_agent_for(INSTANCE)


def test_distinct_across_installs() -> None:
    assert user_agent_for(INSTANCE) != user_agent_for(INSTANCE[::-1])


def test_uuid_never_appears() -> None:
    ua = user_agent_for(INSTANCE)
    assert INSTANCE not in ua
    assert INSTANCE[:8] not in ua


def test_token_length() -> None:
    token = user_agent_for(INSTANCE).split("/", 1)[1].split(" ", 1)[0]
    assert len(token) == USER_AGENT_TOKEN_LENGTH
