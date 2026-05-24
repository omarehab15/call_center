import argparse
import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
from livekit import api
from livekit.api.sip_service import SipService
from livekit.protocol.sip import (
    CreateSIPDispatchRuleRequest,
    CreateSIPInboundTrunkRequest,
    CreateSIPOutboundTrunkRequest,
    CreateSIPParticipantRequest,
    ListSIPDispatchRuleRequest,
    ListSIPInboundTrunkRequest,
    ListSIPOutboundTrunkRequest,
    SIPDispatchRule,
    SIPDispatchRuleDirect,
    SIPDispatchRuleInfo,
    SIPInboundTrunkInfo,
    SIPOutboundTrunkInfo,
)


@dataclass
class SIPConfig:
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    provider_number: str
    outbound_host: str
    destination_country: str
    room_name: str
    sip_auth_username: Optional[str]
    sip_auth_password: Optional[str]
    inbound_trunk_name: str = "my-inbound-trunk"
    outbound_trunk_name: str = "my-outbound-trunk"
    dispatch_rule_name: str = "route-to-agent-room"
    call_to: Optional[str] = None
    wait_until_answered: bool = True


def _load_env_files() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent

    for env_file in (
        repo_root / ".env.local",
        repo_root / ".env",
        here / ".env.local",
        here / ".env",
    ):
        if env_file.exists():
            load_dotenv(env_file, override=False)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return value.strip()


def _parse_bool(value: Optional[str], *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean value: {value!r}. Use one of true/false, yes/no, 1/0."
    )


def _normalize_livekit_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme == "ws":
        return urlunparse(parsed._replace(scheme="http"))
    if parsed.scheme == "wss":
        return urlunparse(parsed._replace(scheme="https"))
    return url


def _build_config(args: argparse.Namespace) -> SIPConfig:
    call_to = args.call_to if args.call_to is not None else os.getenv("SIP_CALL_TO")
    room_name = (
        args.room_name
        if args.room_name is not None
        else os.getenv("SIP_ROOM_NAME", "agent-room")
    )

    return SIPConfig(
        livekit_url=_normalize_livekit_url(_require_env("LIVEKIT_URL")),
        livekit_api_key=_require_env("LIVEKIT_API_KEY"),
        livekit_api_secret=_require_env("LIVEKIT_API_SECRET"),
        provider_number=_require_env("SIP_PROVIDER_NUMBER"),
        outbound_host=_require_env("SIP_OUTBOUND_HOST"),
        destination_country=os.getenv("SIP_DESTINATION_COUNTRY", "US").upper(),
        room_name=room_name,
        sip_auth_username=(os.getenv("SIP_AUTH_USERNAME") or "").strip() or None,
        sip_auth_password=(os.getenv("SIP_AUTH_PASSWORD") or "").strip() or None,
        inbound_trunk_name=os.getenv("SIP_INBOUND_TRUNK_NAME", "my-inbound-trunk"),
        outbound_trunk_name=os.getenv("SIP_OUTBOUND_TRUNK_NAME", "my-outbound-trunk"),
        dispatch_rule_name=os.getenv("SIP_DISPATCH_RULE_NAME", "route-to-agent-room"),
        call_to=call_to,
        wait_until_answered=_parse_bool(
            os.getenv("SIP_WAIT_UNTIL_ANSWERED"), default=True
        ),
    )


async def _find_inbound_trunk_by_name(sip_client: SipService, name: str):
    response = await sip_client.list_inbound_trunk(ListSIPInboundTrunkRequest())
    return next((item for item in response.items if item.name == name), None)


async def _find_outbound_trunk_by_name(sip_client: SipService, name: str):
    response = await sip_client.list_outbound_trunk(ListSIPOutboundTrunkRequest())
    return next((item for item in response.items if item.name == name), None)


async def _find_dispatch_rule_by_name(
    sip_client: SipService, name: str
) -> Optional[SIPDispatchRuleInfo]:
    response = await sip_client.list_dispatch_rule(ListSIPDispatchRuleRequest())
    return next((item for item in response.items if item.name == name), None)


async def ensure_inbound_trunk(
    sip_client: SipService, config: SIPConfig
) -> SIPInboundTrunkInfo:
    existing = await _find_inbound_trunk_by_name(sip_client, config.inbound_trunk_name)
    if existing:
        print(
            f"Using existing inbound trunk '{existing.name}' "
            f"(id={existing.sip_trunk_id})."
        )
        return existing

    trunk_kwargs = {
        "name": config.inbound_trunk_name,
        "numbers": [config.provider_number],
    }
    if config.sip_auth_username:
        trunk_kwargs["auth_username"] = config.sip_auth_username
    if config.sip_auth_password:
        trunk_kwargs["auth_password"] = config.sip_auth_password

    inbound_trunk = await sip_client.create_inbound_trunk(
        CreateSIPInboundTrunkRequest(trunk=SIPInboundTrunkInfo(**trunk_kwargs))
    )
    print(
        f"Created inbound trunk '{inbound_trunk.name}' ({inbound_trunk.sip_trunk_id})."
    )
    return inbound_trunk


async def ensure_dispatch_rule(
    sip_client: SipService,
    config: SIPConfig,
    inbound_trunk_id: str,
) -> SIPDispatchRuleInfo:
    existing = await _find_dispatch_rule_by_name(sip_client, config.dispatch_rule_name)
    if existing:
        direct_room = None
        if existing.rule.HasField("dispatch_rule_direct"):
            direct_room = existing.rule.dispatch_rule_direct.room_name

        if (
            direct_room != config.room_name
            or inbound_trunk_id not in existing.trunk_ids
        ):
            print(
                "Found existing dispatch rule with same name but different room/trunk "
                f"configuration (id={existing.sip_dispatch_rule_id})."
            )
            print(
                "Keeping existing rule as-is. "
                "Rename SIP_DISPATCH_RULE_NAME or delete the rule to recreate it."
            )
        else:
            print(
                f"Using existing dispatch rule '{existing.name}' "
                f"(id={existing.sip_dispatch_rule_id})."
            )
        return existing

    dispatch_rule = await sip_client.create_dispatch_rule(
        CreateSIPDispatchRuleRequest(
            name=config.dispatch_rule_name,
            trunk_ids=[inbound_trunk_id],
            rule=SIPDispatchRule(
                dispatch_rule_direct=SIPDispatchRuleDirect(room_name=config.room_name)
            ),
        )
    )
    print(
        f"Created dispatch rule '{dispatch_rule.name}' "
        f"({dispatch_rule.sip_dispatch_rule_id})."
    )
    return dispatch_rule


async def ensure_outbound_trunk(
    sip_client: SipService, config: SIPConfig
) -> SIPOutboundTrunkInfo:
    existing = await _find_outbound_trunk_by_name(
        sip_client, config.outbound_trunk_name
    )
    if existing:
        print(
            f"Using existing outbound trunk '{existing.name}' "
            f"(id={existing.sip_trunk_id})."
        )
        return existing

    trunk_kwargs = {
        "name": config.outbound_trunk_name,
        "address": config.outbound_host,
        "destination_country": config.destination_country,
        "numbers": [config.provider_number],
    }
    if config.sip_auth_username:
        trunk_kwargs["auth_username"] = config.sip_auth_username
    if config.sip_auth_password:
        trunk_kwargs["auth_password"] = config.sip_auth_password

    outbound_trunk = await sip_client.create_outbound_trunk(
        CreateSIPOutboundTrunkRequest(trunk=SIPOutboundTrunkInfo(**trunk_kwargs))
    )
    print(
        f"Created outbound trunk '{outbound_trunk.name}' "
        f"({outbound_trunk.sip_trunk_id})."
    )
    return outbound_trunk


async def start_outbound_call(
    sip_client: SipService,
    config: SIPConfig,
    outbound_trunk_id: str,
    call_to: str,
) -> None:
    participant = await sip_client.create_sip_participant(
        CreateSIPParticipantRequest(
            sip_trunk_id=outbound_trunk_id,
            sip_call_to=call_to,
            room_name=config.room_name,
            wait_until_answered=config.wait_until_answered,
        )
    )
    print(
        "Outbound call started. "
        f"Participant ID: {participant.participant_id} "
        f"(room={config.room_name}, to={call_to})"
    )


async def run_setup(config: SIPConfig, *, place_call: bool) -> None:
    lkapi = api.LiveKitAPI(
        url=config.livekit_url,
        api_key=config.livekit_api_key,
        api_secret=config.livekit_api_secret,
    )
    try:
        sip_client = lkapi.sip
        inbound_trunk = await ensure_inbound_trunk(sip_client, config)
        await ensure_dispatch_rule(sip_client, config, inbound_trunk.sip_trunk_id)
        outbound_trunk = await ensure_outbound_trunk(sip_client, config)

        if place_call:
            if not config.call_to:
                raise ValueError(
                    "No call destination set. Provide --call-to or SIP_CALL_TO."
                )
            await start_outbound_call(
                sip_client=sip_client,
                config=config,
                outbound_trunk_id=outbound_trunk.sip_trunk_id,
                call_to=config.call_to,
            )
    finally:
        await lkapi.aclose()


async def run_call_only(config: SIPConfig) -> None:
    if not config.call_to:
        raise ValueError("No call destination set. Provide --call-to or SIP_CALL_TO.")

    lkapi = api.LiveKitAPI(
        url=config.livekit_url,
        api_key=config.livekit_api_key,
        api_secret=config.livekit_api_secret,
    )
    try:
        sip_client = lkapi.sip
        outbound_trunk = await ensure_outbound_trunk(sip_client, config)
        await start_outbound_call(
            sip_client=sip_client,
            config=config,
            outbound_trunk_id=outbound_trunk.sip_trunk_id,
            call_to=config.call_to,
        )
    finally:
        await lkapi.aclose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Provision LiveKit SIP trunks/dispatch rules and place outbound calls."
    )
    subparsers = parser.add_subparsers(dest="command")

    setup_parser = subparsers.add_parser(
        "setup",
        help="Create/reuse inbound trunk, dispatch rule, and outbound trunk.",
    )
    setup_parser.add_argument(
        "--call-now",
        action="store_true",
        help="After setup, place an outbound call using SIP_CALL_TO or --call-to.",
    )
    setup_parser.add_argument(
        "--call-to",
        default=None,
        help="Phone number to dial (E.164 format). Overrides SIP_CALL_TO.",
    )
    setup_parser.add_argument(
        "--room-name",
        default=None,
        help="Room name to route calls to. Overrides SIP_ROOM_NAME.",
    )

    call_parser = subparsers.add_parser(
        "call",
        help="Place an outbound call using an existing (or newly created) outbound trunk.",
    )
    call_parser.add_argument(
        "--call-to",
        default=None,
        help="Phone number to dial (E.164 format). Overrides SIP_CALL_TO.",
    )
    call_parser.add_argument(
        "--room-name",
        default=None,
        help="Room name for the SIP participant. Overrides SIP_ROOM_NAME.",
    )

    parser.set_defaults(command="setup", call_now=False)
    return parser


async def main() -> None:
    _load_env_files()
    args = build_parser().parse_args()
    config = _build_config(args)

    if args.command == "call":
        await run_call_only(config)
        return

    await run_setup(config, place_call=bool(args.call_now))


if __name__ == "__main__":
    asyncio.run(main())
