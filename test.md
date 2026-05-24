Below is a complete runnable Python example that sets up:

* Inbound trunk + dispatch rule (calls go to a room)
* Outbound trunk
* Makes an outbound call into a room

Based on the official `SipService` client implementation in the Python SDK.

Sources:

* [sip_service.py](https://github.com/livekit/python-sdks/blob/84a60c13965a5b06a7c5d15242eaacda84495ca4/livekit-api/livekit/api/sip_service.py?utm_source=chatgpt.com)
* [SIP API reference](https://docs.livekit.io/reference/telephony/sip-api/?utm_source=chatgpt.com)

---

# Install

```bash
pip install livekit-api
```

---

# Set Environment Variables

```bash
export LIVEKIT_URL=https://your-project.livekit.cloud
export LIVEKIT_API_KEY=your_key
export LIVEKIT_API_SECRET=your_secret
```

---

# Full Example

```python
import os
import asyncio

from livekit import api
from livekit.protocol.sip import (
    CreateSIPInboundTrunkRequest,
    SIPInboundTrunkInfo,
    CreateSIPOutboundTrunkRequest,
    SIPOutboundTrunkInfo,
    CreateSIPDispatchRuleRequest,
    SIPDispatchRule,
    SIPDispatchRuleDirect,
    CreateSIPParticipantRequest,
)

LIVEKIT_URL = os.environ["LIVEKIT_URL"]
API_KEY = os.environ["LIVEKIT_API_KEY"]
API_SECRET = os.environ["LIVEKIT_API_SECRET"]

PROVIDER_NUMBER = "+15551234567"   # number from your SIP provider
OUTBOUND_HOST = "sip.provider.com" # your provider SIP host
DEST_COUNTRY = "US"                # ISO 2-letter code

ROOM_NAME = "agent-room"
CALL_TO = "+15559876543"           # number to dial outbound


async def main():
    lkapi = api.LiveKitAPI(
        url=LIVEKIT_URL,
        api_key=API_KEY,
        api_secret=API_SECRET,
    )

    sip = lkapi.sip

    # -----------------------
    # 1. Create inbound trunk
    # -----------------------
    inbound_trunk = await sip.create_inbound_trunk(
        CreateSIPInboundTrunkRequest(
            trunk=SIPInboundTrunkInfo(
                name="my-inbound-trunk",
                numbers=[PROVIDER_NUMBER],
                auth_username="your_sip_username",
                auth_password="your_sip_password",
            )
        )
    )

    print("Inbound trunk created:", inbound_trunk.sip_trunk_id)

    # -----------------------
    # 2. Create dispatch rule
    # -----------------------
    dispatch_rule = await sip.create_dispatch_rule(
        CreateSIPDispatchRuleRequest(
            name="route-to-agent-room",
            trunk_ids=[inbound_trunk.sip_trunk_id],
            rule=SIPDispatchRule(
                dispatch_rule_direct=SIPDispatchRuleDirect(
                    room_name=ROOM_NAME
                )
            )
        )
    )

    print("Dispatch rule created:", dispatch_rule.sip_dispatch_rule_id)

    # -----------------------
    # 3. Create outbound trunk
    # -----------------------
    outbound_trunk = await sip.create_outbound_trunk(
        CreateSIPOutboundTrunkRequest(
            trunk=SIPOutboundTrunkInfo(
                name="my-outbound-trunk",
                address=OUTBOUND_HOST,
                destination_country=DEST_COUNTRY,
                numbers=[PROVIDER_NUMBER],
                auth_username="your_sip_username",
                auth_password="your_sip_password",
            )
        )
    )

    print("Outbound trunk created:", outbound_trunk.sip_trunk_id)

    # -----------------------
    # 4. Make outbound call
    # -----------------------
    participant = await sip.create_sip_participant(
        CreateSIPParticipantRequest(
            sip_trunk_id=outbound_trunk.sip_trunk_id,
            sip_call_to=CALL_TO,
            room_name=ROOM_NAME,
            wait_until_answered=True,
        )
    )

    print(
        "Outbound call started. Participant ID:",
        participant.participant_id
    )


asyncio.run(main())
```

---

# How This Connects to Your Agent

Your LiveKit Agent must join `ROOM_NAME`.

Flow:

* Inbound calls → SIP participant joins `ROOM_NAME`
* Outbound calls → callee joins `ROOM_NAME`
* Your agent talks normally via audio tracks

---

If you want, I can now show you how to modify your existing [LiveKit](https://livekit.io/?utm_source=chatgpt.com) Agent code so it auto-answers inbound calls cleanly.
