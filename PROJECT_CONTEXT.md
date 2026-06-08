
---

## 15. ملفات SIP المضافة (التحديث الأخير)

### الملفات الجديدة/المعدّلة:

| الملف | التغيير |
|-------|---------|
| `livekit_agent/src/agent.py` | أضفنا `agent_name="inbound-agent"` + `on_enter` + persona system + SIP detection |
| `sip_setup.py` | **ملف جديد** — سكريبت إعداد SIP Trunks + Dispatch Rules (شغّله مرة واحدة) |
| `.env.local.example` | أضفنا SIP + ngrok variables |
| `docker-compose.local.yml` | أضفنا `ngrok` service |
| `start-local.sh` | أضفنا ngrok checks + SIP info |

### متغيرات البيئة الجديدة في `.env.local`:
```env
SIP_PROVIDER_NUMBERS=+966XXXXXXXXX
SIP_OUTBOUND_HOST=your-trunk.pstn.twilio.com
SIP_USERNAME=your_sip_username
SIP_PASSWORD=your_sip_password
SIP_OUTBOUND_TRUNK_ID=          # يتعبى بعد sip_setup.py
NGROK_AUTHTOKEN=your_token
NGROK_DOMAIN=your-name.ngrok-free.app
```

### تسلسل إعداد SIP (خطوة بخطوة):
1. اشتري رقم Twilio + أنشئ SIP Trunk (من docs.livekit.io/telephony/start/providers/twilio/)
2. احفظ: Termination URI + username + password من Twilio
3. اشترك في ngrok.com + احصل على static domain مجاني
4. عبّي `.env.local` بكل القيم الجديدة
5. شغّل `python sip_setup.py` مرة واحدة
6. الـ sip_setup سيطبع الـ Inbound Trunk ID — روح Twilio واضبط Origination URI = `sip:<ngrok-domain>;transport=tcp`
7. شغّل `./start-local.sh` — ngrok هيشتغل تلقائياً مع المشروع
8. اتصل على رقمك وكلّم فهد 🎉

### Personas المتاحة في `agent.py`:
- `default` — الـ assistant العام (فهد)
- `customer_service` — خدمة العملاء
- `sales` — المبيعات

لإضافة persona جديدة: أضف entry في `PERSONAS` dict في `agent.py` وأضف dispatch rule جديد في `sip_setup.py`.
