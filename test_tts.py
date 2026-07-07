"""
اختبار حقيقي لمعرفة هل الصوت متاح للتوليد الفعلي (TTS) على حسابك ولا لأ.
استبدل API_KEY و VOICE_ID بالقيم بتاعتك.
"""
from elevenlabs import ElevenLabs

API_KEY = "sk_adbbe87d3c1fcb957d70d641b76ff00d0177985812ef820e"
VOICE_ID = "ZqvIIuD5aI9JFejebHiH"

client = ElevenLabs(api_key=API_KEY)

try:
    audio = client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text="اختبار",
        model_id="eleven_turbo_v2_5",
    )
    # لو السطر ده اتنفذ من غير exception، يبقى الصوت شغال فعلاً للتوليد
    with open("test_output.mp3", "wb") as f:
        for chunk in audio:
            f.write(chunk)
    print("✅ الصوت شغال ومتاح للتوليد الفعلي على حسابك")
except Exception as e:
    print("❌ الصوت مش متاح للتوليد على خطتك الحالية")
    print("تفاصيل الخطأ:", e)
