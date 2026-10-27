import json
import os
import shutil
import sys
import streamlit as st
from streamlit.components.v1 import html as component_html


class BrowserVoiceAgent:
    """Manages browser-based text-to-speech delivery with fallback to local Mac speech."""

    def __init__(self, voice_preference: str = "", rate: float = 1.15):
        self.voice_preference = voice_preference
        self.rate = rate
        self.voice_map = {
            "samantha": "Samantha",
            "alex": "Alex",
            "daniel": "Daniel",
            "google uk english male": "Daniel",
            "google uk english female": "Samantha",
            "google us english": "Alex",
        }

    def speak(self, alerts, enabled: bool) -> None:
        """Speak in the connected browser, with native fallback when running locally on macOS."""
        if not enabled or not alerts:
            return

        alert_id, _, message = alerts[0]

        # Only speak a new alert
        if st.session_state.get("last_browser_alert_id") == alert_id:
            return
        st.session_state.last_browser_alert_id = alert_id

        # 1. Native macOS fallback for local execution
        if sys.platform == "darwin" and shutil.which("say"):
            voice_flag = ""
            if self.voice_preference:
                pref_lower = self.voice_preference.lower()
                mapped_voice = self.voice_map.get(pref_lower)
                if mapped_voice:
                    voice_flag = f"-v {mapped_voice}"

            wpm = int(175 * self.rate)
            escaped_message = message.replace('"', '\\"').replace("'", "\\'")
            os.system("killall say 2>/dev/null")
            os.system(f'say {voice_flag} -r {wpm} "{escaped_message}" &')

        # 2. In-browser Web Speech API (works on Streamlit Cloud, mobile devices, remote browsers)
        speech = json.dumps(message)
        pref = json.dumps(self.voice_preference) if self.voice_preference else '""'
        js_code = f"""
        <script>
          const text = {speech};
          const prefVoice = {pref};
          if ("speechSynthesis" in window && text) {{
            const play = () => {{
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = {self.rate};
                const voices = window.speechSynthesis.getVoices();
                if (prefVoice) {{
                    const preferred = voices.find(v => v.name.toLowerCase().includes(prefVoice.toLowerCase()));
                    if (preferred) {{
                        utterance.voice = preferred;
                    }}
                }}
                if (!utterance.voice) {{
                    utterance.lang = navigator.language || "en-US";
                }}
                window.speechSynthesis.speak(utterance);
            }};

            if (window.speechSynthesis.getVoices().length > 0) {{
                play();
            }} else {{
                window.speechSynthesis.onvoiceschanged = play;
            }}
          }}
        </script>
        """
        component_html(js_code, height=0)
