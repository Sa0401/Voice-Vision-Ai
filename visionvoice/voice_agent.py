import json
import streamlit as st
from streamlit.components.v1 import html as component_html

class BrowserVoiceAgent:
    """Manages browser-based text-to-speech delivery with customizable voice and rate."""

    def __init__(self, voice_preference: str = "", rate: float = 1.15):
        self.voice_preference = voice_preference
        self.rate = rate

    def speak(self, alerts, enabled: bool) -> None:
        """Speak in the browser that opened the app, including remote phones."""
        if not enabled or not alerts:
            return
        
        alert_id, _, message = alerts[0]
        if st.session_state.get("last_browser_alert_id") == alert_id:
            return
        
        st.session_state.last_browser_alert_id = alert_id
        speech = json.dumps(message)
        
        if self.voice_preference:
            pref = json.dumps(self.voice_preference)
            js_code = f"""
            <script>
              const text = {speech};
              if ("speechSynthesis" in window && text) {{
                const play = () => {{
                    window.speechSynthesis.cancel();
                    const utterance = new SpeechSynthesisUtterance(text);
                    utterance.rate = {self.rate};
                    const voices = window.speechSynthesis.getVoices();
                    const preferredVoice = voices.find(v => v.name.toLowerCase().includes({pref}.toLowerCase()));
                    if (preferredVoice) {{
                        utterance.voice = preferredVoice;
                    }} else {{
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
        else:
            js_code = f"""
            <script>
              const text = {speech};
              if ("speechSynthesis" in window && text) {{
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = {self.rate};
                utterance.lang = navigator.language || "en-US";
                window.speechSynthesis.speak(utterance);
              }}
            </script>
            """

        component_html(js_code, height=0)
