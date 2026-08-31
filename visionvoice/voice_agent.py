import os
import streamlit as st

class BrowserVoiceAgent:
    """Manages text-to-speech delivery using the native Mac 'say' command."""

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
        """Speak instantly using the Mac's built-in background speech engine."""
        if not enabled or not alerts:
            return
        
        alert_id, _, message = alerts[0]
        
        # Only speak a new alert
        if st.session_state.get("last_browser_alert_id") == alert_id:
            return
        st.session_state.last_browser_alert_id = alert_id
        
        voice_flag = ""
        if self.voice_preference:
            pref_lower = self.voice_preference.lower()
            mapped_voice = self.voice_map.get(pref_lower)
            if mapped_voice:
                voice_flag = f"-v {mapped_voice}"
                
        # Calculate roughly how fast to speak (default Mac rate is ~175 wpm)
        wpm = int(175 * self.rate)
        
        escaped_message = message.replace('"', '\\"').replace("'", "\\'")
        
        # Kill any currently speaking voice so they don't overlap, then instantly speak the new one in the background!
        os.system("killall say 2>/dev/null")
        os.system(f'say {voice_flag} -r {wpm} "{escaped_message}" &')
