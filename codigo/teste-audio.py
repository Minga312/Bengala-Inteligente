import pyttsx3

def say_phrase(phrase, voice_id):
    engine = pyttsx3.init()
    engine.setProperty('voice', voices[1].id)
    engine.say(phrase)
    engine.runAndWait()

if __name__ == "__main__":
    phrase = "Olá, mundo!"
    voice_id = "brazil"  # Substitua "brazil" pelo identificador da voz brasileira desejada
    
    say_phrase(phrase, voice_id)
