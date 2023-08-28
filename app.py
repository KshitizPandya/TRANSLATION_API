import os
from fastapi import FastAPI
from typing import Dict, Any
from pydantic import BaseModel
from google.cloud import vision
import cv2
from googletrans import Translator
import base64
import numpy as np

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "scenic-aileron-379405-f9d2a9277374.json"

app = FastAPI()


class RequestModel(BaseModel):
    base64_image: str
    language: str


class ResponseModel(BaseModel):
    text: str
    translation: str
    language: str


def extract_text_from_image(image_data):
    try:
        decoded_image = base64.b64decode(image_data)
        image = cv2.imdecode(np.frombuffer(decoded_image, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

    _, encoded_image = cv2.imencode('.png', image)
    content = encoded_image.tobytes()
    vision_image = vision.Image(content=content)

    client = vision.ImageAnnotatorClient()
    try:
        response = client.document_text_detection(image=vision_image)
        # print(response)
    except Exception as e:
        print(f"Error detecting text: {e}")
        return None

    texts = []
    for page in response.full_text_annotation.pages:
        for i, block in enumerate(page.blocks):
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    texts.append(word_text)
    # print(texts)
    output = ' '.join(texts)
    # print(output)
    return output


def translate_text(text, target_language):
    translator = Translator()
    try:
        translation = translator.translate(text, dest=target_language)

    except Exception as e:
        print(f"Error: {e}")
        return None

    return translation.text


def get_language_code(language):
    language_codes = {"Afrikaans": "af", "Albanian": "sq", "Amharic": "am", "Arabic": "ar", "Armenian": "hy",
                      "Assamese": "as",
                      "Aymara": "ay", "Azerbaijani": "az", "Bambara": "bm", "Basque": "eu", "Belarusian": "be",
                      "Bengali": "bn",
                      "Bhojpuri": "bh", "Bosnian": "bs", "Bulgarian": "bg", "Catalan": "ca", "Cebuano": "ceb",
                      "Chichewa": "ny",
                      "Chinese (Simplified)": "zh-CN", "Chinese (Traditional)": "zh-TW", "Corsican": "co",
                      "Croatian": "hr",
                      "Czech": "cs", "Danish": "da", "Dhivehi": "dv", "Dogri": "doi", "Dutch": "nl", "English": "en",
                      "Esperanto": "eo",
                      "Estonian": "et", "Ewe": "ee", "Filipino": "fil", "Finnish": "fi", "French": "fr",
                      "Frisian": "fy", "Galician": "gl",
                      "Georgian": "ka", "German": "de", "Greek": "el", "Guarani": "gn", "Gujarati": "gu",
                      "Haitian Creole": "ht", "Hausa": "ha",
                      "Hawaiian": "haw", "Hebrew": "he", "Hindi": "hi", "Hmong": "hmn", "Hungarian": "hu",
                      "Icelandic": "is", "Igbo": "ig",
                      "Ilocano": "ilo", "Indonesian": "id", "Irish": "ga", "Italian": "it", "Japanese": "ja",
                      "Javanese": "jv", "Kannada": "kn",
                      "Kazakh": "kk", "Khmer": "km", "Kinyarwanda": "rw", "Konkani": "kok", "Korean": "ko",
                      "Krio": "kri", "Kurdish (Kurmanji)": "ku",
                      "Kurdish (Sorani)": "ckb", "Kyrgyz": "ky", "Lao": "lo", "Latin": "la", "Latvian": "lv",
                      "Lingala": "ln", "Lithuanian": "lt",
                      "Luganda": "lg", "Luxembourgish": "lb", "Macedonian": "mk", "Maithili": "mai", "Malagasy": "mg",
                      "Malay": "ms", "Malayalam": "ml",
                      "Maltese": "mt", "Maori": "mi", "Marathi": "mr", "Meiteilon (Manipuri)": "mni", "Mizo": "lus",
                      "Mongolian": "mn", "Myanmar (Burmese)": "my",
                      "Nepali": "ne", "Norwegian": "no", "Odia (Oriya)": "or", "Oromo": "om", "Pashto": "ps",
                      "Persian": "fa", "Polish": "pl", "Portuguese": "pt",
                      "Punjabi": "pa", "Quechua": "qu", "Romanian": "ro", "Russian": "ru", "Samoan": "sm",
                      "Sanskrit": "sa", "Scots Gaelic": "gd", "Sepedi": "nso",
                      "Serbian": "sr", "Sesotho": "st", "Shona": "sn", "Sindhi": "sd", "Sinhala": "si", "Slovak": "sk",
                      "Slovenian": "sl", "Somali": "so",
                      "Spanish": "es", "Sundanese": "su", "Swahili": "sw", "Swedish": "sv", "Tajik": "tg",
                      "Tamil": "ta", "Tatar": "tt", "Telugu": "te",
                      "Thai": "th", "Tigrinya": "ti", "Tsonga": "ts", "Turkish": "tr", "Turkmen": "tk", "Twi": "tw",
                      "Ukrainian": "uk", "Urdu": "ur", "Uyghur": "ug",
                      "Uzbek": "uz", "Vietnamese": "vi", "Welsh": "cy", "Xhosa": "xh", "Yiddish": "yi", "Yoruba": "yo",
                      "Zulu": "zu"}

    try:
        return language_codes[language]
    except KeyError:
        print(f"Error: Language {language} not supported.")
        return None


@app.post("/translate")
def extract_and_translate(request: RequestModel) -> dict[str, str] | dict[str, str] | dict[str, str] | dict[str, str]:
    image_data = request.base64_image
    target_language = request.language

    ocr_text = extract_text_from_image(image_data)
    # print(ocr_text)
    if not ocr_text:
        return {"text": "", "translation": "Error: Text extraction failed."}

    language_code = get_language_code(target_language)

    if not language_code:
        return {"text": "", "translation": "Error: Language not supported."}

    translation = translate_text(ocr_text, language_code)

    if not translation:
        return {"text": "", "translation": "Error: Translation failed."}

    return {"text": ocr_text, "language": target_language, "translation": translation}
