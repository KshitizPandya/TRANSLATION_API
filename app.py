import os
from fastapi import FastAPI
from typing import Dict
from pydantic import BaseModel
from google.cloud import vision
import cv2
from googletrans import Translator
import base64
import numpy as np

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "polyglot-379405-1cb386f1dbbd.json"

app = FastAPI()


class RequestModel(BaseModel):
    base64_image: str
    language: str


class ResponseModel(BaseModel):
    text: str
    translation: str


# def extract_text_from_image(image_data):
#     try:
#         decoded_image = base64.b64decode(image_data)
#         image = cv2.imdecode(np.frombuffer(decoded_image, np.uint8), cv2.IMREAD_COLOR)
#     except Exception as e:
#         print(f"Error: {e}")
#         return None

#     _, encoded_image = cv2.imencode('.png', image)
#     content = encoded_image.tobytes()
#     vision_image = vision.Image(content=content)

#     client = vision.ImageAnnotatorClient()
#     try:
#         response = client.document_text_detection(image=vision_image)
#     except Exception as e:
#         print(f"Error: {e}")
#         return None

#     texts = []
#     for page in response.full_text_annotation.pages:
#         for i, block in enumerate(page.blocks):
#             for paragraph in block.paragraphs:
#                 for word in paragraph.words:
#                     word_text = ''.join([symbol.text for symbol in word.symbols])
#                     texts.append(word_text)

#     output = ' '.join(texts)
#     return output
# print(".")

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
        print(response)
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

    output = ' '.join(texts)
    return output


def translate_text(text, target_language):
#     print("This is the text feeded for translation: ", text, type(text))
#     print("This is the target_language feeded for translation: ", target_language, type(target_language))
    translator = Translator()
    try:
        translation = translator.translate(text, dest=target_language)

    except Exception as e:
        print(f"Error: {e}")
        return None

#     print("HIHIHIHIHIHIHI_1: ", translation.text)
    return translation.text


def get_language_code(language):
    indic_language_codes = {"english": "en", "hindi": "hi", "gujarati": "gu", "arabic": "ar", "assamese": "as",
                            "bengali": "bn", "kannada": "kn", "malayalam": "ml", "marathi": "mr", "nepali": "ne",
                            "punjabi": "pa", "tamil": "ta", "telugu": "te", "thai": "th", "urdu": "ur",
                            "vietnamese": "vi"}
    foreign_language_codes = {"french": "fr", "german": "de", "italian": "it", "japanese": "ja", "korean": "ko",
                              "portuguese": "pt", "russian": "ru", "spanish": "es", "turkish": "tr"}

    try:
        return indic_language_codes[language]
    except KeyError:
        try:
            return foreign_language_codes[language]
        except KeyError:
            print(f"Error: Language {language} not supported.")
            return None


@app.post("/translate")
def extract_and_translate(request: RequestModel) -> ResponseModel:
    image_data = request.base64_image
    target_language = request.language.lower()

    ocr_text = extract_text_from_image(image_data)
    if not ocr_text:
        return {"text": "", "translation": "Error: Text extraction failed."}

    language_code = get_language_code(target_language)
    # print(language_code)

    if not language_code:
        return {"text": "", "translation": "Error: Language not supported."}

    translation = translate_text(ocr_text, language_code)
#     print("HIHIHIHIHIHIHI_2: ", translation)


    if not translation:
        return {"text": "", "translation": "Error: Translation failed."}

    return {"text": ocr_text, "translation": translation}
