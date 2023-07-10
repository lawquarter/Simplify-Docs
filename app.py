import streamlit as st
from google.auth import credentials
from google.oauth2 import service_account
import google.cloud.aiplatform as aiplatform
from vertexai.preview.language_models import TextGenerationModel
import vertexai
import json
from PIL import Image
import pytesseract
from pdfminer.high_level import extract_text
from google.cloud import translate_v2 as translate
from elevenlabs import set_api_key, generate, stream
import re

set_api_key("YOUR_API_KEY_HERE")

def strip_markdown(md):
    pattern = r'[\*_\[\]()\#]'
    text = re.sub(pattern, '', md)
    return text

with open("service_account.json") as f:
    service_account_info = json.load(f)

my_credentials = service_account.Credentials.from_service_account_info(
    service_account_info
)

translate_client = translate.Client(credentials=my_credentials)
aiplatform.init(credentials=my_credentials,)

with open("service_account.json", encoding="utf-8") as f:
    project_json = json.load(f)
    project_id = project_json["project_id"]

vertexai.init(project=project_id, location="us-central1")

def explain(message):
    model = TextGenerationModel.from_pretrained("text-bison")
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40
    }
    prompt = "\n\nYou are tasked with explaining the above to an elderly person. You are to explain the points above in markdown format including any actions the person should take and any steps they should take to verify the information. Use these headings in markdown format '## Summary' '## Impact', '## Action', '## Verify' and '## Recommended.'"
    full_message = message + prompt
    response = model.predict(full_message, **parameters)
    return response.text

def translate_text(text, target_lang):
    result = translate_client.translate(text, target_language=target_lang)
    return result['translatedText']

def read_pdf(file):
    text = extract_text(file)
    return text

def ocr_image(image):
    text = pytesseract.image_to_string(image)
    return text

def main():
    st.title('Simplify Docs')
    st.sidebar.subheader("Instructions")
    st.sidebar.markdown("""
        Simplify Docs is an application that helps you understand complex documents by converting them into simpler, easy-to-understand explanations. 
        Follow these steps to get started:

        1. :open_file_folder: **Upload an Image or Document**: Click the 'Browse files' button to upload an image (png, jpeg, jpg) or a PDF document.
        2. :camera: **Or, Take a Picture**: If you're on a device with a camera, click the 'Take a Picture' button to capture an image.
        3. :eyes: **View the Extracted Text**: The app will extract the text from your uploaded or captured image. Click on 'Extracted Text' to view the content.
        4. :bulb: **Get an Explanation**: After viewing the extracted text, click the 'Explain this to me' button to receive a simplified explanation of the text. 

        :hourglass: If the explanation doesn't appear right away, please wait a moment as the application is processing your request.
        """)


    languages = {'Italian': 'it', 'Spanish': 'es', 'Maltese': 'mt', 'French': 'fr', 'German': 'de', 'Russian': 'ru', 'Portuguese': 'pt', 'Dutch': 'nl', 'Turkish': 'tr', 'Greek': 'el'}
    
    #lang_option = st.radio(
    #        "Would you like the explanation to be translated?",
    #        ('No', 'Yes'))
    lang_option = 'No'
    
    if lang_option == 'Yes':
        language = st.selectbox('Select the language for translation:', list(languages.keys()))

    try:
        option = st.radio(
            "How would you like to provide the document?",
            ('Upload a document', 'Take a picture'))

        if option == 'Upload a document':
            uploaded_file = st.file_uploader("Upload Image or Document", type=['png','jpeg','jpg','pdf'])
            if uploaded_file is not None:
                if uploaded_file.type == "application/pdf":
                    extracted_text = read_pdf(uploaded_file)
                else:
                    image = Image.open(uploaded_file)
                    st.image(image, use_column_width=True)
                    extracted_text = ocr_image(image)
                with st.expander("Extracted Text"):
                    st.write("Extracted Text: ", extracted_text)
                if st.button('Explain this to me'):
                    with st.spinner('Generating explanation...'):
                        response = explain(extracted_text)
                  
                        if lang_option == 'Yes':
                            response = translate_text(response, languages[language])
                    st.subheader("Explanation:")
                    st.markdown(response)
                    stripped_response = strip_markdown(response)
                    audio = generate(
                        text=stripped_response,
                        voice="Bella",
                        model="eleven_monolingual_v1"
                        )
                    st.audio(audio)

        elif option == 'Take a picture':
            img_file_buffer = st.camera_input("Take a Picture")
            if img_file_buffer is not None:
                image = Image.open(img_file_buffer)
                st.image(image, use_column_width=True)
                extracted_text = ocr_image(image)
                with st.expander("Extracted Text"):
                    st.write("Extracted Text: ", extracted_text)

                if st.button('Explain Photo Text'):
                    with st.spinner('Generating explanation...'):
                        response = explain(extracted_text)
                        stripped_response = strip_markdown(response)
                        audio = generate(
                            text=stripped_response,
                            voice="Bella",
                            model="eleven_monolingual_v1"
                            )
                        if lang_option == 'Yes':
                            response = translate_text(response, languages[language])
                    st.subheader("Explanation:")
                    st.markdown(response)
                    st.audio(audio)

    except Exception as e:
        st.error(f'An error occurred: {str(e)}')

if __name__ == "__main__":
    main()
