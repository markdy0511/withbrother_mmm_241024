import streamlit as st
import base64

def get_base64_of_bin_file(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def build_markup_for_logo(png_file, image_width="60%", image_height=""):
    binary_string = get_base64_of_bin_file(png_file)
    return """
            <style>
                .st-emotion-cache-1mi2ry5.eczjsme9 {
                    background-image: url("data:image/png;base64,%s");
                    background-repeat: no-repeat;
                    background-position: center;
                    background-size: %s %s;
                    height: 100px; /* 로고의 높이 설정 */
                    margin-bottom: 20px; /* 로고 아래 공간 추가 */
                }
            </style>
            """ % (
        binary_string, image_width, image_height)


def add_logo(png_file):
    logo_markup = build_markup_for_logo(png_file)
    st.markdown(
        logo_markup,
        unsafe_allow_html=True,
    )