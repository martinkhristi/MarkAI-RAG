import streamlit as st
import pandas as pd
import os
import time
from markitdown import MarkItDown
from langchain_groq.chat_models import ChatGroq
from pandasai import SmartDataframe

# Set up Streamlit page
st.set_page_config(page_title="Use RAG with Groq API, Microsoft MarkItDown , and PandasAI for smart document querying", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #F5F5F5;
    }
    .css-1bc7jzt {
        color: #0072C6;
    }
    .sidebar .sidebar-content {
        background-color: #F5F5F5;
        color: #333333;
    }
    .stButton button {
        background-color: #0072C6;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add a heading
st.markdown("<h1 style='text-align: center; color: #0072C6;'>Data Analysis Platform</h1>", unsafe_allow_html=True)

# Directly load the Groq LLM with API key
@st.cache_resource
def load_groq_llm():
    return ChatGroq(model_name="llama-3.3-70b-versatile", api_key="GROQ_API_KEY")

# Sidebar for user inputs
st.sidebar.title("Settings")
file_type = st.sidebar.radio("Select file type", ("CSV", "Excel", "PDF", "PowerPoint", "Word", "Image"))
uploaded_file = st.sidebar.file_uploader(f"Upload a {file_type} file", type=["csv", "xls", "xlsx", "pdf", "pptx", "docx", "jpg", "png"])

# Main application content starts here
if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]

    # Initialize MarkItDown
    md = MarkItDown()

    if file_type in ["PDF", "PowerPoint", "Word", "Image"]:
        st.subheader("File Conversion to Markdown")
        try:
            markdown_result = md.convert(uploaded_file)
            st.write("### Converted Markdown Content")
            st.text(markdown_result.text_content)
        except Exception as e:
            st.error(f"Error during conversion: {e}")

    elif file_type in ["CSV", "Excel"]:
        try:
            if file_type == "CSV":
                data = pd.read_csv(uploaded_file)
            else:
                # Explicitly specify the engine for Excel files
                data = pd.read_excel(uploaded_file, engine="openpyxl")
            
            # Quick preview of the data
            st.subheader("Data Preview")
            st.write(data.head())

            # General Information
            st.subheader("General Information")
            st.write(f"Shape of the dataset: {data.shape}")
            st.write(f"Data Types:\n{data.dtypes}")
            st.write(f"Memory Usage: {data.memory_usage(deep=True).sum()} bytes")

            # Load Groq LLM
            groq_llm = load_groq_llm()

            # SmartDataframe setup for language model interaction
            df_groq = SmartDataframe(data, config={'llm': groq_llm})

            # User query input for natural language analysis
            query = st.text_input("Enter your query about the data:")
            if query:
                try:
                    start_time = time.time()  # Start timing
                    response = df_groq.chat(query)
                    end_time = time.time()  # End timing
                    processing_time = end_time - start_time
                    
                    st.write(response)
                    st.success(f"Query processed in {processing_time:.2f} seconds.")
                except Exception as e:
                    st.error(f"An error occurred during query processing: {e}")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Footer
st.markdown(
    "<footer style='text-align: center; padding: 10px; background-color: #0072C6; color: #FFFFFF;'>Powered by AI and Open Source Tools</footer>",
    unsafe_allow_html=True,
)
