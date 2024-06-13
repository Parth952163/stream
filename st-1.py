import streamlit as st
import pandas as pd 
import streamlit as st

# Define the page title
page_title = "File Uploader"
#st.title("File uploader")
# Set the page configuration
st.set_page_config(page_title=page_title)

#st.set_page_config(page_title = "File Uploader")

df  = st.file_uploader(label= "Upload data")

if df:
    df - pd.read_csv(df)
    st.write(df.head())