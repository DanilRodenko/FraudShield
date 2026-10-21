import streamlit as st

# Entry point: defines the navigation; each page lives in views/
st.set_page_config(page_title="FraudShield", page_icon="🛡️", layout="wide")

pages = [
    st.Page("views/queue.py", title="Review queue", icon="🚩", default=True),
    st.Page("views/about.py", title="About the model", icon="ℹ️"),
]
st.navigation(pages).run()