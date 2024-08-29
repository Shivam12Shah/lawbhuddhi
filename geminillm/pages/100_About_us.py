import streamlit as st

st.set_page_config(
    page_title="About Us",
    page_icon="logo.png",
    layout="centered",
)

# Create two columns for logo and markdown
cols = st.columns([1, 2])

with cols[0]:
    st.image("logo.png", width=170)
    st.title("About Us")
with cols[1]:
  
    st.markdown("""
    ## Law Buddy 
    ### Your AI Legal Assistant

    Welcome to **Law Buddy**, a hackathon project developed by a team of passionate students from **LNCT University, Bhopal**. Our project aims to create an **AI-based conversational chatbot** that helps users get quick and accurate answers to case-related queries, summaries of judgments, and court documents in English and scheduled languages of the Indian Constitution, 1950.
    This project is designed to streamline the process of legal information retrieval, making it easier for everyone to access vital legal resources efficiently.
    """)

st.subheader("Meet the Team")

team_members = [
    {"name": "Aman Kahar", "role": "Streamlit Python Development", "image": "aman.jpg", "linkedin": "https://www.linkedin.com/in/amankahar/" , "discription":"Aman is responsible for developing the user interface using Streamlit and integrating various components of the application." },
    {"name": "Anushka Joshi", "role": "Backend Development", "image": "anushka_j.jpg" , "discription" :"Anushka handles the backend development, ensuring robust data processing and server-side operations."},
    {"name": "Ayushmaan Nema", "role": "Backend Development", "image": "ayushman.jpg","discription":"Ayushmaan has also worked on backend tasks and helps in optimizing server performance."},
    {"name": "Aditya Singh Kushwah", "role": "Deployment", "image": "aditya.jpg","discription":"Aditya manages the deployment of the application, ensuring it runs smoothly on production servers."},
    {"name": "Mahak Porwal", "role": "GitHub Repository Management", "image": "mahak.jpg","discription":"Mahak is in charge of managing the GitHub repository, including version control and documentation."},
    {"name": "Anushka Gupta", "role": "LangChain RAG Implementation", "image": "anushka.jpg","discription":"Anushka works on integrating LangChain RAG for advanced natural language processing tasks."},
    {"name": "Bipasha Das", "role": "Research and Devlopment", "image": "bipasha.jpg","discription":"Bipasha has  worked on Research work and gained knowledge about the flow of the project"},
]

for member in team_members:
    cols = st.columns([1, 2])  
    with cols[0]:
        st.image(member["image"], width=100)  
    with cols[1]:
        st.markdown(f"**{member['name']}**: {member['role']}")
        if "linkedin" in member:
            st.markdown(f"[LinkedIn]({member['linkedin']})")
        st.markdown(f"({member['discription']})")
        st.markdown("---")

st.markdown("""
We are excited to bring **Law Buddy** to life and believe it has the potential to make a significant impact in the legal domain.

Thank you for visiting our project!
""")