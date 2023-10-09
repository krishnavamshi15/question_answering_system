# Import necessary libraries
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# Define the Streamlit app
def main():
    # Set the page title
    st.title("Question-Answering Web App")

    # Load the pre-trained model and tokenizer
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    # Create input fields for context and question
    context = st.text_area("Enter the context:", "Paris is the capital of France.")
    question = st.text_input("Enter your question:", "What is the capital of France?")

    # Create a button to trigger question answering
    if st.button("Answer"):
        # Use the question-answering pipeline to get the answer
        answer = qa_pipeline(question=question, context=context)
        st.subheader("Answer:")
        st.write(answer["answer"])

if __name__ == "__main__":
    main()
