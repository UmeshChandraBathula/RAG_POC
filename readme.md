# Exercise: Build an Onboarding Assistant for the Umbrella Corporation

### Overview

In this exercise, you will build an **onboarding assistant** for the **Umbrella Corporation**, a fictional company from the *Resident Evil* franchise. The assistant will guide new employees through company policies, answer questions about internal regulations, and provide personalized responses based on the employee's information. You will use **Streamlit** to build a graphical user interface and **LangChain** to integrate **Retrieval Augmented Generation (RAG)**. The assistant will augment its responses using a vector store filled with information extracted from a provided PDF of Umbrella Corporation's internal regulations.

The assistant will leverage two data sources:
1. **Employee-specific data**: Generated by a synthetic data generator that emulates an employee database.
2. **Company policies and regulations**: Retrieved from the provided PDF, stored in a local vector database, and used to answer questions relevant to Umbrella Corporation's rules.

---

### Business Context: Umbrella Corporation Employee Onboarding

The **Umbrella Corporation** is a large, multinational company involved in pharmaceutical development, genetic research, and biotechnology. While known for its cutting-edge work, the company also maintains strict internal policies and procedures, making it essential for new employees to quickly familiarize themselves with various rules and regulations. 

The onboarding assistant you will create is designed to help new employees by providing personalized answers to their questions about company policies, benefits, work schedules, safety protocols, and other key aspects of their employment at Umbrella Corporation. The assistant will personalize its responses based on the specific employee interacting with it and will also retrieve answers from a vector store of the company's internal regulations.

---

### Instructions

#### 1. Fork and Set Up the GitHub Repository

Fork the provided GitHub repository to your own GitHub account and clone it to your local machine. The repository contains boilerplate code, including a **Synthetic Data Generator** module that will simulate employee data, such as personal details, role, and start date, which will be used by the assistant to personalize interactions.

Once cloned, create and activate a virtual environment in your local project directory, and install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Now you're ready to start building the application!

#### 2. Display Employee Data

When an employee interacts with the chatbot, display their **personal details** in the sidebar using Streamlit’s `st.sidebar` feature. This will provide context for debugging and testing. You can use `st.write` to display the JSON generated by the synthetic data module. The employee data will include:

- **Name**
- **Employee ID**
- **Position**
- **Date of hire**
- **Department**
- **Skills**
- **Work location**

The synthetic data will be generated by the `generate_employee_data()` function from the **`synthetic_data` module**, which is provided in the repository. You simply need to **import the module and use the function** to generate this information.

#### 3. Ingest Company Policies into a Vector Store

Next, you will use a provided PDF document containing the internal rules and regulations of the Umbrella Corporation. Your task is to **ingest this PDF into a vector store** using a tool like **FAISS** or **Pinecone**, which allows for efficient similarity searches. The vector store will serve as the knowledge base for answering employee questions about company policies. You will:

- Split the document into chunks.
- Generate embeddings for each chunk.
- Store the embeddings in the vector store, which will later be queried to augment responses.

This PDF might include sections on:
- **General company policies**
- **Security protocols**
- **Workplace safety guidelines**
- **Employee benefits**
- **Company ethics and code of conduct**

#### 4. Build the Chatbot Interface

Your chatbot should take user queries and respond using a language model API (such as GPT-4). The assistant will augment its responses with two data sources:
1. **Employee-specific data** (e.g., name, role, skills, and department).
2. **Company regulations** stored in the vector store.

The assistant will handle common onboarding queries, such as:
- **“What are the safety protocols in the lab?”**
- **“Can you explain the company’s leave policy?”**
- **“What benefits am I eligible for as a new employee?”**

#### 5. Structure the Application Using LangChain

Use **LangChain** to manage the interaction pipeline. You should define a **prompt template** that includes placeholders for the employee data and retrieved policy information. The assistant will use this prompt to generate personalized and policy-informed responses.

It's recommended (but not required) to organize your code using a class structure, with methods such as:
- `render()`: to render the Streamlit front end.
- `get_employee_data()`: to fetch the synthetic employee data.
- `get_response()`: to generate a response by augmenting the employee data and retrieving relevant information from the vector store.
- `retrieve_context()`: to query the vector store with embeddings generated from the employee’s question.

#### 6. Implement Streaming Responses

Enable **streaming** in Streamlit to display the assistant’s responses in real time as the language model generates them. This feature will enhance the user experience by providing immediate feedback during the conversation.

#### 7. Cache Employee Data

Use **Streamlit’s session state** or **LangChain’s caching mechanisms** to store employee data for the duration of the session. This avoids re-fetching or re-generating the synthetic employee data with each new query and ensures a consistent context for the entire conversation.

#### 8. Test Common Employee Use Cases

Test the assistant with several onboarding scenarios. For example:
- **Scenario 1:** A new employee is curious about the company’s rules for taking time off.
- **Scenario 2:** An employee needs information on security protocols for a particular research facility.
- **Scenario 3:** A new hire asks about the safety measures in place for genetic research labs.

Ensure that the chatbot provides clear and helpful responses, augmenting answers with both employee data and retrieved policy information.

---

### Additional Hints

- **Prompt Design**: Design the prompt template so that the language model has access to all the relevant context (employee data, retrieved policy info) for generating responses. For example:

  ```text
  "You are an onboarding assistant for Umbrella Corporation. The employee, {name}, works as a {position} in the {department} department. Their employee ID is {employee_id}, and their start date was {start_date}. 
  Based on the question: '{employee_query}', retrieve any relevant information from the company regulations and generate a helpful response."
  ```

- **Querying the Vector Store**: When an employee asks a question, convert the query into embeddings and retrieve relevant chunks from the vector store. These chunks will provide context for the language model to generate an appropriate response.

- **Session State**: Use **Streamlit’s session state** to maintain the employee data and query history across multiple interactions.

- **User Interface**: Customize the UI to make it employee-friendly, including buttons, text input areas, and a sidebar displaying the employee’s details.

---

### Deliverables

- A **Streamlit application** that loads synthetic employee data and displays it in the sidebar.
- A **chatbot interface** that responds to employee queries by augmenting the prompt with employee-specific data and retrieving relevant company policy information from the vector store.
- Use of **LangChain** to manage prompts, retrievals, and language model interactions.
- Implementation of **streaming responses** for real-time feedback.
- **Caching** of employee data and conversation history for the session.

Once completed, ensure your chatbot is functional in your forked repository. Let me know if you have any questions!

Good luck!