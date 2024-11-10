from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
class Assistant:
    def __init__(
        self,
        system_prompt,
        llm,
        message_history=[],
        vector_store=None,
        employee_information=None,
    ):
        self.system_prompt = system_prompt
        self.llm = llm
        self.messages = message_history
        self.vector_store = vector_store
        self.employee_information = employee_information
        self.chain = self.get_conversation_chain()

    
    
    def get_conversation_chain(self, user_input):
        prompt = ChatPromptTemplate(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder("conversation_history"),
                ("human", "{user_input}")
            ]
        )

        llm = self.llm

        chain =  (
            {
                "retrieved_plicy_information": self.vector_store.as_retriver(),
                "employee_information": lambda x: self.employee_information,
                "user_input": RunnablePassthrough()
                "conversation_history": lambda x: self.messages
            }
            | self.prompt
            | self.llm
            | self.output_parser
        )

        output_parser =  StrOutputParser()


