from langchain_core.prompts import ChatPromptTemplate

document_analysis_prompt = ChatPromptTemplate.from_template("""
             You are a highly capable assistant trained to analyze & summarize documents.
            Return ONLY valid JSON matching the exact schema below.

            {format_instructions}

            Analyze the following document:
            {document_text} 
    """)

document_comparison_prompt = ChatPromptTemplate.from_template("""
    You will be provided content from PDFs. Your tasks are as follows:
        1.Compare the content in 2 PDFs
        2. Identify the difference in PDF and note down the page number
        3. the output you provided must be page wise comparison content
        4. If any page don't have any change, mention as "No change"

        Input documents:
        {combined_docs}

        your response should follow this format:
        {format_instruction}                                                    
    """   
)

PROMPT_REGISTRY={"document_analysis": document_analysis_prompt, "document_comparison": document_comparison_prompt}