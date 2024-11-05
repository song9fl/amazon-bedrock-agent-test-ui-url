import json
import os
from services import bedrock_agent_runtime
import streamlit as st
import uuid

# Get config from environment variables
agent_id = os.environ.get("BEDROCK_AGENT_ID")
agent_alias_id = os.environ.get("BEDROCK_AGENT_ALIAS_ID", "TSTALIASID")  # TSTALIASID is the default test alias ID
ui_title = os.environ.get("BEDROCK_AGENT_TEST_UI_TITLE", "Agents for Amazon Bedrock Test UI")
ui_icon = os.environ.get("BEDROCK_AGENT_TEST_UI_ICON")

def init_state():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.citations = []
    st.session_state.trace = {}

# General page configuration and initialization
st.set_page_config(page_title=ui_title, page_icon=ui_icon, layout="wide")
st.title(ui_title)
if len(st.session_state.items()) == 0:
    init_state()

# Sidebar button to reset session state
with st.sidebar:
    if st.button("Reset Session"):
        init_state()

# Messages in the conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat input that invokes the agent
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("...")
        
        # Invoke the agent and get the response
        response = bedrock_agent_runtime.invoke_agent(
            agent_id,
            agent_alias_id,
            st.session_state.session_id,
            prompt
        )

        output_text = response["output_text"]

        # Refactored citation handling with fallback to trace and unique clickable citations
        try:
            citations_text = ""
            unique_citations = {}
            if response["citations"]:  # If citations are directly available
                for i, citation in enumerate(response["citations"], start=1):
                    retrieved_ref = citation["retrievedReferences"][0]
                    s3_uri = retrieved_ref["location"]["s3Location"]["uri"]
                    
                    # Convert s3:// URI to HTTPS format
                    https_url = s3_uri.replace("s3://kcknowledgebase/", "https://miscdocsihave.s3.us-east-1.amazonaws.com/")
                    
                    if https_url not in unique_citations:
                        # Make the citation clickable with HTTPS link
                        unique_citations[https_url] = f"[{len(unique_citations) + 1}] [{https_url}]({https_url})"
            else:  # Fallback to using trace if citations are empty
                trace_references = []
                for trace_item in response.get("trace", {}).get("orchestrationTrace", []):
                    if "observation" in trace_item and "retrievedReferences" in trace_item["observation"].get("knowledgeBaseLookupOutput", {}):
                        trace_references.extend(trace_item["observation"]["knowledgeBaseLookupOutput"]["retrievedReferences"])

                # Format trace-based citations as unique and clickable
                for ref in trace_references:
                    s3_uri = ref["location"]["s3Location"]["uri"]
                    https_url = s3_uri.replace("s3://kcknowledgebase/", "https://miscdocsihave.s3.us-east-1.amazonaws.com/")
                    
                    if https_url not in unique_citations:
                        unique_citations[https_url] = f"[{len(unique_citations) + 1}] [{https_url}]({https_url})"

            # Construct the final citations text
            citations_text = "\n".join(unique_citations.values())

            # Append citations to output_text if available, or show a placeholder message
            output_text += "\n\n" + citations_text if citations_text else "\n\n(No citations available)"
        
        except KeyError as e:
            st.write(f"Error processing citations: {e}")

        # Display the final cleaned output text
        placeholder.markdown(output_text, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": output_text})
        st.session_state.citations = response.get("citations", [])
        st.session_state.trace = response.get("trace", {})
