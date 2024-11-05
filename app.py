import json
import re
import os
from services import bedrock_agent_runtime
import streamlit as st
import uuid

# Get config from environment variables
agent_id = os.environ.get("BEDROCK_AGENT_ID")
agent_alias_id = os.environ.get("BEDROCK_AGENT_ALIAS_ID", "TSTALIASID")  # Default test alias ID
ui_title = os.environ.get("BEDROCK_AGENT_TEST_UI_TITLE", "Learning Assistant Chatbot")
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

        output_text = response.get("output_text", "No response text available")

        # Step 1: Try to parse as JSON and extract the "result" field
        result_text = ""
        try:
            parsed_output = json.loads(output_text)
            result_text = parsed_output.get("result", output_text)
        except json.JSONDecodeError:
            # Step 2: If not JSON, clean out instruction text using regex
            result_text = re.sub(r'^{.*?"result":\s*"(.*?)"}', r'\1', output_text)

        # Refactored citation handling with fallback to trace and unique clickable citations
        try:
            unique_citations = {}
            if response.get("citations"):  # If citations are directly available
                for citation in response["citations"]:
                    retrieved_ref = citation["retrievedReferences"][0]
                    s3_uri = retrieved_ref["location"]["s3Location"]["uri"]
                    
                    # Convert s3:// URI to HTTPS format
                    https_url = s3_uri.replace("s3://kcknowledgebase/", "https://miscdocsihave.s3.us-east-1.amazonaws.com/")
                    
                    if https_url not in unique_citations:
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

            # Construct the final citations text and remove all placeholder patterns like %[X]%, [X], and %%
            citations_text = "\n".join(unique_citations.values())

            # Remove any remaining placeholder patterns like %[X]% or [X] and any standalone %%
            result_text = re.sub(r"%\[\d+\]%+", "", result_text)  # Remove any %[X]% pattern
            result_text = re.sub(r"\[\d+\]", "", result_text)     # Remove any [X] pattern
            result_text = re.sub(r"\s*%%\s*", "", result_text)    # Remove any standalone %% pattern
            result_text = re.sub(r'"+$', '', result_text)  # Remove any trailing quotes

            if citations_text:
                result_text += "\n\n" + citations_text
        
        except KeyError as e:
            st.write(f"Error processing citations: {e}")

        # Display the final cleaned result text
        placeholder.markdown(result_text, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": result_text})
        st.session_state.citations = response.get("citations", [])
        st.session_state.trace = response.get("trace", {})

# Sidebar for trace information
trace_types_map = {
    "Pre-Processing": ["preGuardrailTrace", "preProcessingTrace"],
    "Orchestration": ["orchestrationTrace"],
    "Post-Processing": ["postProcessingTrace", "postGuardrailTrace"]
}

with st.sidebar:
    st.title("Trace")

    step_num = 1
    for trace_type_header in trace_types_map:
        st.subheader(trace_type_header)

        has_trace = False
        for trace_type in trace_types_map[trace_type_header]:
            if trace_type in st.session_state.trace:
                has_trace = True
                for trace in st.session_state.trace[trace_type]:
                    with st.expander(f"Trace Step {step_num}", expanded=False):
                        trace_str = json.dumps(trace, indent=2)
                        st.code(trace_str, language="json")
                    step_num += 1
        if not has_trace:
            st.text("None")

    # Display citations in the sidebar if available
    st.subheader("Citations")
    if st.session_state.citations:
        citation_num = 1
        for citation in st.session_state.citations:
            for retrieved_ref in citation["retrievedReferences"]:
                with st.expander(f"Citation [{citation_num}]", expanded=False):
                    citation_str = json.dumps({
                        "generatedResponsePart": citation["generatedResponsePart"],
                        "retrievedReference": retrieved_ref
                    }, indent=2)
                    st.code(citation_str, language="json")
                citation_num += 1
    else:
        st.text("None")
