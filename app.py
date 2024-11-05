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

# Sidebar for trace information
trace_types_map = {
    "Pre-Processing": ["preGuardrailTrace", "preProcessingTrace"],
    "Orchestration": ["orchestrationTrace"],
    "Post-Processing": ["postProcessingTrace", "postGuardrailTrace"]
}

trace_info_types_map = {
    "preProcessingTrace": ["modelInvocationInput", "modelInvocationOutput"],
    "orchestrationTrace": ["invocationInput", "modelInvocationInput", "modelInvocationOutput", "observation", "rationale"],
    "postProcessingTrace": ["modelInvocationInput", "modelInvocationOutput", "observation"]
}

with st.sidebar:
    st.title("Trace")

    # Show each trace type in separate sections
    step_num = 1
    for trace_type_header in trace_types_map:
        st.subheader(trace_type_header)

        # Organize traces by step similar to how it is shown in the Bedrock console
        has_trace = False
        for trace_type in trace_types_map[trace_type_header]:
            if trace_type in st.session_state.trace:
                has_trace = True
                trace_steps = {}

                for trace in st.session_state.trace[trace_type]:
                    if trace_type in trace_info_types_map:
                        trace_info_types = trace_info_types_map[trace_type]
                        for trace_info_type in trace_info_types:
                            if trace_info_type in trace:
                                trace_id = trace[trace_info_type]["traceId"]
                                if trace_id not in trace_steps:
                                    trace_steps[trace_id] = [trace]
                                else:
                                    trace_steps[trace_id].append(trace)
                                break
                    else:
                        trace_id = trace["traceId"]
                        trace_steps[trace_id] = [{trace_type: trace}]

                # Show trace steps in JSON similar to the Bedrock console
                for trace_id in trace_steps.keys():
                    with st.expander(f"Trace Step {step_num}", expanded=False):
                        for trace in trace_steps[trace_id]:
                            trace_str = json.dumps(trace, indent=2)
                            st.code(trace_str, language="json", line_numbers=trace_str.count("\n"))
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
                    st.code(citation_str, language="json", line_numbers=citation_str.count("\n"))
                citation_num += 1
    else:
        st.text("None")
