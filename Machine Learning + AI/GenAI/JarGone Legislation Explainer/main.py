import requests
from transformers import pipeline, MarianMTModel, MarianTokenizer
import anthropic
import streamlit as st
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="", # -> This key is old
)

api_url = "https://api.anthropic.com/v1/messages"
api_key = "" # -> This key is old

model_name = "Helsinki-NLP/opus-mt-en-cy"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
headers = {
    "Content-Type": "application/json",
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01"
}

def claude_response(prompt, welsh = False, redraft = False, redraft_prompt = "", model="claude-3-opus-20240229", max_tokens=1024, stops=["\n\nHuman:"], temp=0.85, level=1):
    """
    Sends a request to the Anthropic messages API with a given prompt and parameters, and prints the response or an error message.
    If the 'welsh' parameter is True, the response is translated to Welsh.
    """
    # Define the parameters for the API request
    if selected_audience_level == "1":
        system = "You are a explanatory note-writer, you take sections of legislation on Bills and translate them into plain, common English so that the wider public can easily understand their meaning. It is EXTREMELY important you understand that the target audience has absolutely no legal background. \n\nThe detailed guide of writing the notes is given as the following:\n\n75.\tShould provide commentary on the individual provisions of the bill. To assist with the presentation of information on legislation.gov.uk (the intention is that notes will be hyper-linked with their related provisions) a note should be included for each clause and schedule, or for groups of clauses or schedules if it is felt more appropriate to provide commentary on a group of provisions.\n\n76.\tIf there is nothing useful to be added to what the bill itself says, the commentary need say no more than that “This provision / group of provisions is self-explanatory.” The commentary must never simply restate what the bill says. (See also paragraph 10.101 about the inclusion of a paragraph on the extent and application of the provision(s).)\n\n77.\tWhere a provision is not self-explanatory, and so warrants something more by way of commentary, it is still not necessary to say much by way of summary of what the provision does. The writer of the commentary should stand back from the detail of the provision and try to summarise it in one or two sentences, using everyday language. It will rarely be appropriate for the commentary to go through a clause subsection by subsection or a Schedule paragraph by paragraph.\n\n78.\tThe purpose of the commentary is to add value – what will it be helpful for the reader to know that is not apparent from reading the bill itself? Information that it might be helpful to include is: \n●\tfactual background;\n●\tan explanation of how the provision interacts with other legislation (whether other provisions of the bill or of existing legislation);\n●\tdefinitions of technical terms used in the bill;\n●\tillustrative examples of how the bill will work in practice (these might perhaps be worked examples of a calculation or examples of how a new offence might be committed); \n●\tan explanation of how the department plans to use a regulation making power.\n\n79.\tIn answering the questions that might occur to the reader, thought should be given as to whether any material that has been prepared for other purposes (for example, for inclusion in a briefing document) might usefully be incorporated into the explanatory notes. ENs have to be neutral in tone but this does not prevent the use of existing material that meets this test.\n\n80.\tIn the case of the bill’s substantive provisions, the commentary on an individual provision or group of provisions should usually include a separate paragraph at the end explaining the extent and application of the provision(s). As regards extent, use the form of words “This [clause] [Part] [Schedule] forms part of the law of ...”. It may be appropriate not to include this information for each individual provision or group of provisions if the bill is fairly short and all the provisions have the same extent and application.\n\n81.\tIf the bill imposes fines which are expressed by reference to amounts determined under subordinate legislation (e.g. a fine at a level on the standard scale) the notes should state the current amount.\n\n82.\tWhere a bill is silent as to the nature of any fees to be charged under the bill, the notes should explain the scope of the fee-charging provisions.\n\nYou will be provided with legislation initially. Sometimes, you may be provided with a prior explanation and a modification request too. You only write the explanation and nothing else, make sure the explanation is in dotpoints."
    else:
        system = "You are a explanatory note-writer, you take sections of legislation on Bills and translate them into plain, common English so that the wider public can easily understand their meaning. It is EXTREMELY important you understand you are talking to an audience with a legal background.\n\nThe detailed guide of writing the notes is given as the following:\n\n75.\tShould provide commentary on the individual provisions of the bill. To assist with the presentation of information on legislation.gov.uk (the intention is that notes will be hyper-linked with their related provisions) a note should be included for each clause and schedule, or for groups of clauses or schedules if it is felt more appropriate to provide commentary on a group of provisions.\n\n76.\tIf there is nothing useful to be added to what the bill itself says, the commentary need say no more than that “This provision / group of provisions is self-explanatory.” The commentary must never simply restate what the bill says. (See also paragraph 10.101 about the inclusion of a paragraph on the extent and application of the provision(s).)\n\n77.\tWhere a provision is not self-explanatory, and so warrants something more by way of commentary, it is still not necessary to say much by way of summary of what the provision does. The writer of the commentary should stand back from the detail of the provision and try to summarise it in one or two sentences, using everyday language. It will rarely be appropriate for the commentary to go through a clause subsection by subsection or a Schedule paragraph by paragraph.\n\n78.\tThe purpose of the commentary is to add value – what will it be helpful for the reader to know that is not apparent from reading the bill itself? Information that it might be helpful to include is: \n●\tfactual background;\n●\tan explanation of how the provision interacts with other legislation (whether other provisions of the bill or of existing legislation);\n●\tdefinitions of technical terms used in the bill;\n●\tillustrative examples of how the bill will work in practice (these might perhaps be worked examples of a calculation or examples of how a new offence might be committed); \n●\tan explanation of how the department plans to use a regulation making power.\n\n79.\tIn answering the questions that might occur to the reader, thought should be given as to whether any material that has been prepared for other purposes (for example, for inclusion in a briefing document) might usefully be incorporated into the explanatory notes. ENs have to be neutral in tone but this does not prevent the use of existing material that meets this test.\n\n80.\tIn the case of the bill’s substantive provisions, the commentary on an individual provision or group of provisions should usually include a separate paragraph at the end explaining the extent and application of the provision(s). As regards extent, use the form of words “This [clause] [Part] [Schedule] forms part of the law of ...”. It may be appropriate not to include this information for each individual provision or group of provisions if the bill is fairly short and all the provisions have the same extent and application.\n\n81.\tIf the bill imposes fines which are expressed by reference to amounts determined under subordinate legislation (e.g. a fine at a level on the standard scale) the notes should state the current amount.\n\n82.\tWhere a bill is silent as to the nature of any fees to be charged under the bill, the notes should explain the scope of the fee-charging provisions.\n\nYou will be provided with legislation initially. Sometimes, you may be provided with a prior explanation and a modification request too. You only write the explanation and nothing else, make sure the explanation is in dotpoints."
    
    if redraft:
        params = {
            "messages":[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Legislation: \n\n1Definition of “domestic abuse”\n(1)This section defines “domestic abuse” for the purposes of this Act.\n(2)Behaviour of a person (“A”) towards another person (“B”) is “domestic abuse” if—\n(a)A and B are each aged 16 or over and are personally connected to each other, and\n(b)the behaviour is abusive.\n(3)Behaviour is “abusive” if it consists of any of the following—\n(a)physical or sexual abuse;\n(b)violent or threatening behaviour;\n(c)controlling or coercive behaviour;\n(d)economic abuse (see subsection (4));\n(e)psychological, emotional or other abuse;\nand it does not matter whether the behaviour consists of a single incident or a course of conduct.\n(4)“Economic abuse” means any behaviour that has a substantial adverse effect on B’s ability to—\n(a)acquire, use or maintain money or other property, or\n(b)obtain goods or services.\n(5)For the purposes of this Act A’s behaviour may be behaviour “towards” B despite the fact that it consists of conduct directed at another person (for example, B’s child).\n(6)References in this Act to being abusive towards another person are to be read in accordance with this section.\n(7)For the meaning of “personally connected”, see section 2."
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "- This section defines the term \"domestic abuse\". The definition applies for the purposes of the Act, but it is expected to be adopted more generally, for example by public authorities and frontline practitioners. The definition of domestic abuse is in two parts. The first part deals with the relationship between the abuser and the abused. The second part defines what constitutes abusive behaviour. There are two criteria governing the relationship between the abuser and the abused. The first criterion provides that both the person who is carrying out the behaviour and the person to whom the behaviour is directed towards must be aged over 16. Abusive behaviour directed at a person under 16 would be dealt with as child abuse rather than domestic abuse. The second criterion provides that both persons must be personally connected (as defined in section 2).\n\n- Subsections (2)(b) and (3) sets out the types of behaviours that would constitute domestic abuse, if the two relationship criteria above are met. The five behaviours listed are not mutually exclusive. Behaviours that constitute \"physical or sexual abuse\" and \"violent or threatening behaviour\" are self-explanatory and likely to be readily understood by the majority of members of public and agencies responding to domestic abuse, but other terms may be less well understood and require further explanation. The reference to \"behaviour\" covers both a single incident and a course of conduct.\n\n- Subsection (3)(c) refers to \"controlling or coercive behaviour\".\n\n- Controlling behaviour is: a range of acts designed to make a person subordinate and/or dependent by isolating them from sources of support, exploiting their resources and capacities for personal gain, depriving them of the means needed for independence, resistance and escape and regulating their everyday behaviour.\n\n- Coercive behaviour is: a continuing act or a pattern of acts of assault, threats, humiliation and intimidation or other abuse that is used to harm, punish or frighten their victim.\n\n- Subsection (3)(d) refers to \"economic abuse\", the definition of which in subsection (4) provides that behaviours which constitute such abuse must have a substantial and adverse effect on a victim’s ability to acquire, use or maintain money or other property, or to obtain goods or services. The purpose of including the qualification \"substantial and adverse effect\" is to ensure that isolated incidents, such as damaging someone’s car, or not disclosing financial information, are not inadvertently captured. \"Property\" would cover items such a mobile phone or a car and also include pets or other animals (for example agricultural livestock). \"Goods and services\" would cover, for example, utilities such as heating, or items such as food and clothing.\n\n- Subsection (5) provides that a person may indirectly abuse another person through a third party, such as a child or another member of the same household. For example, the abuser may direct behaviour towards a child in the household, in order to facilitate or perpetuate the abuse of her or his partner."
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Legislation:\n\n1Fraud\n(1)A person is guilty of fraud if he is in breach of any of the sections listed in subsection (2) (which provide for different ways of committing the offence).\n(2)The sections are—\n(a)section 2 (fraud by false representation),\n(b)section 3 (fraud by failing to disclose information), and\n(c)section 4 (fraud by abuse of position).\n(3)A person who is guilty of fraud is liable—\n(a)on summary conviction, to imprisonment for a term not exceeding [F1the general limit in a magistrates’ court] or to a fine not exceeding the statutory maximum (or to both);\n(b)on conviction on indictment, to imprisonment for a term not exceeding 10 years or to a fine (or to both).\n(4)Subsection (3)(a) applies in relation to Northern Ireland as if the reference to 12 months were a reference to 6 months.\n\nPrior Explanation:\n\n- This section sets out the offence of fraud and the three ways in which it can be committed. The maximum penalty for the offence is provided in subsection (3).\n\n- Subsection (1) provides that a person is guilty of fraud if he is in breach of any of sections 2, 3 or 4. Subsection (2) lists the three ways in which the offence can be committed as provided by sections 2, 3 and 4. These are fraud by false representation (section 2), fraud by failing to disclose information (section 3) and fraud by abuse of position (section 4).\n\n- The offence is triable either way. The maximum penalty is 12 months imprisonment or a fine up to the statutory maximum (currently £5,000) or both on summary conviction (6 months and the statutory maximum in Northern Ireland) or 10 years imprisonment or an unlimited fine or both on conviction on indictment.\n\nModification Request:\n\nThere's no need to write the last sentence in the second dotppoint.\n\n"
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "- This section sets out the offence of fraud and the three ways in which it can be committed. The maximum penalty for the offence is provided in subsection (3).\n\n- Subsection (1) provides that a person is guilty of fraud if he is in breach of any of sections 2, 3 or 4. Subsection (2) lists the three ways in which the offence can be committed as provided by sections 2, 3 and 4.\n\n- The offence is triable either way. The maximum penalty is 12 months imprisonment or a fine up to the statutory maximum (currently £5,000) or both on summary conviction (6 months and the statutory maximum in Northern Ireland) or 10 years imprisonment or an unlimited fine or both on conviction on indictment."
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": redraft_prompt + " This is the relevant text for the previous prompt." + prompt
                            }
                        ]
                    }
                ],
            "system" : system,
            "model": model,
            "max_tokens": max_tokens,
            "stop_sequences": stops,
            "temperature": temp
        }
    else:
        params = {
            "messages":[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Legislation: \n\n1Definition of “domestic abuse”\n(1)This section defines “domestic abuse” for the purposes of this Act.\n(2)Behaviour of a person (“A”) towards another person (“B”) is “domestic abuse” if—\n(a)A and B are each aged 16 or over and are personally connected to each other, and\n(b)the behaviour is abusive.\n(3)Behaviour is “abusive” if it consists of any of the following—\n(a)physical or sexual abuse;\n(b)violent or threatening behaviour;\n(c)controlling or coercive behaviour;\n(d)economic abuse (see subsection (4));\n(e)psychological, emotional or other abuse;\nand it does not matter whether the behaviour consists of a single incident or a course of conduct.\n(4)“Economic abuse” means any behaviour that has a substantial adverse effect on B’s ability to—\n(a)acquire, use or maintain money or other property, or\n(b)obtain goods or services.\n(5)For the purposes of this Act A’s behaviour may be behaviour “towards” B despite the fact that it consists of conduct directed at another person (for example, B’s child).\n(6)References in this Act to being abusive towards another person are to be read in accordance with this section.\n(7)For the meaning of “personally connected”, see section 2."
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "- This section defines the term \"domestic abuse\". The definition applies for the purposes of the Act, but it is expected to be adopted more generally, for example by public authorities and frontline practitioners. The definition of domestic abuse is in two parts. The first part deals with the relationship between the abuser and the abused. The second part defines what constitutes abusive behaviour. There are two criteria governing the relationship between the abuser and the abused. The first criterion provides that both the person who is carrying out the behaviour and the person to whom the behaviour is directed towards must be aged over 16. Abusive behaviour directed at a person under 16 would be dealt with as child abuse rather than domestic abuse. The second criterion provides that both persons must be personally connected (as defined in section 2).\n\n- Subsections (2)(b) and (3) sets out the types of behaviours that would constitute domestic abuse, if the two relationship criteria above are met. The five behaviours listed are not mutually exclusive. Behaviours that constitute \"physical or sexual abuse\" and \"violent or threatening behaviour\" are self-explanatory and likely to be readily understood by the majority of members of public and agencies responding to domestic abuse, but other terms may be less well understood and require further explanation. The reference to \"behaviour\" covers both a single incident and a course of conduct.\n\n- Subsection (3)(c) refers to \"controlling or coercive behaviour\".\n\n- Controlling behaviour is: a range of acts designed to make a person subordinate and/or dependent by isolating them from sources of support, exploiting their resources and capacities for personal gain, depriving them of the means needed for independence, resistance and escape and regulating their everyday behaviour.\n\n- Coercive behaviour is: a continuing act or a pattern of acts of assault, threats, humiliation and intimidation or other abuse that is used to harm, punish or frighten their victim.\n\n- Subsection (3)(d) refers to \"economic abuse\", the definition of which in subsection (4) provides that behaviours which constitute such abuse must have a substantial and adverse effect on a victim’s ability to acquire, use or maintain money or other property, or to obtain goods or services. The purpose of including the qualification \"substantial and adverse effect\" is to ensure that isolated incidents, such as damaging someone’s car, or not disclosing financial information, are not inadvertently captured. \"Property\" would cover items such a mobile phone or a car and also include pets or other animals (for example agricultural livestock). \"Goods and services\" would cover, for example, utilities such as heating, or items such as food and clothing.\n\n- Subsection (5) provides that a person may indirectly abuse another person through a third party, such as a child or another member of the same household. For example, the abuser may direct behaviour towards a child in the household, in order to facilitate or perpetuate the abuse of her or his partner."
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Legislation:\n\n1Fraud\n(1)A person is guilty of fraud if he is in breach of any of the sections listed in subsection (2) (which provide for different ways of committing the offence).\n(2)The sections are—\n(a)section 2 (fraud by false representation),\n(b)section 3 (fraud by failing to disclose information), and\n(c)section 4 (fraud by abuse of position).\n(3)A person who is guilty of fraud is liable—\n(a)on summary conviction, to imprisonment for a term not exceeding [F1the general limit in a magistrates’ court] or to a fine not exceeding the statutory maximum (or to both);\n(b)on conviction on indictment, to imprisonment for a term not exceeding 10 years or to a fine (or to both).\n(4)Subsection (3)(a) applies in relation to Northern Ireland as if the reference to 12 months were a reference to 6 months.\n\nPrior Explanation:\n\n- This section sets out the offence of fraud and the three ways in which it can be committed. The maximum penalty for the offence is provided in subsection (3).\n\n- Subsection (1) provides that a person is guilty of fraud if he is in breach of any of sections 2, 3 or 4. Subsection (2) lists the three ways in which the offence can be committed as provided by sections 2, 3 and 4. These are fraud by false representation (section 2), fraud by failing to disclose information (section 3) and fraud by abuse of position (section 4).\n\n- The offence is triable either way. The maximum penalty is 12 months imprisonment or a fine up to the statutory maximum (currently £5,000) or both on summary conviction (6 months and the statutory maximum in Northern Ireland) or 10 years imprisonment or an unlimited fine or both on conviction on indictment.\n\nModification Request:\n\nThere's no need to write the last sentence in the second dotppoint.\n\n"
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "- This section sets out the offence of fraud and the three ways in which it can be committed. The maximum penalty for the offence is provided in subsection (3).\n\n- Subsection (1) provides that a person is guilty of fraud if he is in breach of any of sections 2, 3 or 4. Subsection (2) lists the three ways in which the offence can be committed as provided by sections 2, 3 and 4.\n\n- The offence is triable either way. The maximum penalty is 12 months imprisonment or a fine up to the statutory maximum (currently £5,000) or both on summary conviction (6 months and the statutory maximum in Northern Ireland) or 10 years imprisonment or an unlimited fine or both on conviction on indictment."
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
            "system" : system,
            "model": model,
            "max_tokens": max_tokens,
            "stop_sequences": stops,
            "temperature": temp
        }
    
    # Send the request to the API
    response = requests.post(api_url, json=params, headers=headers)
    
    # If the request is successful, print the response
    if response.status_code == 200:
        result = response.json()
        generated_text = result["content"][0]["text"]  # Updated access to generated text
        # If the 'welsh' parameter is True, translate the response to Welsh
        if welsh:
            generated_text = english_to_welsh(generated_text)
        print(generated_text)
    # If the request is not successful, print an error message
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    
    return generated_text
    
def english_to_welsh(prompt):
    """
    Translates a given prompt to Welsh.
    """
    # Translate the prompt to Welsh
    welsh_prompt = model.generate(**tokenizer(prompt, return_tensors="pt", padding=True))
    return tokenizer.decode(welsh_prompt[0], skip_special_tokens=True)

def send_email(explanatory_notes):
    # Email configuration
    print("Email sent!")


# Set page config
st.set_page_config(page_title="Legislation Explainer", layout="wide")

# Load data from CSV file
df = pd.read_csv("test_data.csv")

# Initialize state variables
if "section_dropdowns" not in st.session_state:
    st.session_state.section_dropdowns = {}

if "selected_language" not in st.session_state:
    st.session_state.selected_language = "English"

def create_section_dropdown(act, section, section_title):
    section_key = f"{act}_{section}_{section_title.replace(' ', '_')}"  # Generate a unique key
    
    with st.expander(section_title, expanded=False):
        # Initialize state variables for this section dropdown
        if section_key not in st.session_state.section_dropdowns:
            st.session_state.section_dropdowns[section_key] = {
                "explanatory_notes": "",
                "explanatory_notes_approved": None,
                "redraft_prompt": ""
            }

        # Get the state variables for this section dropdown
        explanatory_notes = st.session_state.section_dropdowns[section_key]["explanatory_notes"]
        explanatory_notes_approved = st.session_state.section_dropdowns[section_key]["explanatory_notes_approved"]
        redraft_prompt = st.session_state.section_dropdowns[section_key]["redraft_prompt"]


        # Define the LLM function
        def generate_explanatory_notes(legislation):
            # Use a language model to generate explanatory notes
            explanatory_notes = f"{claude_response(legislation)}"
            return explanatory_notes

        def generate_explanatory_notes_with_redraft(legislation, original_notes, redraft_prompt):
            # Use the LLM to generate revised explanatory notes based on the redraft prompt
            revised_notes = f"{claude_response(legislation, redraft=True, redraft_prompt=redraft_prompt)}"
            return revised_notes

        def thumbs_up_callback():
            st.session_state.section_dropdowns[section_key]["explanatory_notes_approved"] = True

        def thumbs_down_callback():
            st.session_state.section_dropdowns[section_key]["explanatory_notes_approved"] = False

        def generate_notes_callback():
            # Reset the thumbs up/down buttons
            st.session_state.section_dropdowns[section_key]["explanatory_notes_approved"] = None
            if selected_language == "Cymraeg":
                st.session_state.section_dropdowns[section_key]["explanatory_notes"] = claude_response(legislation, welsh=True)
            else:
                st.session_state.section_dropdowns[section_key]["explanatory_notes"] = generate_explanatory_notes(legislation)

        col1, col2, col3, col4 = st.columns([0.8, 0.05, 0.05, 0.1])

        with col3:
            if st.button("👍", key=f"thumbs_up_{section_key}", on_click=thumbs_up_callback):
                pass

        with col4:
            if st.button("👎", key=f"thumbs_down_{section_key}", on_click=thumbs_down_callback):
                pass

        col5, col6 = st.columns(2)

        with col5:
            # The "Legislation" text area is now populated with the selected act and section
            legislation = df[(df["act"] == act) & (df["section"] == section)]["text"].iloc[0]
            st.text_area("Legislation", value=legislation, height=200)
            if st.button(label = "Generate ", key = f"Generate Explanatory Notes_{section_key}", on_click=generate_notes_callback):
                pass

        with col6:
            st.text_area("Explanatory Notes", key = f"Explanatory Notes{section_key}", value=explanatory_notes, height=200)
            if explanatory_notes_approved is True:
                st.success("Explanatory notes approved!", icon="✅")
                
                # Add button to send email
                if st.button("Send Email", key=f"send_email_{section_key}"):
                    send_email(explanatory_notes)
                    
                    
            elif explanatory_notes_approved is False:
                redraft_prompt = st.text_input("Redraft Prompt", value=redraft_prompt, key=f"redraft_prompt_{section_key}", placeholder="Enter redraft prompt here")
                st.session_state.section_dropdowns[section_key]["redraft_prompt"] = redraft_prompt
                if redraft_prompt:
                    st.session_state.section_dropdowns[section_key]["explanatory_notes"] = generate_explanatory_notes_with_redraft(
                        claude_response(legislation, temp=0), explanatory_notes, redraft_prompt
                    )

# Streamlit UI
st.title("Legislation Explainer")

image = "gov.png"

st.sidebar.image(image, use_column_width=False)

with st.sidebar:
    st.title("Select Act and Section")
    act_options = ["Please select ACT"] + list(df["act"].unique())
    selected_act = st.selectbox("Select Act", act_options)

    if selected_act == "Please select ACT":
        section_options = ["Please select SECTION"]
    else:
        section_options = ["All"] + list(df[df["act"] == selected_act]["section"].unique())
    selected_section = st.selectbox("Select Section", section_options)

    st.title("Select Audience")
    
    # Add language selection dropdown
    language_options = ["English", "Cymraeg"]
    selected_language = st.selectbox("Select Language", language_options, index=0)
    st.session_state.selected_language = selected_language
    
    # Add audience level selection dropdown
    audience_level_options = ["No legal background", "Legal professional"]
    selected_audience_level = st.selectbox("Audience", audience_level_options, index=0)
    st.session_state.selected_audience_level = "1" if selected_audience_level == "No legal background" else "2"
    
    st.title("Version Control")
    
    # Add audience level selection dropdown
    version = ["15/04/24", "16/04/24", "17/04/24"]
    selected_version = st.selectbox("Document Version", version, index=0)
    
    branches = ["Aidan's Branch", "Meline's Branch", "Omar's Branch", "Alex's Branch"]
    selected_branch = st.selectbox("Version Branch", branches, index=0)
                
if selected_act != "Please select ACT" and selected_section != "Please select SECTION":
    if selected_section == "All":
        sections_in_act = df[df["act"] == selected_act]["section"].unique()
        for section in sections_in_act:
            section_title = df[(df["act"] == selected_act) & (df["section"] == section)]["section_title"].iloc[0]
            create_section_dropdown(selected_act, section, section_title)
    else:
        section_title = df[(df["act"] == selected_act) & (df["section"] == selected_section)]["section_title"].iloc[0]
        create_section_dropdown(selected_act, selected_section, section_title)



