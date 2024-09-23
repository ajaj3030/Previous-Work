from openai import OpenAI
# from custom_models.vodafone import inference_main
import json

openai_key = "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"

client = OpenAI(
    api_key="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",
    organization='org-4PjxajGSHzCiECC3MurR3DL4',
)

assistants = [
    {"id": "STANDARD", "description": "a general purpose assistant.", "openai_id": "asst_FYEwS0Z4WOlytnIvEhFAD8bO"},
    {"id": "VODAFONE-CUSTOMER-CHURN-REGRESSION", "description": "an assistant that predicts whether a customer will churn.", "openai_id": "asst_yo0b05jFZrITLyHqe4OWt2PJ"},
    {"id": "INTERNET-SEARCH", "description": "an assistant for searching the internet"},
    {"id": "TEXT-TO-IMAGE", "description": "an assistant that creates an image from text."},
    {"id": "IMAGE-TO-IMAGE-AND-TEXT", "description": "an assistant that creates an image from another image and accompanying text."},
    {"id": "IMAGE-TO-TEXT", "description": "an assistant that generates text that describes the image."},
    {"id": "INPUT-FETCHER", "description": "an assistant that fetches the required inputs from the user.", "openai_id": "asst_MXKNgtD1rDssql1BjmSVCOjy"},
]


def retrieve_input(prompt):
    # SPLIT TASK INTO MULTIPLE OPERATIONS
    plan = plan_task(prompt)
    # ALLOCATE OPERATIONS TO ASSISTANTS
    plan = allocate_plan(plan)

    return plan

    
def allocate_plan(plan):

    print(plan)
    json_plan = json.loads(plan)
    operations = json_plan["operations"]
    operations = json.dumps(operations)

    assistant_list = '\n'.join([f"- '{assistant['id']}' - {assistant['description']}" for assistant in assistants])

    instructions = f"""
        You are an Operations Allocation assistant. Your job is to take a list of operations by the user and allocate them to another assistant.

        You can only allocate an assistant from the following list (assistant_id - description):
        {assistant_list}

        You should return the operation allocation in the following JSON structure:
        '{{"operation_id": "<ENTER OPERATION_ID>", "assistant_id": "<ENTER ASSISTANT_ID>"}}'

        'operation_id': the operation_id in the JSON object passed by the user.
        'assistant_id': the assistant_id that you have allocate the task to.

        You should always allocate assistants who's description is most aligned with the 'operation_explanation' passed in by the user.
    """

    planning_assistant_id = "asst_4pcEl2VuubyggRvGXK1hsvM2"

    thread = create_thread(operations)
    run = run_thread(thread.id, planning_assistant_id, instructions)
    message = get_completion(thread.id, run.id)

    allocation = json.loads(message[0].text.value)

    for operation in json_plan["operations"]:
        operation_id = operation["operation_id"]
        operation["assistant_id"] = allocation[operation_id]['assistant_id']

    return json_plan


def plan_task(prompt):

    planning_assistant_id = "asst_PIjyCmQIXMZfHP3HVijcwcLX"

    thread = create_thread(prompt)
    run = run_thread(thread.id, planning_assistant_id)
    message = get_completion(thread.id, run.id)

    return message[0].text.value


def create_thread(prompt):
    thread = client.beta.threads.create(
        messages=[
            {
            "role": "user",
            "content": prompt,
            }
        ]
    )
    return thread


def check_run_status(thread_id, run_id):
    run = client.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run_id
    )
    return run.status


def list_messages(thread_id):
    thread_messages = client.beta.threads.messages.list(thread_id=thread_id)
    return thread_messages.data


def get_completion(thread_id, run_id):
    while check_run_status(thread_id, run_id) != "completed":
        continue
    messages = list_messages(thread_id)
    assistant_response = messages[0].content
    return assistant_response


def run_thread(thread_id, assistant_id, instructions=None):
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions=instructions
    )
    return run


def get_response(prompt, type=None, model="gpt-4-1106-preview", system="You are a helpful assistant."):

    response_format = {"type": "json_object"} if type == "JSON" else None

    completion = client.chat.completions.create(
        model=model,
        response_format = response_format,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message


def process_assistant_operation(operation, openai_id, extra_inputs= None):

    if not extra_inputs:
        thread = create_thread(operation["operation_explanation"])
        print("request to agent", operation["operation_explanation"])
    else:
        combined_input = "Instructions:\n" + operation["operation_explanation"] + "\n\n Additional Inputs:\n"+extra_inputs
        print("request to agent", combined_input)
        thread = create_thread(operation["operation_explanation"])

    run = run_thread(thread.id, openai_id, operation["operation_explanation"])
    output = get_completion(thread.id, run.id)
    return output


def get_operations(plan):
    
    prompt = """A flow where the agent takes text input of a users CV and a job description, generates a list of skills that the user has and a list of skills that the job requires, then inputs this list, the CV, and the job requirements into an agent that writes a cover letter."""
    response = retrieve_input(prompt)
    operations = plan["operations"]

    return operations
    for operation in operations:
        assistant_item = next(item for item in assistants if item["id"] == operation["assistant_id"])
        if assistant_item.get("openai_id",None) is not None:
            assistant_response = process_assistant_operation(operation, assistant_item["openai_id"])

            if operation["assistant_id"] == "INPUT-FETCHER":
                message = assistant_response.value
                operations.pop(0)
                return message, operations
            
            elif operation["assistant_id"] == "VODAFONE-CUSTOMER-CHURN-REGRESSION":
                predictions, accuracy = inference_main()
                # add to main thread

                input_dict[operation["operation_id"]] = assistant_response.value
                
                
        break



if __name__ == "__main__":
    operations = [{'operation_id': 0, 'operation_explanation': "Ask the user to provide a text input of their CV", 'operation_inputs': [], 'operation_order': 0, 'assistant_id': 'INPUT-FETCHER'}, {'operation_id': 1, 'operation_explanation': 'Extract required skills from job description', 'operation_inputs': [], 'operation_order': 1, 'assistant_id': 'IMAGE-TO-TEXT'}, {'operation_id': 2, 'operation_explanation': 'Generate cover letter', 'operation_inputs': [0, 1], 'operation_order': 2, 'assistant_id': 'STANDARD'}]
    output = process_assistant_operation(operations[0], "asst_MXKNgtD1rDssql1BjmSVCOjy")[0]
    print(output.text.value)



    

