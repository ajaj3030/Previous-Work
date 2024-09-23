from flask import Flask, jsonify, request
from flask_cors import CORS
from models.openai_api import retrieve_input, get_operations, assistants,process_assistant_operation
from flask_socketio import SocketIO, emit
from time import sleep
from custom_models import vodafone

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
remaining_operations = None
all_operations = None
input_map = {}
awaiting_input = None

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({"message": "Hello"})

@socketio.on("plan")
def plan(data):

    global remaining_operations
    global input_map
    global awaiting_input
    global all_operations
    print("awaiting input: ", awaiting_input, "input: ", input_map)
    print("message sent")
    user_input = data['user_input']

    
    
    # plan = retrieve_input(user_input)


    if not remaining_operations:
        if not "voda" in user_input.lower():
            # remaining_operations = get_operations(plan)
            print("got operations")
            remaining_operations = [
                                    {'operation_id': 0, 'operation_explanation': "Ask the user to paste the text for their CV in", 'operation_inputs': [], 'operation_order': 0, 'assistant_id': 'INPUT-FETCHER'}, 
                                    {'operation_id': 1, 'operation_explanation': 'Ask the user to input the job description and requirements', 'operation_inputs': [], 'operation_order': 1, 'assistant_id': 'INPUT-FETCHER'}, 
                                    {'operation_id': 2, 'operation_explanation': 'Generate a bullet list of overlaps between the provided input CV and the provided job requirements', 'operation_inputs': [0, 1], 'operation_order': 2, 'assistant_id': 'STANDARD'},
                                    {'operation_id': 3, 'operation_explanation': 'Use the CV, job requirements and other inputs to write the user a cover letter', 'operation_inputs': [0, 1, 2], 'operation_order': 3, 'assistant_id': 'STANDARD'}
                                    ]
        else:
            
            remaining_operations = [
                                    {'operation_id': 0, 'operation_explanation': "Ask the user to upload the a csv for the vodaphone data, tell them to make sure its in the correct format.", 'operation_inputs': [], 'operation_order': 0, 'assistant_id': 'INPUT-FETCHER'}, 
                                    {'operation_id': 1, 'operation_explanation': 'Run  inference on the vodaphone data', 'operation_inputs': [], 'operation_order': 1, 'assistant_id': 'VODAFONE-CUSTOMER-CHURN-REGRESSION'}, 
                                    ]


    if awaiting_input is not None:

        if not data['user_input'] == "VODAFONE CSV":
            emit("plan_response", {'data': "Data recieved, forwarding it to the VODAFONE-CHURN-LOGISTIC-REGRESSION-1 agent, please wait for its response."}, broadcast=True)
            pass 
        else:
            emit("plan_response", {'data': "Input recieved, forwarding it to the STANDARD-1 agent, please wait for its response."}, broadcast=True)
            input_map[awaiting_input] = user_input
            
        awaiting_input = False    
    
    print(remaining_operations)

    for operation in remaining_operations:

        assistant_item = next(item for item in assistants if item["id"] == operation["assistant_id"])


        if assistant_item.get("openai_id") is  not None:

            if operation["assistant_id"] == "INPUT-FETCHER":


                assistant_response = process_assistant_operation(operation, assistant_item["openai_id"])
                print("input_fetcher assistant_response", assistant_response[0].text.value)

                remaining_operations.pop(0)
                emit("plan_response", {'data': assistant_response[0].text.value}, broadcast=True)
                awaiting_input = operation["operation_id"]
                return 
            
            else:
                operation_inputs_ids = operation["operation_inputs"]
                # for each operation input_id loop through all_operations and get the operations with matching the same operation_id
                
                

                operation_inputs = [input_map[input_id] for input_id in operation_inputs_ids]


                
                inputs_str = "\n\n".join(operation_inputs)
            
                assistant_response = process_assistant_operation(operation, assistant_item["openai_id"], inputs_str)
                print("assistant_response", assistant_response[0].text.value)

                # this needs to work, the output of this operation needs to be put into the map
                input_map[operation["operation_id"]] = assistant_response[0].text.value
                emit("plan_response", {'data': assistant_response[0].text.value}, broadcast=True)
        
        else: 
            emit("plan_response", {'data': vodafone.inference_main()}, broadcast=True)

@socketio.on("input")
def on_input(data):
    
    user_input = data['user_input']



@app.route("/http-call")
def http_call():
    """return JSON with string data as the value"""
    data = {'data':'This text was fetched using an HTTP call to server on render'}
    return jsonify(data)

@socketio.on("connect")
def connected():
    """event listener when client connects to the server"""
    print(request.sid)
    print("client has connected")
    # emit("connect",{"data":f"id: {request.sid} is connected"})

@socketio.on('data')
def handle_message(data):
    """event listener when client types a message"""
    print("data from the front end: ",str(data))
    emit("data",{'data':data,'id':request.sid},broadcast=True)

@socketio.on("disconnect")
def disconnected():
    """event listener when client disconnects to the server"""
    print("user disconnected")
    # emit("disconnect",f"user {request.sid} disconnected",broadcast=True)

if __name__ == '__main__':
    socketio.run(app, debug=True)
   

    

