import React, { useContext, useState, useRef, useEffect } from "react";
import { TaskContext } from "../TaskContext";
import Select from "react-select";
import { line } from "d3";
import TextareaAutosize from "react-textarea-autosize";
import { functions, httpsCallable, auth } from "../../../firebase";
import { v4 as uuidv4 } from "uuid";
import axios from "axios";
import FileUploadButton from "./UploadFile";

import { io } from "socket.io-client";

export default function ChatInput({
	thread,
	setThreads,
	setThread,
	tasks,
	inputFocus,
	setInputFocus,
	setNewThread,
	newThread,
	setRelatedMessages,
	relatedMessages,
	currentChainId,
	setCurrentChainId,
	currentMessage,
	setCurrentMessage,
}) {


	const [socketInstance, setSocketInstance] = useState("");
	const [loading, setLoading] = useState(true);
	const [buttonStatus, setButtonStatus] = useState(false);
	
	useEffect(() => {
		if (!socketInstance) {
			const socket = io('127.0.0.1:5000/', {
				transports: ['websocket'],
				cors: {
					origin: 'https://localhost:3000/',
				},
			});

			socket.on("connect", (data) => {
				console.log(data);
			});

			setSocketInstance(socket);

			socket.on("disconnect", (data) => {
				console.log(data);
			});

			return function cleanup() {
				socket.disconnect();
			};
		}
	}, []);

	useEffect(() => {
		if (socketInstance) {
			socketInstance.on("plan_response", (data) => {
				// Handle the response received from the server after emitting the event
				const received_message = {
					messageId: uuidv4(),
					contentType: "text",
					content: JSON.stringify(data["data"]),
					sender: 'User',
					recipient: "",
					createdAt: Math.floor(Date.now() / 1000),
				};
				setRelatedMessages((prevMessages) => [
					...prevMessages,
					received_message,
				]);
				setCurrentMessage(received_message);
			});
		}
	}, [socketInstance]);

	

	const [isFocused, setIsFocused] = useState(false); // new state variable to track focus
	const agents = ["Agent 1", "Agent 2", "Agent 3"]; // replace with real
	const [prompt, setPrompt] = useState("");

	const agentOptions = agents.map((agent) => ({
		value: agent,
		label: agent,
	}));

	const handleSubmission = (e) => {
		e.preventDefault(); // prevent the default form submit action
		console.log("called");

		const message = {
			messageId: uuidv4(),
			contentType: "text",
			content: prompt,
			sender: 'User',
			recipient: "",
			createdAt: Math.floor(Date.now() / 1000),
		};

		setRelatedMessages((prevMessages) => [...prevMessages, message]);
		setCurrentMessage(message);
		setPrompt("");

		// Emitting a socket event instead of using axios to submit to the 'plan' endpoint
		socketInstance.emit('plan', { user_input: prompt }, (response) => {

			
		});
	}; // banana


	const [currentRows, setCurrentRows] = useState(1);
	const textAreaRef = useRef(null);
	const lineHeight = 24; // This is the line-height of the textarea

	// Calculate number of lines in textarea
	useEffect(() => {
		const textAreaElement = textAreaRef.current;
		const rows = Math.floor(textAreaElement.scrollHeight / lineHeight);
		setCurrentRows(rows);
	}, [prompt]);

	const handleChange = (e) => {
		setPrompt(e.target.value);
	};

	useEffect(() => {
		const handleKeyDown = (e) => {
			const isMac =
				window.navigator.platform.toUpperCase().indexOf("MAC") >= 0;
			const isCmdOrCtrl = isMac ? e.metaKey : e.ctrlKey;

			if (isCmdOrCtrl && e.key === "Enter") {
				console.log("Command + Enter was pressed");
				handleSubmission(e);
			}
		};
		window.addEventListener("keydown", handleKeyDown);
		return () => {
			window.removeEventListener("keydown", handleKeyDown);
		};
	}, [handleSubmission]);

	return (
		<div className="flex flex-row w-2/3 items-center justify-center pb-6 bg-transparent">
			<div className="flex flex-row px-4 py-2 rounded-lg bg-white border border-gray-300 shadow-[0px_-5px_20px_0px_#ffffff] drop-shadow-xl text-gray-900 flex-grow">
				<div className="flex relative bg-transparent w-full">

					<FileUploadButton />

					<form
						onSubmit={handleSubmission}
						className={`flex flex-row bg-transparent w-full justify-between space-x-4 ${
							currentRows > 1 ? "items-end" : "items-center"
						}`}>

						<TextareaAutosize
							ref={textAreaRef}
							minRows={1}
							maxRows={8}
							placeholder="Send a message"
							value={prompt}
							onChange={handleChange}
							onFocus={() => setIsFocused(true)}
							className={`
                          w-full bg-transparent border-none ring-none outline-none text-base leading-6 m-0 p-0 resize-none custom-scrollbar
                          ${isFocused ? "focused" : "unfocused"}
                      `}
							style={{ outline: "none" }}
						/>
						<button
							type="submit"
							disabled={!prompt || loading}
							className="flex items-center justify-center text-white font-bold p-2 rounded-lg bg-purple-600/80 disabled:bg-white">
							<Send
								size="5"
								color={`${
									!prompt || loading ? "gray" : "white"
								}`}
							/>
						</button>
					</form>
				</div>
			</div>
		</div>
	);
}

function Send({ size = "6", color = "currentColor" }) {
	return (
		<svg
			xmlns="http://www.w3.org/2000/svg"
			viewBox="0 0 24 24"
			fill={color}
			class={`w-${size} h-${size}`}>
			<path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
		</svg>
	);
}



