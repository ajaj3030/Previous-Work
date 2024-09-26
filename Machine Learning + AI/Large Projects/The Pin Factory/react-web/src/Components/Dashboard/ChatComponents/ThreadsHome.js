import React, {
	useState,
	useEffect,
	useRef,
	useCallback,
	Fragment,
} from "react";
import {
	XMark,
	Search,
	ThreadsIcon,
	PencilIcon,
	PlusIcon,
} from "../SVG_icons.js";
import ThreadFront from "./ThreadFront.js";
import { functions, httpsCallable, auth } from "../../../firebase";

export default function ThreadsHome({
	setEnterThread,
	threads,
	setThread,
	setThreads,
	setNewThread,
}) {
	const [searchValue, setSearchValue] = useState("");
	const [showDeleteThreadButton, setShowDeleteThreadButton] = useState(false);
	const [selectedNavButton, setSelectedNavButton] = useState("All");

	const removeThread = async (indexToRemove, threadId) => {
		const deleteThread = httpsCallable(functions, "deletethread");
		try {
			await deleteThread({
				threadId: threadId,
				userId: auth.currentUser.uid,
			});
			// Update threadsToDisplay state only after the thread has been deleted in the backend
			setThreadsToDisplay((prevThreads) =>
				prevThreads.filter((thread, index) => index !== indexToRemove),
			);
		} catch (error) {
			console.error("Error deleting thread:", error);
		}
	};

	const createNewThread = httpsCallable(functions, "createnewthread");

	const createNewThreadInDB = async () => {
		try {
			const response = await createNewThread({
				userId: auth.currentUser.uid,
			});
			console.log("response for new thread", response.data.thread);
			setThread(response.data.thread);
		} catch (error) {
			console.error("Error creating new thread:", error);
		}
	};

	const getLastMessages = useCallback(
		httpsCallable(functions, "getlastmessages"),
		[], // Add any dependencies here
	);

	useEffect(() => {
		if (auth.currentUser) {
			console.log(auth.currentUser.uid);
			getLastMessages({ userId: auth.currentUser.uid })
				.then((result) => {
					// Store the threads in state
					setThreads(result.data.threads);
				})
				.catch((error) => {
					console.error("Error:", error); // Handle the error
					return {
						data: "Error getting threads",
						error: error.message,
					};
				});
		}
	}, []);

	const allRef = useRef(null);
	const privateRef = useRef(null);
	const collaborativeRef = useRef(null);
	const unreadRef = useRef(null);
	const [widths, setWidths] = useState({
		All: 0,
		Private: 0,
		Collaborative: 0,
		Unread: 0,
	});
	const [heights, setHeights] = useState({
		All: 0,
		Private: 0,
		Collaborative: 0,
		Unread: 0,
	});

	useEffect(() => {
		setWidths({
			All: allRef.current.offsetWidth,
			Private: privateRef.current.offsetWidth,
			Collaborative: collaborativeRef.current.offsetWidth,
			Unread: unreadRef.current.offsetWidth,
		});
		setHeights({
			All: allRef.current.offsetHeight,
			Private: privateRef.current.offsetHeight,
			Collaborative: collaborativeRef.current.offsetHeight,
			Unread: unreadRef.current.offsetHeight,
		});
	}, []);

	const [
		selectedNavUnderlineBorderLeftMargin,
		setSelectedNavUnderlineBorderLeftMargin,
	] = useState(0);

	const [threadsToDisplay, setThreadsToDisplay] = useState([]);

	useEffect(() => {
		if (threads) {
			if (searchValue) {
				const display = threads.filter(
					(thread) =>
						thread?.threadName?.includes(searchValue) ||
						thread?.agents?.some((agent) =>
							agent.includes(searchValue),
						) ||
						thread?.humans?.some((human) =>
							human.humanName.includes(searchValue),
						),
				);
				setThreadsToDisplay(display);
			} else if (selectedNavButton === "All") {
				setThreadsToDisplay(threads);
			} else if (selectedNavButton === "Private") {
				setThreadsToDisplay(
					threads.filter(
						(thread) =>
							thread?.humans.length === 1 &&
							thread?.creatorUserId === auth.currentUser.uid,
					),
				);
			} else if (selectedNavButton === "Collaborative") {
				setThreadsToDisplay(
					threads.filter((thread) => thread?.humans.length > 1),
				);
			} else if (selectedNavButton === "Unread") {
				setThreadsToDisplay(
					threads.filter(
						(thread) =>
							// thread.lastMessage.state === "UNREAD" &&
							thread &&
							thread?.lastMessage.sender !== "User" &&
							thread.readByUsers &&
							!thread?.readByUsers.includes(auth.currentUser.uid),
					),
				);
			}
		}
	}, [selectedNavButton, threads, searchValue]);

	useEffect(() => {
		// Update the selectedNavUnderlineBorderLeftMargin to be the sum of the widths of the nav buttons to the left of the selected one plus the spacing between them
		let sum = 20;
		for (let i = 0; i < Object.keys(widths).length; i++) {
			if (Object.keys(widths)[i] === selectedNavButton) {
				break;
			}
			sum += widths[Object.keys(widths)[i]];
			sum += 32;
		}
		setSelectedNavUnderlineBorderLeftMargin(
			sum + (Object.keys(widths).length - 1) * 4,
		);
	}, [selectedNavButton]);

	return (
		<div className="relative w-full h-full bg-white">
			{/* THREADS HEADER */}
			<h1 className="text-xl font-semibold text-gray-900 px-8 pt-6">
				Threads
			</h1>
			<h1 className="px-8 text-sm font-medium text-gray-500 pt-1">
				Work with multiple AI agents. Collaborate with other people.
				Engineer threads to optimise your outcomes.
			</h1>
			<div className="relative px-8 py-0 border-b border-gray-200 w-full pt-2">
				<div className="relative flex flex-row space-x-2 w-full justify-between items-center">
					{/* NAV BAR - LEFT SIDE UNDER HEADER */}
					<div className="flex flex-row items-center justify-items-start justify-start space-x-8">
						<button
							ref={allRef}
							onClick={() => setSelectedNavButton("All")}
							className={`text-sm font-semibold ${
								selectedNavButton === "All"
									? "text-brand-sd"
									: "text-gray-900 hover:text-gray-500"
							}`}>
							All
						</button>
						<button
							ref={privateRef}
							onClick={() => setSelectedNavButton("Private")}
							className={`text-sm font-semibold ${
								selectedNavButton === "Private"
									? "text-brand-sd"
									: "text-gray-900 hover:text-gray-500"
							}`}>
							Private
						</button>
						<button
							ref={collaborativeRef}
							onClick={() =>
								setSelectedNavButton("Collaborative")
							}
							className={`text-sm font-semibold ${
								selectedNavButton === "Collaborative"
									? "text-brand-sd"
									: "text-gray-900 hover:text-gray-500"
							}`}>
							Collaborative
						</button>
						<button
							ref={unreadRef}
							onClick={() => setSelectedNavButton("Unread")}
							className={`text-sm font-semibold ${
								selectedNavButton === "Unread"
									? "text-brand-sd"
									: "text-gray-900 hover:text-gray-500"
							}`}>
							Unread
						</button>
					</div>
					{/* RIGHT SIDE */}
					<div className="flex flex-row items-center justify-start space-x-4 pb-2">
						{/* SEARCH BAR */}
						<div className="flex flex-row items-center justify-between w-96 rounded-full bg-gray-200/70 px-4 py-2">
							<div className="flex flex-row items-center justify-start space-x-2 w-full">
								{/* Search Icon */}
								<Search
									size={"5"}
									color={"#4b5563"}
									strokeWidth={"0.4"}
								/>
								{/* Search Input */}
								<input
									type="text"
									value={searchValue}
									placeholder="Threads, agents, people"
									onChange={(e) =>
										setSearchValue(e.target.value)
									} // Step 3
									className="text-sm text-gray-900 bg-transparent outline-none w-full"
								/>
							</div>
							{/* Empty Search Button */}
							{searchValue && (
								<button onClick={() => setSearchValue("")}>
									<XMark
										size={"5"}
										color={"#4b5563"}
										strokeWidth={"0.4"}
									/>
								</button>
							)}
						</div>
						{/* NEW THREAD BUTTON */}
						<button
							onClick={() =>
								setShowDeleteThreadButton(
									!showDeleteThreadButton,
								)
							}
							className="flex flex-row space-x-2 items-center justify-center p-2 rounded-md bg-white border border-brand-sd hover:opacity-70">
							{/* Add icon */}
							<PencilIcon
								color="#15803d"
								height="16"
								width="16"
							/>
						</button>
						{/* NEW THREAD BUTTON */}
						<button
							onClick={async () => {
								await createNewThreadInDB();
								setEnterThread(true);
								setNewThread(true);
							}}
							className="flex flex-row space-x-2 items-center justify-center py-2 px-4 rounded-md bg-brand-sd hover:opacity-70">
							{/* Add icon */}
							<ThreadsIcon color="white" />
							{/* Add text */}
							<h1 className="text-sm text-white font-semibold">
								Create thread
							</h1>
						</button>
					</div>
				</div>
				{/* Border than underlines nav bar - left margin calculated by selected item and width */}
				<div
					className="absolute bottom-0 left-0 z-10 h-[3px] translate-y-1/2 bg-brand-sd transition-all duration-300 ease-in-out"
					style={{
						width: `${widths[selectedNavButton] + 12}px`,
						transform: `translateX(${-5}px)`,
						marginLeft: `${selectedNavUnderlineBorderLeftMargin}px`,
					}}></div>
			</div>
			{/* END THREADS HEADER */}

			<div className="px-8 py-4">
				{threadsToDisplay === null ? (
					<div>
						{searchValue ? (
							<div className="flex w-full justify-center text-sm font-medium text-gray-400 mt-24">
								We couldn't find any threads matching your
								search.
							</div>
						) : (
							<div>
								{selectedNavButton === "Unread" ? (
									<div className="flex w-full justify-center text-sm font-medium text-gray-400 mt-24">
										You're up to date! You have no unread
										threads.
									</div>
								) : (
									<ThreadsEmptyState />
								)}
							</div>
						)}
					</div>
				) : (
					<div className="grid grid-cols-3 gap-4">
						{threadsToDisplay.map((thread, index) => {
							console.log(thread);
							return thread && thread.lastMessage ? (
								<ThreadFront
									key={index}
									index={index}
									thread={thread}
									setThread={setThread}
									setEnterThread={setEnterThread}
									setNewThread={setNewThread}
									showDeleteThreadButton={
										showDeleteThreadButton
									}
									setShowDeleteThreadButton={
										setShowDeleteThreadButton
									}
									setThreadsToDisplay={setThreadsToDisplay}
									removeThread={removeThread}
								/>
							) : null;
						})}
					</div>
				)}
			</div>
		</div>
	);
}

function ThreadsEmptyState() {
	return (
		<div className="flex flex-col items-center justify-center space-y-4 text-center mt-24">
			<p className="text-2xl text-gray-900 font-medium">
				Introducing threads
			</p>
			<p className="text-gray-700 font-medium">
				Engineer prompts. Collaborate on threads. Work with multiple
				agents.
			</p>
			<div className="pt-8">
				<button
					type="button"
					className="flex items-center justify-center space-x-1 rounded-md bg-brand-sd px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-brand-hover">
					<ThreadsIcon color="white" height="16" width="16" />
					<h1>Create your first thread</h1>
				</button>
			</div>
		</div>
	);
}
