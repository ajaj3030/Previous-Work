import React, { useState } from "react";
import ThreadsHome from "./ChatComponents/ThreadsHome";
import Thread from "./ChatComponents/Thread";

export default function ChatView() {
	const [enterThread, setEnterThread] = useState(false);
	const [thread, setThread] = useState(false);
	const [newThread, setNewThread] = useState(false);
	const [threads, setThreads] = useState();

	return (
		<div className="relative w-full h-full bg-transparent">
			{enterThread ? (
				<Thread
					thread={thread}
					setThreads={setThreads}
					setThread={setThread}
					setEnterThread={setEnterThread}
					setNewThread={setNewThread}
					newThread={newThread}
				/>
			) : (
				<ThreadsHome
					setEnterThread={setEnterThread}
					threads={threads}
					setThread={setThread}
					setThreads={setThreads}
					setNewThread={setNewThread}
					newThread={newThread}
				/>
			)}
		</div>
	);
}
