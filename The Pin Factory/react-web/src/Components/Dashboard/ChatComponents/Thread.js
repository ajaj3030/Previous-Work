import React, { useState, useContext, useEffect, useCallback } from "react";
import {
	ArrowLeftIcon,
	UserGroupIcon,
	LockClosedIcon,
	PlusIcon,
	PencilIcon,
	CheckIcon,
	PaperAirplaneIcon,
} from "@heroicons/react/20/solid";
import ChatInput from "./ChatInput";
import Message from "./Message";
import { TaskContext } from "../TaskContext";
import { ComponentTree } from "../DashboardHomeComponents/ComponentTree";
import "../../utils/scrollbar.css";
import ProfilePicture from "../../../assets/test_pp/IMG_2109.PNG";
import AvatarProfilePicture from "../../../assets/baby-animation/profile_picture.png";
import { EngineeringIcon } from "../SVG_icons";
import { functions, httpsCallable, auth } from "../../../firebase";

export default function Thread({
	thread,
	setThreads,
	setThread,
	setEnterThread,
	newThread,
	setNewThread,
}) {
	const [relatedMessages, setRelatedMessages] = useState([]);
	const [currentMessage, setCurrentMessage] = useState({});
	// need to tests currentChainId
	const [currentChainId, setCurrentChainId] = useState();
	const getthreadmessagesbychainid = useCallback(
		httpsCallable(functions, "getthreadmessagesbychainid"),
		[],
	);
	useEffect(() => {
		if (!newThread) {
			// replaced getthreadmessages with getthreadmessagesbychainid
			const result = getthreadmessagesbychainid({
				threadId: thread.threadId,
			})
				.then((result) => {
					setRelatedMessages(result.data.messages);
					setCurrentChainId(result.data.lastMessage.chainId);
					setCurrentMessage({
						messageId: result.data.lastMessage.lastMessageId,
					});
				})
				.catch((error) => {
					console.log("error", error);
				});
		}
	}, [newThread, thread.threadId]);

	const [isEditable, setIsEditable] = useState(false);
	const [threadName, setThreadName] = useState(thread.threadName);
	const [threadDescription, setThreadDescription] = useState(
		thread?.threadDescription ||
			`A thread for us all to collaborate on delivering the
  financial presentation for Goldilocks next week. Feel free
  to make new threads and try things out.`,
	);

	const updateThreadNameAndDescription = httpsCallable(
		functions,
		"updatethreadnameanddescription",
	);
	const handleIconClick = async () => {
		setIsEditable(!isEditable);
		if (isEditable) {
			await updateThreadNameAndDescription({
				threadName: threadName,
				threadDescription: threadDescription,
				threadId: thread.threadId,
			});
		}
	};
	const [inviteModal, setInviteModal] = useState(false);
	const [inviteeEmail, setInviteeEmail] = useState("");

	const addInviteeToThread = httpsCallable(functions, "addinviteetothread");
	const handleAddInvitee = async () => {
		if (!inviteeEmail) {
			// Check that inviteeEmail is not empty or undefined
			console.error("Error: No invitee email entered");
			return;
		}
		try {
			await addInviteeToThread({
				threadId: thread.threadId,
				userId: auth.currentUser.uid,
				inviteeEmail: inviteeEmail,
			});
			setInviteeEmail(""); // Clear input after successful invitation
			setInviteModal(false); // Close modal after successful invitation
		} catch (error) {
			console.error("Error inviting user:", error);
		}
	};

	const [showTree, setShowTree] = useState(false);

	return (
		<div className="relative flex w-full h-full justify-between bg-white">
			{/* Thread header */}
			{!showTree && (
				<div className="w-1/4 flex flex-col space-y-4 justify-start item-start p-6 border-r border-gray-900/10 bg-white">
					<button onClick={() => setEnterThread(false)}>
						<ArrowLeftIcon className="w-6 text-gray-900 hover:opacity-70" />
					</button>

					<div className="flex flex-col justify-start items-start space-y-1">
						<div className="flex flex-row justify-between w-full pb-2">
							<input
								className={`text-lg font-semibold text-gray-900 w-full mr-2 rounded-md pl-1 bg-transparent ${
									isEditable
										? " ring-green-700 ring-1"
										: " bg-transparent"
								}`}
								disabled={!isEditable}
								onChange={(e) => setThreadName(e.target.value)}
								value={threadName}></input>
						</div>
						{thread.humans?.length > 1 ? (
							<div className="flex items-center space-x-1  text-purple-700 rounded-md bg-purple-100 px-2 py-1">
								<UserGroupIcon className="w-4" />
								<p className="text-xs font-bold">Group</p>
							</div>
						) : (
							<div className="flex items-center space-x-1  text-zinc-700 rounded-md bg-zinc-200 px-2 py-1">
								<LockClosedIcon className="w-3" />
								<p className="text-xs font-bold">Private</p>
							</div>
						)}
					</div>

					<textarea
						className={`text-sm text-gray-700 leading-4.5 rounded-md pt-2 pl-1 ${
							isEditable
								? " ring-green-700 ring-1"
								: " bg-transparent"
						}`}
						disabled={!isEditable}
						value={threadDescription}
						onChange={(e) =>
							setThreadDescription(e.target.value)
						}></textarea>
				</div>
			)}
			{showTree && <ComponentTree thread={thread}></ComponentTree>}
			
			<div className="flex flex-col items-center relative w-3/4 h-full bg-transparent">

				{/* Chat Body */}
				<div className="relative w-full overflow-y-auto overscroll-none flex flex-col-reverse flex-grow custom-scrollbar">
					{Array.isArray(relatedMessages) &&
						relatedMessages
							.slice()
							.reverse()
							.map((message, index) => (
								<Message
									key={message.messageId}
									task={message}
									index={index}
								/>
							))}
				</div>

				<div className="flex justify-center relative w-full">
					{/* ChatInput is kept outside of the scrollable area */}
					<ChatInput
						// thread={thread}
						// setThreads={setThreads}
						// setThread={setThread}
						// setNewThread={setNewThread}
						// newThread={newThread}
						relatedMessages={relatedMessages}
						setRelatedMessages={setRelatedMessages}
						// currentChainId={currentChainId}
						// setCurrentChainId={setCurrentChainId}
						currentMessage={currentMessage}
						setCurrentMessage={setCurrentMessage}
					/>
				</div>

				<button
					onClick={() => {
						setShowTree(!showTree);
					}}
					className="absolute top-0 right-0 px-4 py-2 rounded-md bg-white border border-gray-400 mr-8 mt-4 box-shadow-lg hover:opacity-70">
					<div className="flex items-center justify-center space-x-2">
						<EngineeringIcon color={"black"} />
						<h1 className="text-sm font-semibold text-gray-900">
							Engineer
						</h1>
					</div>
				</button>
			</div>
		</div>
	);
}


// <button NEXT TO THREAD NAME
// onClick={handleIconClick}
// className="flex flex-row space-x-2 p-2 rounded-md bg-white border border-green-700 hover:opacity-70">
// {/* Add icon */}
// {isEditable ? (
// 	<CheckIcon className="w-3 h-3" />
// ) : (
// 	<PencilIcon
// 		color="#ea580c"
// 		className="w-3 h-3"
// 	/>
// )}
// </button>


// {/* Humans Line */}
// <div className="flex items-center justify-between mr-2 pt-6">
// <div className="flex items-center space-x-4">
// 	<h1 className="text-md font-medium text-gray-600">
// 		People
// 	</h1>
// 	{thread?.humans?.length > 1 && (
// 		<div className="flex -space-x-2">
// 			{thread.humans
// 				.slice(0, 3)
// 				.map((human, index, array) => (
// 					<div
// 						key={human.humanId}
// 						className={`relative ${
// 							index === 0
// 								? "z-20"
// 								: index === 1
// 								? "z-10"
// 								: "z-0"
// 						} place-items-center`}>
// 						<img
// 							src={ProfilePicture}
// 							alt="human name"
// 							className="h-6 w-6 rounded-full outline outline-2 outline-white"
// 						/>
// 						{index === array?.length - 1 &&
// 							thread?.humans?.length >
// 								3 && (
// 								<div className="absolute flex inset-0 pl-[5.5px] pb-[1px] items-center justify-center text-sm font-semibold text-white bg-gray-400 bg-opacity-50 rounded-full">
// 									+
// 								</div>
// 							)}
// 					</div>
// 				))}
// 		</div>
// 	)}
// </div>
// {thread?.humans?.length > 1 ? (
// 	<h1 className="text-md font-bold text-gray-900">
// 		{thread.humans.length}
// 	</h1>
// ) : (
// 	<button
// 		onClick={() => setInviteModal(!inviteModal)}
// 		className="flex items-center space-x-1 text-purple-700 rounded-md bg-purple-100 px-2 py-1 hover:opacity-70 text-md font-medium">
// 		Invite &nbsp; <UserGroupIcon className="w-4" />
// 	</button>
// )}
// </div>
// {inviteModal && (
// <div>
// 	<p className="text-xs font-bold">
// 		Enter email of invitee
// 	</p>
// 	<div className="flex-row flex pt-1 justify-between">
// 		<input
// 			className={`text-sm rounded-md p-1 ring-1 mr-1  w-full${
// 				inviteModal ? `` : ``
// 			}`}
// 			type="text"
// 			value={inviteeEmail}
// 			onChange={(e) =>
// 				setInviteeEmail(e.target.value)
// 			}
// 		/>
// 		<PaperAirplaneIcon
// 			onClick={handleAddInvitee}
// 			className="h-6 w-6 pt-1 cursor-pointer"
// 		/>
// 	</div>
// </div>
// )}

// {/* Agents Line */}
// <div className="flex items-center justify-between mr-4 pt-2">
// <div className="flex items-center space-x-4">
// 	<h1 className="text-md font-medium text-gray-600">
// 		Agents
// 	</h1>
// 	<div className="flex -space-x-2">
// 		{thread.agents
// 			?.slice(0, 3)
// 			.map((agent, index, array) => (
// 				<div
// 					key={agent.id}
// 					className={`relative ${
// 						index === 0
// 							? "z-20"
// 							: index === 1
// 							? "z-10"
// 							: "z-0"
// 					} place-items-center`}>
// 					{/* LEFT IMAGE */}
// 					<div
// 						className={`flex flex-row items-end justify-center w-6 pt-0.5 px-0.5 bg-green-700 aspect-square rounded-full outline outline-2 ${
// 							thread.agents?.length > 1
// 								? "outline-white"
// 								: "outline-gray-900/10"
// 						} overflow-hidden`}>
// 						<img
// 							src={AvatarProfilePicture}
// 							className={`w-auto aspect-square`}
// 						/>
// 					</div>
// 					{index === array?.length - 1 &&
// 						thread.agents?.length > 3 && (
// 							<div className="absolute flex inset-0 pl-[5.5px] pb-[1px] items-center justify-center text-sm font-semibold text-white bg-gray-400 bg-opacity-50 rounded-full">
// 								+
// 							</div>
// 						)}
// 				</div>
// 			))}
// 	</div>
// </div>
// {thread.agents?.length > 1 ? (
// 	<h1 className="text-md font-bold text-gray-900">
// 		{thread.agents?.length}
// 	</h1>
// ) : (
// 	<h1 className="text-sm font-bold text-gray-900">
// 		{thread.agents?.[0]}
// 	</h1>
// )}
// </div>