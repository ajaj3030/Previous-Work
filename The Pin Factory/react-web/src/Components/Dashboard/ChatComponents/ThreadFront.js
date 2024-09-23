import React, { useEffect, useState } from "react";
import AvatarProfilePicture from "../../../assets/baby-animation/profile_picture.png";
import {
	ArrowStalk,
	ArrowHead,
} from "../../Brochureware/Header/LogInButton.js";
import { CheckIcon } from "../SVG_icons.js";
import ProfilePicture from "../../../assets/test_pp/IMG_2109.PNG";
import { TrashIcon } from "@heroicons/react/24/outline";

export default function ThreadFront({
	thread,
	index,
	setThread,
	setEnterThread,
	setNewThread,
	setShowDeleteThreadButton,
	showDeleteThreadButton,
	setThreadsToDisplay,
	removeThread,
}) {
	const [buttonHovered, setButtonHovered] = useState(false);
	const [childHovered, setChildHovered] = useState(false);
	const handleHoverChange = (isHovered) => {
		setChildHovered(isHovered);
	};

	useEffect(() => {
		console.log(thread);
	}, [thread]);

	return (
		<button
			onMouseEnter={() => setButtonHovered(true)}
			onMouseLeave={() => setButtonHovered(false)}
			onClick={() => {
				setThread(thread);
				setEnterThread(true);
				setNewThread(false);
			}}
			// onClick={() => console.log(thread.name)}
			className={`flex flex-row w-full items-start h-48 justify-between rounded-md border py-3 focus:outline-none group px-4
        ${
			thread.lastMessage.sender !== "User" &&
			thread.lastMessage.state === "UNREAD"
				? // if new activity
				  buttonHovered
					? "bg-white shadow-[0_0px_4px_-1px_rgba(0,0,0,0.1)] border-gray-900/20"
					: "bg-white shadow-[0_0px_8px_-1px_rgba(0,0,0,0.1)] border-gray-900/20"
				: // if no new activity
				buttonHovered
				? "bg-gray-50 border-gray-900/10"
				: "bg-gray-50 border-gray-900/10"
		}`}>
			<div className="flex flex-row w-full h-full justify-start items-start gap-x-4 items-start mt-1">
				{/* LEFT IMAGE */}
				<div className="flex mt-1 flex-row items-end justify-center w-8 pt-1 px-1 bg-emerald-600 aspect-square rounded-full outline outline-2 outline-gray-900/10 overflow-hidden">
					<img
						src={AvatarProfilePicture}
						className={`w-auto aspect-square`}
					/>
				</div>

				{/* MAIN CONTENT */}
				<div
					className={`flex flex-col justify-between w-full h-full text-start space-y-3`}>
					{/* TOP 2 OF MAIN CONTENT */}

					<div className={`flex flex-col space-y-1 text-start`}>
						{/* TOP LINE */}
						<div
							className={`flex items-center justify-between mr-2`}>
							<div
								className={`flex-grow flex items-center space-x-4 ${
									buttonHovered ? "opacity-50" : ""
								}`}>
								{/* NAME */}
								<div className="flex items-center">
									<p
										className={`text-sm font-semibold leading-6 text-gray-900 line-clamp-1`}>
										{thread.threadName}
									</p>
									{/* MESSAGES UNREAD */}
									{thread.lastMessage.sender !== "User" &&
										thread.lastMessage.state ===
											"UNREAD" && (
											<div className="ml-[8px] flex items-center justify-center w-[6px] h-[6px] rounded-full bg-orange-600/90" />
										)}
								</div>
								{/* RIGHT ARROW */}
								<div className={`flex items-center`}>
									<ArrowStalk
										className={`w-3 h-5 group-hover:scale-x-150`}
									/>
									<ArrowHead
										className={`w-2 h-3 -ml-1.5 group-hover:translate-x-0.5`}
									/>
								</div>
							</div>
							{showDeleteThreadButton && (
								<button
									className=" hover:text-orange-600 rounded-full text-orange-400 text-md"
									onClick={(e) => {
										e.stopPropagation(); // This prevents the button's onClick event from triggering
										removeThread(index, thread.threadId); // index is the index of the thread in the threadsToDisplay array
									}}>
									<TrashIcon className="h-5 w-5" />
								</button>
							)}
						</div>

						{/* MAIN TEXT BODY */}
						<p
							className={`text-sm leading-4.5 text-gray-500/90 line-clamp-3`}>
							{thread.lastMessage.content || "No messages yet"}
						</p>
						{thread.lastMessage.sender === "User" ? (
							<div>
								{thread.lastMessage.state === "READ" ? (
									<CheckIcon
										height="14"
										width="14"
										strokeWidth="1.5"
										color="#60a5fa"
									/>
								) : (
									<CheckIcon
										height="14"
										width="14"
										strokeWidth="1.5"
										color="gray"
									/>
								)}
							</div>
						) : (
							<div>
								<p
									className={`text-xs font-semibold leading-6 text-gray-500`}>
									{/* change the real sender */}
									{thread.lastMessage.senderName}
								</p>
							</div>
						)}
					</div>

					{/* BOTTOM OF MAIN */}
					<div className="flex flex-row items-center justify-start justify-items-start w-full">
						{/* HUMANS */}
						<div className="flex flex-row items-center justify-start space-x-1 w-1/3">
							<p className="text-xs font-medium leading-6 text-gray-500/90">
								Humans
							</p>
							<HumanBubble
								humans={thread.humans}
								onHoverChange={handleHoverChange}
							/>
						</div>
						{/* AGENTS */}
						<div className="flex flex-row items-center justify-start space-x-1 w-1/3">
							<p className="text-xs font-medium leading-6 text-gray-500/90">
								Agents
							</p>
							<AgentsBubble
								agents={thread.agents}
								onHoverChange={handleHoverChange}
							/>
						</div>
						{/* NUMBER OF CHAINS */}
						<div className="flex flex-row items-center justify-start space-x-1 w-1/3">
							<p className="text-xs font-medium leading-6 text-gray-500/90">
								Chains
							</p>
							<p className="text-xs font-bold leading-6 text-gray-700">
								4
							</p>
						</div>
					</div>
				</div>
			</div>
		</button>
	);
}

function HumanBubble({ humans, onHoverChange }) {
	const [bubbleIsOpen, setBubbleIsOpen] = useState(false);
	const [bubbleIsTransitioning, setBubbleIsTransitioning] = useState(false);

	const handleTransitionEnd = () => {
		if (bubbleIsTransitioning) setBubbleIsTransitioning(false);
	};

	const handleClick = (event) => {
		event.stopPropagation();
	};

	return (
		<div
			onMouseEnter={() => {
				setBubbleIsOpen(true);
				onHoverChange(true);
			}}
			onMouseLeave={() => {
				setBubbleIsOpen(false);
				setBubbleIsTransitioning(true);
				onHoverChange(false);
			}}
			onClick={handleClick}
			className="cursor-default flex flex-row relative">
			<div className="flex -space-x-2">
				{humans.slice(0, 3).map((human, index, array) => (
					<div
						key={human.humanId}
						className={`relative ${
							index === 0 ? "z-20" : index === 1 ? "z-10" : "z-0"
						} place-items-center`}>
						<img
							src={ProfilePicture}
							alt="human name"
							className={`h-5 w-5 rounded-full outline ${
								humans.length === 1
									? "outline-1 outline-gray-900/10"
									: "outline-2 outline-white"
							}`}
						/>
						{index === array.length - 1 && humans.length > 3 && (
							<div className="absolute flex inset-0 pl-[5.5px] pb-[1px] items-center justify-center text-sm font-semibold text-white bg-gray-400 bg-opacity-50 rounded-full">
								+
							</div>
						)}
					</div>
				))}
			</div>
			{(bubbleIsOpen || bubbleIsTransitioning) && (
				<div
					className={`absolute left-full pl-2 -mt-4 z-50 max-w-min ${
						bubbleIsOpen
							? "opacity-100 transition-opacity duration-300 ease-out"
							: "opacity-0 transition-opacity duration-150 ease-in"
					}`}
					onTransitionEnd={handleTransitionEnd}>
					<div className="w-64 overflow-hidden rounded-md bg-white border border-gray-900/10 divide-y divide-gray-900/10 shadow-[0_10px_50px_-10px_rgba(0,0,0,0.3)]">
						<p className="text-sm font-semibold text-gray-600 py-2 px-3">
							{humans.length} Humans
						</p>
						<div className="flex flex-col space-y-1.5 px-3 py-2 max-h-48 overflow-y-auto overscroll-contain">
							{humans.map((human, index, array) => (
								<div
									className="flex flex-row items-center justify-start space-x-2"
									key={human.humanId}>
									<img
										src={ProfilePicture}
										alt="human name"
										className="h-6 w-6 rounded-full outline outline-2 outline-gray-900/10"
									/>
									<p className="text-xs font-medium text-gray-900">
										{human.humanName}
									</p>
								</div>
							))}
						</div>
					</div>
				</div>
			)}
		</div>
	);
}

function AgentsBubble({ agents, onHoverChange }) {
	const [bubbleIsOpen, setBubbleIsOpen] = useState(false);
	const [bubbleIsTransitioning, setBubbleIsTransitioning] = useState(false);

	const handleTransitionEnd = () => {
		if (bubbleIsTransitioning) setBubbleIsTransitioning(false);
	};

	const handleClick = (event) => {
		event.stopPropagation();
	};

	return (
		<div
			onMouseEnter={() => {
				setBubbleIsOpen(true);
				onHoverChange(true);
			}}
			onMouseLeave={() => {
				setBubbleIsOpen(false);
				setBubbleIsTransitioning(true);
				onHoverChange(false);
			}}
			onClick={handleClick}
			className="cursor-default flex flex-row relative">
			<div className="flex -space-x-2">
				{agents.slice(0, 3).map((agent, index, array) => (
					<div
						key={agent.id}
						className={`relative ${
							index === 0 ? "z-20" : index === 1 ? "z-10" : "z-0"
						} place-items-center`}>
						{/* LEFT IMAGE */}
						<div
							className={`flex flex-row items-end justify-center w-5 pt-0.5 px-0.5 bg-orange-400 aspect-square rounded-full outline ${
								agents.length === 1
									? "outline-1 outline-gray-900/10"
									: "outline-2 outline-white"
							} overflow-hidden`}>
							<img
								src={AvatarProfilePicture}
								className={`w-auto aspect-square`}
							/>
						</div>
						{index === array.length - 1 && agents.length > 3 && (
							<div className="absolute flex inset-0 pl-[5.5px] pb-[1px] items-center justify-center text-sm font-semibold text-white bg-gray-400 bg-opacity-50 rounded-full">
								+
							</div>
						)}
					</div>
				))}
			</div>
			{(bubbleIsOpen || bubbleIsTransitioning) && (
				<div
					className={`absolute left-full pl-2 -mt-4 z-50 max-w-min ${
						bubbleIsOpen
							? "opacity-100 transition-opacity duration-300 ease-out"
							: "opacity-0 transition-opacity duration-150 ease-in"
					}`}
					onTransitionEnd={handleTransitionEnd}>
					<div className="w-64 overflow-hidden rounded-md bg-white border border-gray-900/10 divide-y divide-gray-900/10 shadow-[0_10px_50px_-10px_rgba(0,0,0,0.3)]">
						<p className="text-sm font-semibold text-gray-600 py-2 px-3">
							{agents.length} Agents
						</p>
						<div className="flex flex-col space-y-1.5 px-3 py-2 max-h-48 overflow-y-auto overscroll-contain">
							{agents.map((agent, index, array) => (
								<div
									className="flex flex-row items-center justify-start space-x-2"
									key={agent.id}>
									<div className="flex flex-row items-end justify-center w-6 pt-0.5 px-0.5 bg-orange-400 aspect-square rounded-full outline outline-2 outline-gray-900/10 overflow-hidden">
										<img
											src={AvatarProfilePicture}
											className={`w-auto aspect-square`}
										/>
									</div>
									<p className="text-xs font-medium text-gray-900">
										{agent}
									</p>
								</div>
							))}
						</div>
					</div>
				</div>
			)}
		</div>
	);
}
