import React, { useEffect, useRef, useState, useContext } from "react";
import DashboardHeader from "./DashboardHeader";
import AvatarController from "../Brochureware/AvatarAnimation/AvatarController";
import CreateThread from "./CreateThread2";
import { PlusIcon } from "./SVG_icons";
import StatusGraphic from "./CreateAgentForms/StatusGraphic";
import "../utils/scrollbar.css";

import { AvatarWalkingForwards } from "../Brochureware/AvatarAnimation/AvatarWalkForwards";
import openai_logo from "../../assets/openai-logos/PNGs/openai-logomark.png";

const { v4: uuidv4 } = require("uuid");

const test_agents = [
	{
		id: uuidv4(),
		name: "Colin",
		model: {
			id: "gpt-4",
			name: "GPT-4",
			avatar: openai_logo,
			status: "",
			description:
				"GPT-4 is a language model that uses deep learning to produce human-like text. It takes in a prompt, and attempts to complete it.",
			capabilities: {
				memory: true,
				training: false,
				communication: true,
				plugins: true,
			},
		},
		avatar: {
			uid: uuidv4(),
			id: "BASIC",
			animated: <AvatarWalkingForwards status={"forwards"} avatar={"VODAFONE"} />,
			static: <AvatarWalkingForwards status={"still"} avatar={"VODAFONE"} />,
			sex: {
				id: "male",
				pronouns: {
					pronoun: "he",
					objective: "him",
					possessive: "his",
				},
			},
		},
		setup: {
			agent: "COMPLETED",
			memory: "COMPLETED",
			training: "COMPLETED",
			connect: "COMPLETED",
		},
	},
	{
		id: uuidv4(),
		name: "Colin",
		model: {
			id: "gpt-4",
			name: "GPT-4",
			avatar: openai_logo,
			status: "",
			description:
				"GPT-4 is a language model that uses deep learning to produce human-like text. It takes in a prompt, and attempts to complete it.",
			capabilities: {
				memory: true,
				training: false,
				communication: true,
				plugins: true,
			},
		},
		avatar: {
			uid: uuidv4(),
			id: "BASIC",
			animated: <AvatarWalkingForwards status={"forwards"} />,
			static: <AvatarWalkingForwards status={"still"} />,
			sex: {
				id: "male",
				pronouns: {
					pronoun: "he",
					objective: "him",
					possessive: "his",
				},
			},
		},
		setup: {
			agent: "COMPLETED",
			memory: "COMPLETED",
			training: "COMPLETED",
			connect: "COMPLETED",
		},
	},
	{
		id: uuidv4(),
		name: "Colin",
		model: {
			id: "gpt-4",
			name: "GPT-4",
			avatar: openai_logo,
			status: "",
			description:
				"GPT-4 is a language model that uses deep learning to produce human-like text. It takes in a prompt, and attempts to complete it.",
			capabilities: {
				memory: true,
				training: false,
				communication: true,
				plugins: true,
			},
		},
		avatar: {
			uid: uuidv4(),
			id: "BASIC",
			animated: <AvatarWalkingForwards status={"forwards"} />,
			static: <AvatarWalkingForwards status={"still"} />,
			sex: {
				id: "male",
				pronouns: {
					pronoun: "he",
					objective: "him",
					possessive: "his",
				},
			},
		},
		setup: {
			agent: "COMPLETED",
			memory: "COMPLETED",
			training: "COMPLETED",
			connect: "COMPLETED",
		},
	},
];

export default function Agents() {
	// Header height
	const HeaderHeight = 8;

	// Agent grid sizing
	const agentRef = useRef(null);
	const [agentWidth, setAgentWidth] = useState(0);
	const [agentHeight, setAgentHeight] = useState(0);
	useEffect(() => {
		setAgentWidth(agentRef.current.offsetWidth);
		setAgentHeight(agentRef.current.offsetHeight);
	}, []);

	// Avatar grid sizing
	const ref = useRef(null);
	const [width, setWidth] = useState(0);
	const [height, setHeight] = useState(0);
	useEffect(() => {
		setWidth(ref.current.offsetWidth);
		setHeight(ref.current.offsetHeight);
	}, []);

	// Avatar hover state
	const [avatarIdHovered, setAvatarIdHovered] = useState(null);

	const [selectedNavButton, setSelectedNavButton] = useState("Agents");

	const agentsRef = useRef(null);
	const modelsRef = useRef(null);
	const pluginsRef = useRef(null);
	const memoryRef = useRef(null);
	const [widths, setWidths] = useState({
		Agents: 0,
		Models: 0,
		"Plug-Ins": 0,
		Memory: 0,
	});
	const [heights, setHeights] = useState({
		Agents: 0,
		Models: 0,
		"Plug-Ins": 0,
		Memory: 0,
	});

	// useEffect(() => {
	// 	setWidths({
	// 		Agents: agentsRef.current.offsetWidth,
	// 		Models: modelsRef.current.offsetWidth,
	// 		"Plug-Ins": pluginsRef.current.offsetWidth,
	// 		Memory: memoryRef.current.offsetWidth,
	// 	});
	// 	setHeights({
	// 		Agents: agentsRef.current.offsetHeight,
	// 		Models: modelsRef.current.offsetHeight,
	// 		"Plug-Ins": pluginsRef.current.offsetHeight,
	// 		Memory: memoryRef.current.offsetHeight,
	// 	});
	// }, []);

	// const [
	// 	selectedNavUnderlineBorderLeftMargin,
	// 	setSelectedNavUnderlineBorderLeftMargin,
	// ] = useState(0);

	// useEffect(() => {
	// 	// Update the selectedNavUnderlineBorderLeftMargin to be the sum of the widths of the nav buttons to the left of the selected one plus the spacing between them
	// 	let sum = 20;
	// 	for (let i = 0; i < Object.keys(widths).length; i++) {
	// 		if (Object.keys(widths)[i] === selectedNavButton) {
	// 			break;
	// 		}
	// 		sum += widths[Object.keys(widths)[i]];
	// 		sum += 32;
	// 	}
	// 	setSelectedNavUnderlineBorderLeftMargin(
	// 		sum + (Object.keys(widths).length - 1) * 4,
	// 	);
	// }, [selectedNavButton]);

	return (
		<div
			className="relative bg-transparent"
			// style={{ height: `${100 - HeaderHeight}vh` }}
		>
			{/* HEADER */}
			{/* <DashboardHeader HeaderHeight={HeaderHeight} /> */}
			{/* CONTENT */}

			{/* LEFT */}
			<div
				className="absolute top-0 left-0 w-1/2 bg-white h-full text-black flex flex-col"
				style={{
					height: `${100}vh`,
					marginTop: `${0}vh`,
				}}
			>

				{/* <h1 className="text-xl font-semibold text-gray-900 px-8 pt-6">
					Threads
				</h1>
				<h1 className="px-8 text-sm font-medium text-gray-500 pt-1">
					Proprietary. Open-source. Custom. Use any model you want.
				</h1>
				<div className="relative px-8 py-0 border-b border-gray-200 w-full pt-2" />

				<div className="px-12">
					<CreateThread />
				</div> */}


				<h1 className="text-xl font-semibold text-gray-900 px-8 pt-6">
					Your team
				</h1>
				<h1 className="px-8 text-sm font-medium text-gray-500 pt-1">
					Proprietary. Open-source. Custom. Use any model you want.
				</h1>
				
				<div className="relative px-8 py-0 border-b border-gray-200 w-full pt-2" />

				{/* Agent grid */}
				<div className="overflow-y-auto flex-grow custom-scrollbar">
					<div className="w-full grid grid-cols-4 gap-4 justify-between p-8">
						{test_agents.map((agent) => (
							<button
								ref={agentRef}
								// onClick={(e) => handleAgentClick(e, agent)}
								onMouseEnter={() =>
									setAvatarIdHovered(agent.id)
								}
								onMouseLeave={() => setAvatarIdHovered(null)}
								className="flex flex-none flex-col items-center rounded-md border border-gray-900/10 shadow-[0_0px_8px_-1px_rgba(0,0,0,0.1)] hover:bg-gray-50 py-4 px-8">
								<div className="flex items-center justify-center">
									{avatarIdHovered == agent.id ? (
										<div className="mt-1">
											{agent.avatar.animated}
										</div>
									) : (
										<div className="mt-1">
											{agent.avatar.static}
										</div>
									)}
								</div>

								<p className="mt-3 font-semibold text-gray-900">
									{agent.name}
								</p>

								<p className="text-xs font-sembiold text-gray-500">
									{agent.model.name}
								</p>

								<div className="mt-3 flex items-center justify-center mt-2">
									<StatusGraphic agent={agent} />
								</div>
							</button>
						))}
						{Array.from(
							{
								length:
									test_agents.length < 12
										? 12 - test_agents.length
										: 1,
							},
							(_, index) => (
								<button
									key={index}
									// onClick={(e) => handleAgentClick(e, agent)}
									onMouseEnter={() =>
										setAvatarIdHovered("space-" + index)
									}
									onMouseLeave={() =>
										setAvatarIdHovered(null)
									}
									className="flex flex-none flex-col items-center justify-center rounded-md border border-gray-900/5 shadow-inner bg-gray-50 py-4 px-8"
									style={{
										width: `${agentWidth}px`,
										height: `${agentHeight}px`,
									}}>
									{avatarIdHovered == "space-" + index && (
										<PlusIcon size="6" color="gray" />
									)}
								</button>
							),
						)}
					</div>
				</div>
			</div>

			{/* RIGHT */}
			<div
				className="fixed top-0 right-0 h-full w-1/2 shadow-[inset_0_0px_50px_20px_#9ca3af]"
				style={{
					height: `${100}vh`,
					marginTop: `${0}vh`,
				}}>
				<div className="relative h-full absolute top-0 left-0 right-0">
					<div className={`relative h-full`}>
						{/* BACKGROUND */}
						<div className="w-full h-full ring-1 ring-gray-900/10">
							<svg
								className="absolute inset-0 h-full w-full stroke-blue-900/30 [mask-image:radial-gradient(100%_100%_at_bottom_right,#FFFFFFFF,#FFFFFF8C)]"
								aria-hidden="true">
								<defs>
									<pattern
										id="83fd4e5a-9d52-42fc-97b6-718e5d7ee527"
										width={200}
										height={200}
										x="100%"
										y={-1}
										patternUnits="userSpaceOnUse">
										<path
											d="M130 200V.5M.5 .5H200"
											strokeDasharray="6,6"
											strokeWidth="1.2"
											fill="none"
										/>
									</pattern>
								</defs>
								<rect
									width="100%"
									height="100%"
									strokeWidth={0}
									fill="#f9fafb"
								/>
								<svg
									x="100%"
									y={-1}
									className="overflow-visible fill-gray-100">
									<path
										d="M-470.5 0h201v201h-201Z"
										strokeWidth={0}
									/>
								</svg>
								<svg
									x="100%"
									y={-1}
									className="overflow-visible fill-gray-100">
									<path
										d="M-470.5 0h201v201h-201Z"
										strokeWidth={0}
									/>
								</svg>
								<rect
									width="100%"
									height="100%"
									strokeWidth={0}
									fill="url(#83fd4e5a-9d52-42fc-97b6-718e5d7ee527)"
								/>
							</svg>
						</div>

						{/* Avatar Controller*/}
						<div
							ref={ref}
							className="absolute left-0 top-0 right-0 z-20 h-full">
							<div className="px-6 lg:px-8 mx-auto max-w-xl">
								<AvatarController
									containerWidth={width}
									containerHeight={height}
									agents_input={test_agents}
								/>
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
	);
}



// <div className="fixed top-0 right-0">
// 				<button
// 					onClick={() => {
// 						history.push('/threads');
// 					}}
// 					className="flex flex-row space-x-2 items-center justify-center py-2 px-4 rounded-md bg-brand-sd hover:opacity-70">
// 					{/* Add icon */}
// 					<ThreadsIcon color="white" />
// 					{/* Add text */}
// 					<h1 className="text-sm text-white font-semibold">
// 						Create thread
// 					</h1>
// 				</button>
// 			</div>




// <div className="relative px-8 py-0 border-b border-gray-200 w-full pt-2">
// 					<div className="relative flex flex-row space-x-2 w-full justify-between items-center">
// 						{/* NAV BAR - LEFT SIDE UNDER HEADER */}
// 						<div className="flex flex-row items-center justify-items-start justify-start space-x-8">
// 							<button
// 								ref={agentsRef}
// 								onClick={() => setSelectedNavButton("Agents")}
// 								className={`text-sm font-semibold ${
// 									selectedNavButton === "Agents"
// 										? "text-orange-600/80"
// 										: "text-gray-900"
// 								}`}>
// 								Agents
// 							</button>
// 							<button
// 								ref={modelsRef}
// 								onClick={() => setSelectedNavButton("Models")}
// 								className={`text-sm font-semibold ${
// 									selectedNavButton === "Models"
// 										? "text-orange-600/80"
// 										: "text-gray-900"
// 								}`}>
// 								Models
// 							</button>
// 							<button
// 								ref={pluginsRef}
// 								onClick={() => setSelectedNavButton("Plug-Ins")}
// 								className={`text-sm font-semibold ${
// 									selectedNavButton === "Plug-Ins"
// 										? "text-orange-600/80"
// 										: "text-gray-900"
// 								}`}>
// 								Plug-Ins
// 							</button>
// 							<button
// 								ref={memoryRef}
// 								onClick={() => setSelectedNavButton("Memory")}
// 								className={`text-sm font-semibold ${
// 									selectedNavButton === "Memory"
// 										? "text-orange-600/80"
// 										: "text-gray-900"
// 								}`}>
// 								Memory
// 							</button>
// 						</div>
// 						{/* RIGHT SIDE */}
// 						<div className="flex flex-row items-center justify-start space-x-4 pb-2">
// 							{/* NEW AGENT BUTTON */}
// 							<button
// 								// onClick={() => setNewThread(true)}
// 								className="flex flex-row space-x-2 items-center justify-center py-2 px-4 rounded-md bg-orange-600/80 hover:opacity-70">
// 								{/* Add icon */}
// 								<PlusIcon color="white" size="4" />
// 								{/* Add text */}
// 								<h1 className="text-sm text-white font-semibold">
// 									Create agent
// 								</h1>
// 							</button>
// 						</div>
// 					</div>
// 					{/* Border than underlines nav bar - left margin calculated by selected item and width */}
// 					<div
// 						className="absolute bottom-0 left-0 z-10 h-[3px] translate-y-1/2 bg-orange-600/80 transition-all duration-300 ease-in-out"
// 						style={{
// 							width: `${widths[selectedNavButton] + 12}px`,
// 							transform: `translateX(${-5}px)`,
// 							marginLeft: `${selectedNavUnderlineBorderLeftMargin}px`,
// 						}}></div>
// 				</div>